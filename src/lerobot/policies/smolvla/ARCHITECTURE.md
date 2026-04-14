# SmolVLA + TransformerRLT Architecture

## Overview

SmolVLA is a Vision-Language-Action (VLA) model that combines a pretrained vision-language model (VLM) with a lightweight action expert to predict robot actions. TransformerRLT is a small encoder-decoder transformer added on top of SmolVLA to learn a compact **RL token** (`z_rl`) — a single embedding that summarizes the full perceptual context and can be used as a state representation for reinforcement learning.

---

## SmolVLA Architecture

SmolVLA is composed of two interleaved transformer stacks that run in parallel over every layer:

```
Images ──► SigLIP Vision Encoder ──► Connector/Resampler ──► image tokens
                                                                    │
Language tokens ──► Embedding layer ──► language tokens             │
                                                                    │
State ──► Linear(state_dim, vlm_hidden) ──► state token             │
                                                                    │
                        ┌───────────────────────────────────────────┘
                        │            PREFIX (VLM stream)
                        ▼
          ┌─────────────────────────┐
          │   VLM Text Model        │  SmolVLM2-500M backbone
          │   (Llama-style)         │  hidden_size H (e.g. 2048)
          │   16 transformer layers │  bfloat16
          │   RoPE + RMSNorm        │
          └────────────┬────────────┘
                       │ KV cache / cross-attention
                       ▼
          ┌─────────────────────────┐        Noisy actions x_t ──► action_in_proj
          │   Action Expert         │  ◄───── Timestep t ──────────► time MLP
          │   (smaller Llama)       │                    SUFFIX (expert stream)
          │   hidden = 0.75 × H     │
          │   same depth as VLM     │
          └────────────┬────────────┘
                       │
                  action_out_proj
                       │
                       ▼
               Predicted velocity v_t
               (batch, chunk_size, action_dim)
```

### Key components

| Component | Details |
|---|---|
| **Backbone** | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| **Vision encoder** | SigLIP, images normalized to `[-1, 1]`, resized to `512×512` |
| **Connector** | Learned resampler projecting patch features to the text hidden dim |
| **VLM text model** | Llama-style transformer, 16 layers, hidden size `H` (≈2048), bfloat16 |
| **Action expert** | Smaller Llama-style model, hidden size `0.75 × H`, same layer count |
| **Attention mode** | Interleaved self-attention (every 2 layers) and cross-attention (expert ← VLM) |
| **State projection** | `Linear(max_state_dim, H)` → 1 token appended to prefix |
| **Action input** | `Linear(max_action_dim, expert_hidden)` fused with sinusoidal timestep via MLP |
| **Action output** | `Linear(expert_hidden, max_action_dim)` |

### VLM + Expert joint forward

The two streams (prefix = VLM, suffix = expert) are processed together through each layer. The attention mask is structured so that:
- Prefix tokens (image, language, state) attend only among themselves (bidirectional within prefix).
- Suffix tokens (noisy actions) attend to the full prefix via cross-attention and to previous suffix positions causally.

This lets the expert condition on the rich VLM representations without re-running the VLM at every denoising step.

### Flow-matching training objective

SmolVLA is trained with **flow matching**: given a clean action `a`, a noise sample `ε`, and a random timestep `t ~ Beta(1.5, 1.0)`:

```
x_t = t·ε + (1−t)·a          # noisy action at time t
u_t = ε − a                   # target velocity
loss = MSE(v_t, u_t)          # predicted vs. target velocity
```

At inference, actions are denoised from pure noise over `num_steps=10` Euler steps.

---

## TransformerRLT Architecture

TransformerRLT is a lightweight encoder-decoder transformer that distills the VLM's `M` final-layer prefix token embeddings `z = {z_1, ..., z_M}` into a single **RL token** `z_rl`. The bottleneck design forces `z_rl` to encode all task-relevant information.

```
VLM prefix outputs z = [z_1, ..., z_M]     shape: (batch, M, H)
         │
         ▼
   input_proj: Linear(H, d_model)           project to RLT internal dim
         │
         ▼
   Positional Encoding (batch-first)
         │
         │   Append learned <rl> token e_rl to the END
         ▼
   [z_1, z_2, ..., z_M, e_rl]              shape: (batch, M+1, d_model)
         │
         ▼
   ┌─────────────────────┐
   │   RLT Encoder       │  3 layers, nhead=8, d_model=512, Pre-LN
   └──────────┬──────────┘
              │
         enc_out[:, -1, :]   ← last sequence position = z_rl
              │
              ▼
           z_rl               shape: (batch, d_model)      → used for RL
              │
              │  (used as cross-attention memory in decoder)
              │
   Teacher-forced decoder input:
   [BOS, z_1, z_2, ..., z_{M-1}]           shape: (batch, M, d_model)
   + causal mask
              │
              ▼
   ┌─────────────────────┐
   │   RLT Decoder       │  3 layers, nhead=8, d_model=512, Pre-LN
   │   cross-attn ← z_rl │  memory = z_rl (the bottleneck)
   └──────────┬──────────┘
              │
         output_proj: Linear(d_model, H)
              │
              ▼
          z_recon             shape: (batch, M, H)
```

### Why this works as a bottleneck

The decoder must reconstruct all `M` embeddings `z_recon ≈ z` using **only** `z_rl` as its cross-attention memory. If `z_rl` loses any information, reconstruction quality degrades. The MSE reconstruction loss therefore pushes `z_rl` to be a maximally informative summary of the full state-language context — a learned compression that is useful as a state representation for a downstream RL value function.

### RLT default hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | VLM hidden size `H` (set automatically from backbone config) |
| `d_model` | 512 |
| `nhead` | 8 |
| `num_encoder_layers` | 3 |
| `num_decoder_layers` | 3 |
| `dim_ff` | 2048 |
| `dropout` | 0.1 |

---

## Two-Phase Training

The two components are trained in **separate phases**, controlled by `SmolVLAConfig.training_mode`.

### Phase 1 — Action training (`training_mode="action"`)

Standard SmolVLA behavioral cloning via flow matching. TransformerRLT is not used.

```
Batch (images, language, state, actions)
        │
        ▼
embed_prefix()  ──►  embed_suffix(noisy_actions, t)
        │                        │
        └──────────┬─────────────┘
                   ▼
        VLM + Expert joint forward
                   │
             suffix_out[:, -chunk_size:]
                   │
            action_out_proj
                   │
              MSE(v_t, u_t)   ◄── training loss
```

Frozen components (default): VLM backbone (`train_expert_only=True`), vision encoder (`freeze_vision_encoder=True`).
Trained components (default): action expert, action projections, state projection.

### Phase 2 — RLT reconstruction training (`training_mode="reconstruction"`)

TransformerRLT is trained to reconstruct VLM prefix embeddings from `z_rl`. The VLA action head is completely skipped — no noisy actions, no expert stream.

```
Batch (images, language, state)    ← actions not needed
        │
        ▼
embed_prefix()
        │
        ▼
VLM prefix-only forward            ← no suffix/expert stream
        │
  prefix_out (batch, M, H)         ← VLM final-layer embeddings z
        │
  detach() if train_expert_only    ← freeze VLM; or allow gradients to finetune
        │
        ▼
TransformerRLT.forward(z)
        ├──► z_rl    (batch, d_model)
        └──► z_recon (batch, M, H)
        │
  masked MSE loss over valid (non-padding) positions:
  loss = Σ ||z_recon_i − z_i||² / num_valid_tokens
        │
        ▼
  backprop → update TransformerRLT weights
             (+ VLM weights if train_expert_only=False)
```

### Gradient flow summary

| Component | Phase 1 (action) | Phase 2 (reconstruction) |
|---|---|---|
| Vision encoder | frozen (default) | frozen (default) |
| VLM text model | frozen (default) | frozen if `train_expert_only=True`; trainable if `False` |
| Action expert | trained | not used |
| Action projections | trained | not used |
| TransformerRLT | not used | trained |

---

## Using `z_rl` for RL

After Phase 2, `z_rl` can be used as the state representation for a downstream RL value/policy head. At inference, call `TransformerRLT.encode(z)` to get `z_rl` without running the decoder:

```python
# Get VLM prefix embeddings
prefix_out = vlm_prefix_forward(images, language, state)

# Extract RL token (no decoder pass needed at inference)
z_rl = transformer_rlt.encode(prefix_out.float())   # (batch, d_model)

# Feed to downstream value head / RL policy
value = value_head(z_rl)
```

---

## File Map

| File | Role |
|---|---|
| [configuration_smolvla.py](configuration_smolvla.py) | All hyperparameters including RLT config and `training_mode` |
| [modeling_smolvla.py](modeling_smolvla.py) | `SmolVLAPolicy` (outer wrapper) and `VLAFlowMatching` (core model); training branching logic |
| [smolvlm_with_expert.py](smolvlm_with_expert.py) | `SmolVLMWithExpertModel`: interleaved VLM + action expert transformer |
| [transformer_rlt.py](transformer_rlt.py) | `TransformerRLT`: encoder-decoder bottleneck for learning `z_rl` |
| [processor_smolvla.py](processor_smolvla.py) | Image and text preprocessing |
