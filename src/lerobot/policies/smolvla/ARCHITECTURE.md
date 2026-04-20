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

For the MetaWorld peg-insert-side-v3 setup, M is not fixed — it is the padded prefix length for the longest sequence in the batch. Its constituent parts are: 258 image tokens (1024 patches divided by connector scale factor 4, plus 2 special tokens), up to 48 language tokens padded to the longest in the batch, and 1 state token. In practice M is approximately 270–280 for the task description used. The exact value depends on the tokenized length of the task string and can be confirmed by printing the prefix output shape during a forward pass.

```
VLM prefix outputs z = [z_1, ..., z_M]     shape: (batch, M, H)
         │
         ▼
   input_proj: Linear(H, d_model)           project to RLT internal dim
         │
         ▼
   Positional Encoding — sinusoidal fixed (standard sin/cos formula, registered as buffer, not learned)
         │
         │   Append learned <rl> token e_rl to the END
         ▼
   [z_1, z_2, ..., z_M, e_rl]              shape: (batch, M+1, d_model)
         │
         ▼
   ┌─────────────────────┐
   │   RLT Encoder       │  3 layers, nhead=8, d_model=2048, Pre-LN
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
   │   RLT Decoder       │  3 layers, nhead=8, d_model=2048, Pre-LN
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
| `d_model` | 2048 |
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

TransformerRLT is trained to reconstruct VLM prefix embeddings from `z_rl`. The VLA action head is completely skipped — no noisy actions, no expert stream. Training runs for 10,000 gradient steps as set in the training script. Uses the same MetaWorld peg-insert-side-v3 demo dataset as Phase 1.

**Phase 2 is independent of Phase 1.** The training script (`scripts/train_rlt_token.sh`) does NOT pass `--policy.path=...`, so Phase 2 starts from the raw pretrained VLM weights (`load_vlm_weights=true`) with a randomly-initialized action head and `state_proj`. Only the TransformerRLT is updated during Phase 2. The action head remains random until Phase 3 patches it in from a separate Phase 1 checkpoint (see Checkpoint Handoff below).

Note: `state_proj` is also random in Phase 2 and gets replaced by Phase 1's trained weights in Phase 3. This causes a small distribution shift on the state token between phases; it is accepted as a second-order effect because the state token is ~1/270 of the prefix on MetaWorld.

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
                                      post-final-layer-norm output (same as HuggingFace last_hidden_state)
        │
  detach() if train_expert_only    ← [CORRECTION NEEDED] per RLT paper Eq 2, stop-gradient should be
                                      unconditional. In practice this is always True (train_rlt_token.sh
                                      always sets train_expert_only=true), so the bug is dormant but the
                                      code should be fixed to detach unconditionally regardless of the flag.
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

### Phase 3 — Actor-Critic RL training (external, see `ACTOR_CRITIC_DESIGN.md`)

Everything in SmolVLA + TransformerRLT is **frozen**. A separate lightweight actor-critic (TD3) is trained using `z_rl` as the state representation.

```
Observation (images, language, state)
        │
        ▼
VLAFlowMatching.get_rl_state()     ← new method added to modeling_smolvla.py
        │
        ├─ embed_prefix() → VLM prefix-only forward → prefix_out (B, M, 2048)
        └─ TransformerRLT.encode(prefix_out) → z_rl (B, 512)

z_rl (512) + proprio (32) → rl_state (544)    ← stored in ReplayBuffer

rl_state + VLA_ref_action (10×4=40) → RLTActor MLP → action_mean (10×4)
                                                   + fixed_noise  → executed_action

rl_state + executed_action → CriticEnsemble → Q1, Q2
```

### Gradient flow summary

| Component | Phase 1 (action) | Phase 2 (reconstruction) | Phase 3 (actor-critic RL) |
|---|---|---|---|
| Vision encoder | frozen (default) | frozen (default) | frozen |
| VLM text model | frozen (default) | frozen — train_expert_only=true always set in training script | frozen |
| Action expert | trained | not used | frozen |
| Action projections | trained | not used | frozen |
| TransformerRLT | not used | trained | frozen |
| RLTActor | not used | not used | trained |
| CriticEnsemble | not used | not used | trained |

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

## Checkpoint Handoff Between Phases

**Phase 1 → Phase 2**: no handoff. Phase 2 starts from the pretrained VLM (via `load_vlm_weights=true`) with a randomly-initialized action head. Only TransformerRLT is trained.

**Phase 2 → Phase 3**: **two checkpoints required.** Phase 3 (`src/lerobot/scripts/train_actor_critic_rlt.py`) takes both `--vla_checkpoint` (Phase 1) and `--rlt_checkpoint` (Phase 2). It loads Phase 2 as the base policy (for VLM + TransformerRLT) and then patches in the action-head modules (`action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`, `state_proj`) from Phase 1 so the policy can also produce VLA reference actions for regularization. Everything is then frozen; only RLTActor and RLTCritic are trained.

---

| File | Role |
|---|---|
| [configuration_smolvla.py](configuration_smolvla.py) | All hyperparameters including RLT config and `training_mode` |
| [modeling_smolvla.py](modeling_smolvla.py) | `SmolVLAPolicy` (outer wrapper) and `VLAFlowMatching` (core model); training branching logic |
| [smolvlm_with_expert.py](smolvlm_with_expert.py) | `SmolVLMWithExpertModel`: interleaved VLM + action expert transformer |
| [transformer_rlt.py](transformer_rlt.py) | `TransformerRLT`: encoder-decoder bottleneck for learning `z_rl` |
| [processor_smolvla.py](processor_smolvla.py) | Image and text preprocessing |
