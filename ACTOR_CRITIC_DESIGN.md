# Actor-Critic (RLT Phase 3) Design

This document covers Phase 3 of the RLT pipeline: the TD3 actor-critic that is trained on top of the frozen SmolVLA + TransformerRLT.
Phases 1 and 2 (SmolVLA BC training and RLT encoder-decoder training) are covered in `src/lerobot/policies/smolvla/ARCHITECTURE.md`.

All implementation decisions are marked **[OUR DECISION]** — sweep those first if something breaks.

---

## 1. Paper Summary

The RL Token (RLT) paper from Physical Intelligence trains a small TD3 actor-critic on top of a frozen VLA. The key insight is that the VLA already solves most of the task; the actor-critic only needs to learn a small residual correction.

**RL Token.** A single compressed vector `z_rl` produced by a small encoder transformer over the VLA's final-layer token embeddings. Trained via an encoder-decoder reconstruction loss — the decoder must autoregressively reconstruct all VLA embeddings from `z_rl` alone, forcing it to be an information bottleneck. One token per task; not transferable across tasks.

**Actor.** Input is `concat(z_rl, proprio, vla_ref_chunk)`. The actor does not generate actions from scratch — it takes the VLA's predicted action chunk as input and learns to edit it. Output is a Gaussian over action chunks with a fixed (non-learned) variance. Reference action dropout at 50% forces the actor to maintain an independent action pathway when the VLA input is zeroed out.

**Critic.** Input is `concat(z_rl, proprio, executed_action_chunk)`. Estimates Q-value at the chunk level. Twin critics (TD3 ensemble); min of the two used for target Q to prevent overestimation. Slow-moving target networks for stable bootstrapping.

**Training loop.** Online: actor/critic update while the sim collects new episodes. G=5 gradient updates per environment step. Two critic updates per actor update (delayed actor updates, standard TD3). Replay buffer mixes VLA warmup rollouts, online RL rollouts, and human interventions.

**Reward.** Sparse binary — +1 at episode end on success, 0 everywhere else. No learned reward, no dense shaping. The β regularization term in the actor loss is computed fresh at training time from the current VLA inference, never stored in the replay buffer.

**Actor loss:** `-Q1(s, a) + β‖a − ã‖²`  
Per the paper, `ã` should be freshly sampled from the frozen VLA at training time. **[OUR DEVIATION — ACCEPTED]** We use the stored VLA reference action from the replay buffer rather than re-running the VLA at training time, because running full VLA inference (10 Euler denoising steps) across 256 batch samples per gradient update is prohibitively expensive. The stored reference was sampled at collection time. See Section 5 for full details.

**Critic loss:** `MSE(Q1(s,a), Q_target) + MSE(Q2(s,a), Q_target)`  
`Q_target = r + γ^C · min(Q1_tgt, Q2_tgt)(s', ã')`  
The `γ^C` discount accounts for the C steps consumed within the chunk before bootstrapping.

**Action chunking.** VLA produces H=50 step chunks; RL actor uses shorter C=10 step chunks for reactivity. Single-step RL (C=1) was tested as an ablation and failed — the value function cannot propagate sparse reward over hundreds of timesteps. The paper stride-2 subsamples transitions from each executed chunk to double effective data; our implementation stores one transition per chunk (see §2 deviations). Each stored transition includes both the executed action and the VLA reference action used at that chunk.

---

## 2. Our Implementation Deviations

| Paper | Our Adaptation | Reason |
|---|---|---|
| VLA is π0.6 | VLA is SmolVLA (SmolVLM2-500M) | Compute constraints |
| H=50 VLA chunks, C=10 RL chunks | Both H=C=10 — retrain VLA with `chunk_size=10` | Simplifies reference action: full VLA output = actor input, no truncation |
| Applied only to the "critical precision phase" with human handoff | Applied from episode start | Simpler for sim; no human in the loop |
| Replay buffer mixes VLA rollouts + online rollouts + human interventions | VLA warmup rollouts + online rollouts only | Sim context — oracle policy available if fallback needed |
| Actor β-loss uses freshly sampled VLA reference at training time | Uses stored VLA reference from replay buffer | Re-running VLA inference on 256 batch samples per gradient step is prohibitively expensive. The stored reference was sampled at collection time. |
| Stride-2 subsampling (5 transitions per C=10 chunk) | One transition per chunk | Zero-padded sub-chunks paired with the same full-chunk reward corrupt the Q-target (the critic sees different sub-chunk lengths all attributed the same reward). Lose ~5× data multiplication, keep correctness. |
| Ref-action dropout at rollout time | Dropout only at actor-loss time | Paper applies dropout at training time to force an unconditioned actor path; applying it also at rollout was an incidental duplication. Rollout always feeds the full VLA reference so exploration isn't noise-limited. |

---

## 3. Actor Architecture

The actor is a 2-layer MLP (hidden dim 256, SiLU activations) implemented using `MLP` imported from `lerobot.policies.sac.modeling_sac`, with a final `nn.Linear(256, C × action_dim)` output head.

Input: `concat(z_rl[2048], proprio[32], vla_ref_flat[40])` = 2120-dim  
Output: action chunk mean of shape `(C=10, action_dim=4)`, Gaussian noise added at rollout time.

`action_dim = 4` for Metaworld Sawyer (xyz + gripper). SmolVLA pads to `max_action_dim=32` internally but `predict_action_chunk` slices back to `config.action_feature.shape[0] = 4` before returning. The actor and critic always operate on real 4-dim actions.

Variance is fixed at 0.1 (std ≈ 0.316) **[OUR DECISION — sweep if exploration too narrow/wide]**. The MLP outputs the mean; Gaussian noise is added at rollout time only (not during critic/actor loss computation).

Reference action dropout is applied inside `compute_td3_actor_loss`, not at rollout time (see §2 deviations). At rollout and eval, the actor always receives the full VLA reference.

---

## 4. Critic Architecture

The critic uses two completely separate MLPs (`CriticHead` from `lerobot.policies.sac.modeling_sac`), initialized with different random seeds via `torch.random.fork_rng()`. Each head is a 2-layer MLP (hidden dim 256) with a final `Linear(256, 1)` scalar output.

Input: `concat(z_rl[2048], proprio[32], executed_action_flat[40])` = 2120-dim  
Output: scalar Q-value (chunk-level)

The critic takes the actor's **executed** action, not the VLA reference. `min(Q1_target, Q2_target)` is used for TD target computation. One frozen Polyak-updated copy per critic (τ = 0.005 **[OUR DECISION]**).

---

## 5. Loss Functions

**Critic loss** (every gradient step):
```
Q_target = stop_gradient(r + (1 - done) × γ^C × min(Q1_tgt(s', ã'), Q2_tgt(s', ã')))
L_critic  = MSE(Q1(s,a), Q_target) + MSE(Q2(s,a), Q_target)
```
γ = 0.99 **[OUR DECISION]**, C = 10 → γ^C ≈ 0.904. The `(1 - done)` term masks out the bootstrap on terminal transitions — when done=True, Q_target = r only, no future value is added. This is critical because the only non-zero reward occurs at the terminal step; bootstrapping past it produces incorrect Q-values. For the target actor at next states, the VLA reference input is set to zeros to avoid re-running the VLA for every next-state in the batch **[OUR DECISION — deviation, see Section 2]**.

**Actor loss** (every 2nd gradient step — delayed actor updates):
```
L_actor = −Q1(s, a) + β × ‖a − ã‖²
```
β = 0.1 **[OUR DECISION — sweep first if actor doesn't improve over VLA]**. **[OUR DEVIATION — ACCEPTED]** `ã` is retrieved from the stored replay buffer field (`complementary_info["vla_ref_action"]`), not freshly sampled from the VLA at training time, because of the cost of VLA inference at batch size 256. `Q1` only (not min) for the actor gradient, per TD3 convention.

---

## 6. State Representation & Replay Buffer

`rl_state = concat(z_rl[2048], proprio[32])` = 2080-dim. Pre-extracting `z_rl` at collection time avoids re-running the ~450M VLA+RLT at every replay sample — safe because `z_rl` is deterministic given the frozen weights.

Each stored transition (one per executed chunk):
```
rl_state:       2080-dim vector — concat of z_rl (2048) and proprio (32)
actor_action:   40-dim vector — the flattened action chunk executed by the actor (with noise), shape C×action_dim
reward:         scalar — 0.0 at all steps, 1.0 only at terminal success step
next_rl_state:  2080-dim vector — rl_state at the next chunk boundary (terminal obs if done)
done:           bool — True only at episode end; used to mask bootstrap in critic loss
vla_ref_action: 40-dim vector — VLA reference action for this chunk. Used by the actor β-loss
                at training time (stored rather than freshly sampled; see §2 deviations).
```

Note on proprio: MetaWorld agent_pos is 4-dimensional (xyz + gripper). The remaining 28 of the 32 proprio dimensions are literal zeros from zero-padding. This zero-padding is consistent between Phase 2 RLT training and Phase 3 RL rollouts.

Buffer capacity 10,000 chunk-transitions ≈ 2,000 episodes at ~5 chunks/episode **[OUR DECISION — double if critic overfits]**.

Use `ReplayBuffer` from `lerobot.rl.buffer` with `complementary_info` for `vla_ref_action`.

---

## 7. Training Loop

```
Warmup: 20 VLA episodes (use_actor=False) → pre-fill replay buffer
        During warmup, pure VLA actions are executed verbatim — no actor, no noise, no dropout.
        The VLA output is stored as both the executed action and the VLA reference in the transition.
        (~12 successes at 60% BC accuracy gives the critic an initial learning signal)

Online RL (1000 episodes):
  for each episode:
    collect_episode(env, vla, actor) → N chunk transitions (one per C=10-step chunk)
    NOTE: terminal transition (r=1, done=True) is stored with next_rl_state = the terminal
          observation returned by MetaworldEnv.step(). The env auto-resets internally but
          the reset obs is discarded; this is fine because (1-done) zeros out the bootstrap.

    for _ in range(G=5 × N):
      batch = replay.sample(256)

      # critic update — uses done flag to mask bootstrap at terminal steps
      critic_loss = compute_td3_critic_loss(batch, ...)   # Q_target uses (1-done) mask
      critic_optimizer.step()

      if global_step % 2 == 0:
        # actor update — uses stored vla_ref_action from batch (not fresh VLA inference)
        actor_loss = compute_td3_actor_loss(batch, ...)
        actor_optimizer.step()
        polyak_update(actor → target_actor, τ=0.005)
        polyak_update(critic → target_critic, τ=0.005)

      global_step += 1

  if episode % 50 == 0:
    evaluate(actor, n_episodes=10)   # actor.eval() called; no dropout, no noise, full VLA ref
```

Both optimizer lr = 3e-4 (Adam). G=5 from paper; delayed actor updates (2 critic : 1 actor) standard TD3.

---

## 8. Environment

Task: `peg-insert-side-v3`, Metaworld simulator. Uses `MetaworldEnv` from `lerobot.envs.metaworld` with `obs_type="pixels_agent_pos"`. Observation dict has image at key `observation.images.corner2` and `agent_pos` (4-dim). Success from `info["is_success"]`.

**MetaworldEnv quirk:** `step()` calls `self.reset()` internally when `terminated=True`. The RL loop must **not** call `env.reset()` again after receiving `terminated=True` — the env is already reset.

Metaworld runs on CPU; actor/critic train on GPU. An oracle policy `SawyerPegInsertionSideV3Policy` is available in `metaworld.policies` if a fallback or warmup from oracle is needed later.

---

## 9. Key Call Paths

### Extracting z_rl from an observation

`get_rl_state()` was added to `VLAFlowMatching` in `modeling_smolvla.py`:

```python
@torch.no_grad()
def get_rl_state(self, images, img_masks, lang_tokens, lang_masks, state):
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids  = torch.cumsum(prefix_pad_masks, dim=1) - 1
    (prefix_out, _), _ = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        fill_kv_cache=False,
    )
    return self.transformer_rlt.encode(prefix_out.float())  # (batch, rlt_d_model=2048)
```

Usage in training script:
```python
z_rl     = vla_policy.model.get_rl_state(images, img_masks, lang_tokens, lang_masks, state)
proprio  = vla_policy.prepare_state(batch)[:, 0, :]   # (batch, 32)
rl_state = torch.cat([z_rl, proprio], dim=-1)          # (batch, 2080)
```

### Getting fresh ã for actor loss

```python
with torch.no_grad():
    vla_ref      = vla_policy.predict_action_chunk(obs_batch)  # (batch, 10, 4)
    vla_ref_flat = vla_ref.flatten(1)                          # (batch, 40)
```

---

## 10. Hyperparameter Reference

| Parameter | Default | Source |
|---|---|---|
| `actor_hidden_dim` | 256 | Paper |
| `actor_hidden_layers` | 2 | Paper |
| `critic_hidden_dim` | 256 | Paper |
| `critic_hidden_layers` | 2 | Paper |
| `chunk_size_rl` (C) | 10 | Paper |
| `ref_action_dropout_prob` | 0.5 | Paper |
| `G` | 5 | Paper |
| `tau` | 0.005 | **[OUR DECISION]** |
| `gamma` | 0.99 | **[OUR DECISION]** |
| `beta` | 0.1 | **[OUR DECISION]** |
| `actor_output_variance` | 0.1 | **[OUR DECISION]** |
| `batch_size_rl` | 256 | **[OUR DECISION]** |
| `replay_buffer_capacity` | 10000 | **[OUR DECISION]** |
| `warmup_episodes` | 20 | **[OUR DECISION]** |
| `actor_lr` / `critic_lr` | 3e-4 | TD3 standard |
| `total_episodes` | 1000 | **[OUR DECISION]** |
| `eval_freq` | 50 | **[OUR DECISION]** |
| `action_dim` | 4 | Metaworld Sawyer (xyz + gripper) |
| `z_rl_dim` | 2048 | Matches `rlt_d_model` in SmolVLAConfig |
| `proprio_dim` | 32 | Matches `max_state_dim` in SmolVLAConfig |

---

## 11. Code Organization

Three new files for Phase 3:

**`src/lerobot/policies/smolvla/actor_critic_rlt.py`** — all RL network code. `RLTActorCriticConfig`, `RLTActor`, `RLTCritic`, `polyak_update`, `make_target`, `compute_td3_critic_loss`, `compute_td3_actor_loss`. Imports `MLP` and `CriticHead` from `lerobot.policies.sac.modeling_sac`.

**`src/lerobot/scripts/train_actor_critic_rlt.py`** — single-process online RL training script. `collect_episode()`, `evaluate()`, `train()`. Imports `ReplayBuffer` from `lerobot.rl.buffer`, `MetaworldEnv` from `lerobot.envs.metaworld`, `SmolVLAPolicy` from `lerobot.policies.smolvla.modeling_smolvla`.

**`scripts/train_rlt_ac.sh`** — SLURM script.

One small addition to an existing file: `get_rl_state()` added to `VLAFlowMatching` in `modeling_smolvla.py` (see §9). Everything else (SmolVLA, TransformerRLT, SAC buffer, SAC MLP) is imported unchanged.

---

## 12. Random Implementation Details

**Terminal transition storage:** The terminal transition (r=1, done=True) is stored with `next_rl_state` computed from the terminal observation returned by `MetaworldEnv.step()`. Internally MetaworldEnv calls `self.reset()` after formatting the terminal obs (see [metaworld.py:254-261](src/lerobot/envs/metaworld.py#L254-L261)), and the reset result is discarded — the returned obs is the pre-reset terminal one. This does not affect the Bellman target because `(1-done)` zeros out the bootstrap on terminals.

**Fresh ã for actor loss:** Accepted deviation (see §2). The training script uses the stored `vla_ref_action` from the replay buffer rather than re-running VLA inference at training time. Running the full VLA denoiser (10 Euler steps) across 256 batch samples per gradient update is prohibitively expensive.

**Checkpoint format (two-checkpoint handoff):**
- Phase 1 (`scripts/peg_insert_side.sh`) saves full SmolVLA checkpoints with the BC-trained action head; no TransformerRLT.
- Phase 2 (`scripts/train_rlt_token.sh`) does NOT load Phase 1 — it starts from the pretrained VLM (`load_vlm_weights=true`) with a randomly-initialized action head, and saves checkpoints containing VLM + random action head + trained TransformerRLT.
- Phase 3 (`scripts/train_rlt_ac.sh`) therefore requires BOTH checkpoints: `--rlt_checkpoint` is loaded as the base policy (for VLM + TransformerRLT), and the action-head modules (`action_in_proj`, `action_out_proj`, `action_time_mlp_{in,out}`, `state_proj`) are patched in from `--vla_checkpoint`. Actor/critic weights are saved separately during Phase 3.

Side effect: `state_proj` is also patched in at Phase 3. In Phase 2 it was random, so the RLT was trained on state-token features produced by a random projection; in Phase 3 it sees the trained projection. Treated as a small distribution shift because the state token is ~1/270 of the prefix on MetaWorld.