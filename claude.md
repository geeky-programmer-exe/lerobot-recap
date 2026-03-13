# Project: RECAP-style RL Fine-tuning of SmolVLA on LIBERO

## What we are building

We are implementing RECAP (advantage-conditioned policy fine-tuning) on top of SmolVLA, a 450M parameter flow-based VLA from HuggingFace. The project runs on the LeRobot codebase (forked, editable install). The environment is LIBERO simulation (Franka Panda, tabletop manipulation).

**Core hypothesis:** RL fine-tuning via advantage conditioning can outperform pure behavior cloning, reducing the clean data requirements bottleneck in robot learning.

**This is a course project (CMU IDL). Pragmatic > perfect.**

---

## Codebase

Forked from: https://github.com/huggingface/lerobot
Install: `pip install -e ".[smolvla]"` from repo root

### Files we care about

```
src/lerobot/
├── common/policies/smolvla/
│   ├── modeling_smolvla.py       ← MAIN FILE. SmolVLA model definition.
│   ├── configuration_smolvla.py  ← Config dataclass
│   └── ...
├── scripts/
│   ├── train.py                  ← Training entrypoint
│   └── eval.py                   ← Evaluation + rollout collection
├── datasets/
│   └── lerobot_dataset.py        ← Dataset loading (LeRobotDataset)
└── envs/                         ← LIBERO env wrapper lives here
```

### Files we ignore

Everything in `src/lerobot/robots/` (real hardware), and all other policies
(ACT, Diffusion, TDMPC, GR00T, Pi0, Pi0Fast, VQ-BeT). Do not touch or reference these.

---

## Model Architecture: SmolVLA

Two components:

**1. VLM Backbone (FROZEN — do not train)**
- SmolVLM2 (SigLIP vision encoder + SmolLM2 language decoder)
- Processes: RGB images (multiple cameras) + language instruction + robot state
- Robot state projected to single token via linear layer
- Outputs: contextual feature tokens passed to action expert

**2. Action Expert (THIS IS WHAT WE TRAIN)**
- ~100M parameter flow matching transformer
- Interleaved cross-attention (attends to VLM features) + self-attention (temporal)
- Input: noisy action + VLM context features
- Output: denoising vector → action chunk
- Training objective: flow matching loss

Key config values:
- Action chunk size: typically 16-32 steps
- Visual tokens per frame: 64 (compressed via PixelShuffle)
- VLM layers used: only first N/2 layers (layer skipping for efficiency)

---

## Environment: LIBERO

- MuJoCo tabletop manipulation, Franka Panda arm (7-DOF)
- 4 task suites: Spatial, Object, Goal, Long (increasing difficulty)
- Each suite: 10 tasks × 50 initial states = 500 episodes
- Success signal: binary, end of episode (built-in, no reward shaping needed)
- Obs: agent-view RGB + wrist-view RGB (224×224) + joint angles + gripper state + language

**Important:** SmolVLA pretrained on SO100 (6-DOF). LIBERO uses Franka (7-DOF).
The state projection linear layer input dim needs to match. Simplest fix:
train from scratch (`--policy.type=smolvla`) rather than loading pretrained weights.
This sidesteps the DOF mismatch entirely.

Start with LIBERO-Spatial (easiest). Move to LIBERO-Long only if time permits.

---

## Phase 1: Baseline (SFT — Behavior Cloning)

Train the flow head on LIBERO expert demonstrations. VLM frozen.

**Dataset:** `lerobot/libero_spatial_no_noops` (on HF Hub)

**Train command:**
```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=lerobot/libero_spatial_no_noops \
  --batch_size=64 \
  --steps=20000
```

**What to verify before moving to Phase 2:**
- VLM backbone parameters have no gradients (confirm by checking `param.requires_grad`)
- Only action expert parameters are being updated
- Flow matching loss is decreasing
- Run eval on LIBERO-Spatial, get a task success rate number

---

## Phase 2: RECAP Implementation (The Novel Part)

RECAP is from π0.6 (not open sourced). We implement it from scratch on top of SmolVLA.

### Core idea

Condition the action expert on a binary advantage token:
- `A_pos` (label=1) = this was a successful episode
- `A_neg` (label=0) = this was a failed episode

During training: use ground truth label from rollout outcome.
During inference: always use `A_pos` to steer toward success.

---

### Step 1: Collect labeled rollouts

**New file:** `src/lerobot/scripts/collect_rollouts.py`

Run the trained SFT baseline in LIBERO. Save episodes to LeRobotDataset format.
Add a field `advantage_label` (int, 1=success, 0=failure) to each episode.

```python
# Pseudocode for collect_rollouts.py
policy = SmolVLAPolicy.from_pretrained("./checkpoints/sft_baseline")
for episode in run_policy_in_libero(policy, n_episodes=500):
    label = 1 if episode.success else 0
    save_episode(episode, advantage_label=label)
```

---

### Step 2: Build mixed dataset

Combine into one LeRobotDataset:
- Original expert demos → `advantage_label=1`
- Successful SFT rollouts → `advantage_label=1`
- Failed SFT rollouts → `advantage_label=0`

Aim for a roughly balanced mix of pos/neg. Too many negatives = policy collapses.

---

### Step 3: Modify modeling_smolvla.py

**Where to add:** Inside the action expert class, not the VLM.

```python
# In action expert __init__:
self.advantage_embedding = nn.Embedding(2, action_expert_hidden_dim)

# In action expert forward():
# advantage_label: torch.LongTensor, shape (batch,), values in {0, 1}
adv_token = self.advantage_embedding(advantage_label)  # (batch, hidden_dim)
# Prepend adv_token to the action token sequence before the attention blocks
# This conditions the entire denoising trajectory
```

**Also modify configuration_smolvla.py:**
```python
use_advantage_conditioning: bool = False
```

**Inference default behavior:**
If `use_advantage_conditioning=True` and no label provided at inference,
default to `advantage_label=1` (always steer toward success).

---

### Step 4: Modify training pipeline

- `train.py` — pass `advantage_label` from dataset batch through to model forward call
- Data transform — ensure `advantage_label` field flows from dataset dict to model input dict
- VLM stays frozen (same as Phase 1)

---

### Step 5: RECAP fine-tuning

```bash
lerobot-train \
  --policy.path=./checkpoints/sft_baseline \
  --dataset.repo_id=./data/mixed_recap_dataset \
  --batch_size=64 \
  --steps=10000 \
  --policy.use_advantage_conditioning=true
```

---

## Evaluation Plan

| Experiment | What it tests |
|---|---|
| SFT baseline | BC with clean data only |
| RECAP (A_pos at inference) | Does advantage conditioning improve over SFT? |
| Ablation: A_neg at inference | Does the token actually steer behavior directionally? |
| Ablation: advantage token zeroed | Is gain from conditioning or just more data? |

Primary metric: **task success rate** on LIBERO-Spatial (500 episodes = 10 tasks × 50 states).

---

## Compute Constraints

- 16GB VRAM on local GPU
- VLM backbone ALWAYS frozen — we only train ~100M action expert params
- Drop `batch_size` to 32 if OOM
- PSC cluster available for longer runs
- Start small: LIBERO-Spatial, 20k steps SFT, verify before scaling

---

## What NOT to do

- Do NOT train the VLM backbone — frozen always, no exceptions
- Do NOT implement PPO, GRPO, or any critic — RECAP is advantage conditioning, not online RL
- Do NOT touch anything in `src/lerobot/robots/`
- Do NOT modify other policy implementations (ACT, Diffusion, etc.)
- Do NOT use JAX — pure PyTorch only
- Do NOT try to get ManiSkill3 working — LIBERO only

---

## References

- SmolVLA blog: https://huggingface.co/blog/smolvla
- SmolVLA base checkpoint: `lerobot/smolvla_base`
- RECAP concept (π0.6, not open sourced): https://www.pi.website/blog/pistar06
- LIBERO: https://libero-project.github.io/
- Flow matching: Lipman et al. arXiv:2210.02747