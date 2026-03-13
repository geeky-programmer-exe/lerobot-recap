# RECAP Baseline Setup — Step by Step

```bash
conda create -n recap python=3.12 -y
conda activate recap
```

Check your CUDA version first:
```bash
nvidia-smi
```

```bash
pip install torch torchvision torchaudio
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Install LeRobot with smolvla + libero extras

```bash
cd <PATH TO LEROBOT FOLDER>
pip install -e ".[smolvla,libero]"  
```

Verify lerobot:
```bash
python -c "import lerobot; print(lerobot.__version__)"
lerobot-train --help
```

```bash
python -c "
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

bench = benchmark.get_benchmark_dict()
suite = bench['libero_spatial']()

env = LiberoEnv(task_suite=suite, task_suite_name='libero_spatial', task_id=0)
obs, info = env.reset()
print('obs keys:', obs.keys())
print('image shape:', obs['pixels']['image'].shape)
env.close()
print('LIBERO OK')
"
```

Note: robosuite macro warnings are harmless, ignore them.
Expected: `obs keys: dict_keys(['pixels'])`, image shape `(256, 256, 3)`, `LIBERO OK`

```bash
pip install wandb
wandb login
```

Verify:
```bash
python -c "import wandb; print(wandb.__version__)"
```

This checks the model architecture loads, parameters
are frozen correctly, and a forward pass works on CPU/low VRAM. also downloads the LIBERO dataset

```bash
python - <<'EOF'
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

cfg = SmolVLAConfig()
cfg.load_vlm_weights = True
cfg.train_expert_only = True

print("Config OK")
print(f"  load_vlm_weights: {cfg.load_vlm_weights}")
print(f"  train_expert_only: {cfg.train_expert_only}")

# Load model (will download SmolVLM2-500M from HF Hub ~1GB, first time only)
policy = SmolVLAPolicy(cfg)
policy.eval()

# Count trainable vs frozen params
trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
print(f"  Trainable params: {trainable/1e6:.1f}M
print(f"  Frozen params:    {frozen/1e6:.1f}M
print("SmolVLA load OK")
EOF
```

Expected: trainable ~100M, frozen ~350M.


## Training

Set dataset path:
```bash
export HF_LEROBOT_HOME=<PATH TO DATA FOLDER>
```

```bash
conda activate recap
cd <PATH TO LEROBOT>

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --batch_size=32 \
  --steps=20000 \
  --log_freq=100 \
  --save_freq=5000 \
  --output_dir=./checkpoints/sft_baseline \
  --wandb.enable=true \
  --wandb.project=recap-smolvla
```

After training completes, run eval on LIBERO-Spatial:

```bash
lerobot-eval \
  --policy.path=./checkpoints/sft_baseline \
  --env.type=libero \
  --env.task_suite=libero_spatial \
  --eval.n_episodes=50 \
  --eval.batch_size=10
```

This gives you the success rate number (primary metric for the paper).

---

## Quick reference: what each flag does

| Flag | Value | Why |
|------|-------|-----|
| `--policy.type=smolvla` | smolvla | Use SmolVLA policy class |
| `--policy.load_vlm_weights=true` | true | Load pretrained SmolVLM2 backbone from HF Hub |
| `--policy.train_expert_only=true` | true | Freeze VLM, only train ~100M action expert |
| `--dataset.repo_id=...` | libero_spatial_no_noops | LIBERO Spatial expert demos |
| `--batch_size=32` | 32 | Reduce to 16 if OOM |
| `--steps=20000` | 20000 | ~20k steps is enough for SFT baseline |
| `--wandb.enable=true` | true | Log to WandB for training curves |
| `--wandb.project=recap-smolvla` | recap-smolvla | WandB project name |

---

## Files that matter (do not touch the rest)

```
src/lerobot/policies/smolvla/
├── modeling_smolvla.py          ← policy forward pass
├── smolvlm_with_expert.py       ← architecture + parameter freezing
└── configuration_smolvla.py    ← config (we'll edit this for RECAP Phase 2)

src/lerobot/scripts/
├── lerobot_train.py             ← training loop
└── eval.py                      ← evaluation

src/lerobot/envs/libero.py       ← LIBERO env wrapper
```
