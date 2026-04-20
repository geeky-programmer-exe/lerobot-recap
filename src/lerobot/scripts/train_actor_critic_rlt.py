"""
Loads a frozen SmolVLA + TransformerRLT checkpoint,
then trains a lightweight TD3 actor-critic using z_rl as the RL state representation.

Usage (see scripts/train_rlt_ac.sh for SLURM wrapper):
    python train_actor_critic_rlt.py \
        --vla_checkpoint /path/to/peg-sft \
        --output_dir /path/to/output \
        --task peg-insert-side-v3 \
        --image_key observation.images.corner2
"""

import time
import torch
import wandb
import argparse
import logging
from pathlib import Path

from lerobot.envs.metaworld import MetaworldEnv
from lerobot.policies.smolvla.actor_critic_rlt import (
    RLTActor,
    RLTActorCriticConfig,
    RLTCritic,
    compute_td3_actor_loss,
    compute_td3_critic_loss,
    make_target,
    polyak_update,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def tokenize_task(vla: SmolVLAPolicy, task_description: str, device: torch.device):
    """Tokenize task description using VLM tokenizer. Called once at startup."""
    tokenizer = vla.model.vlm_with_expert.processor.tokenizer
    text = task_description + "\n"  # SmolVLA appends newline (SmolVLANewLineProcessor)
    cfg = vla.config
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length" if cfg.pad_language_to is not None else True,
        max_length=cfg.tokenizer_max_length,
        truncation=True,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].bool().to(device)


def obs_to_smolvla_batch(
    obs: dict,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    image_key: str,
    device: torch.device,
) -> dict:
    """Convert a single Metaworld observation dict to a SmolVLA batch dict.

    MetaworldEnv (obs_type='pixels_agent_pos') returns:
        obs["pixels"]    — np.ndarray (H, W, 3) uint8
        obs["agent_pos"] — np.ndarray (4,)  [x, y, z, gripper]

    SmolVLA expects:
        batch[image_key]              — (1, 3, H, W) float32 in [0, 1]
        batch["observation.state"]    — (1, 4) float32
        batch[OBS_LANGUAGE_TOKENS]    — (1, seq_len) int64
        batch[OBS_LANGUAGE_ATTENTION_MASK] — (1, seq_len) int64
    """
    img = torch.from_numpy(obs["pixels"].copy()).float() / 255.0  # (H, W, 3)
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)            # (1, 3, H, W)

    state = torch.from_numpy(obs["agent_pos"].copy()).float().unsqueeze(0).to(device)  # (1, 4)

    return {
        image_key: img,
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: lang_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
    }


@torch.no_grad()
def extract_rl_state(
    vla: SmolVLAPolicy,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z_rl, proprio_padded) from a SmolVLA batch.

    z_rl          : (1, rlt_d_model)  — from frozen RLT encoder (default 2048)
    proprio_padded: (1, max_state_dim) — state padded (default 32)
    """
    images, img_masks = vla.prepare_images(batch)
    state = vla.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    images = [img.to(device) for img in images]
    img_masks = [m.to(device) for m in img_masks]
    state = state.to(device)

    z_rl = vla.model.get_rl_state(images, img_masks, lang_tokens, lang_masks, state)
    return z_rl, state


@torch.no_grad()
def get_vla_ref_action(
    vla: SmolVLAPolicy,
    batch: dict,
) -> torch.Tensor:
    """Run VLA denoising to get the reference action chunk.

    Returns: (1, C, action_dim) float32 — already sliced to original_action_dim (4).
    """
    vla.reset()
    return vla.predict_action_chunk(batch)  # (1, C, 4)


def collect_episode(
    env: MetaworldEnv,
    vla: SmolVLAPolicy,
    actor: RLTActor | None,
    replay: ReplayBuffer,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    image_key: str,
    config: RLTActorCriticConfig,
    device: torch.device,
    use_actor: bool = True,
) -> dict:
    """Run one full episode and push chunk-level transitions into the replay buffer.

    During warmup (use_actor=False) the VLA reference action is executed directly.
    During RL training (use_actor=True) the actor produces the executed chunk.

    Returns episode info dict: {"success": bool, "steps": int, "transitions": int}
    """
    obs, _ = env.reset()
    vla.reset()

    episode_success = False
    total_steps = 0
    transitions_added = 0

    while True:
        batch = obs_to_smolvla_batch(obs, lang_tokens, lang_masks, image_key, device)

        # --- Extract RL state at current obs ---
        z_rl, proprio = extract_rl_state(vla, batch, device)
        rl_state = torch.cat([z_rl, proprio], dim=-1)

        # --- Get VLA reference action ---
        vla_ref = get_vla_ref_action(vla, batch)                # (1, C, 4)
        vla_ref_flat = vla_ref.flatten(1)                       # (1, 40)

        # --- Pick executed action ---
        # Rollout always feeds the full VLA ref to the actor. Ref-action dropout
        # is applied only inside the actor loss (see compute_td3_actor_loss).
        if use_actor and actor is not None:
            action_chunk = actor.select_action(z_rl, proprio, vla_ref_flat, add_noise=True)
        else:
            action_chunk = vla_ref  # warmup: execute VLA directly

        # --- Execute C steps in environment ---
        chunk_reward = 0.0
        done = False
        truncated = False

        for step_in_chunk in range(config.chunk_size_rl):
            action_np = action_chunk[0, step_in_chunk].detach().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            chunk_reward += reward
            total_steps += 1
            done = terminated or truncated

            if done:
                episode_success = bool(info.get("is_success", False))
                break

        next_batch = obs_to_smolvla_batch(next_obs, lang_tokens, lang_masks, image_key, device)
        next_z_rl, next_proprio = extract_rl_state(vla, next_batch, device)
        next_rl_state = torch.cat([next_z_rl, next_proprio], dim=-1)

        replay.add(
            state={"rl_state": rl_state.squeeze(0).cpu()},
            action=action_chunk.flatten(1).squeeze(0).detach().cpu(),
            reward=chunk_reward,
            next_state={"rl_state": next_rl_state.squeeze(0).cpu()},
            done=done,
            truncated=truncated,
            complementary_info={"vla_ref_action": vla_ref_flat.squeeze(0).cpu()},
        )
        transitions_added += 1

        if done:
            break
        obs = next_obs

    return {
        "success": episode_success,
        "steps": total_steps,
        "transitions": transitions_added,
    }


@torch.no_grad()
def evaluate(
    env: MetaworldEnv,
    vla: SmolVLAPolicy,
    actor: RLTActor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    image_key: str,
    config: RLTActorCriticConfig,
    device: torch.device,
    n_episodes: int = 10,
) -> float:
    """Run n_episodes with the actor (no noise, no dropout) and return success rate."""
    actor.eval()
    successes = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        vla.reset()
        done = False

        while not done:
            batch = obs_to_smolvla_batch(obs, lang_tokens, lang_masks, image_key, device)
            z_rl, proprio = extract_rl_state(vla, batch, device)
            vla_ref = get_vla_ref_action(vla, batch)
            vla_ref_flat = vla_ref.flatten(1)

            action_chunk = actor.select_action(z_rl, proprio, vla_ref_flat, add_noise=False)

            for step_in_chunk in range(config.chunk_size_rl):
                action_np = action_chunk[0, step_in_chunk].cpu().numpy()
                obs, _, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                if done:
                    successes += int(info.get("is_success", False))
                    break

    actor.train()
    return successes / n_episodes


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Config ---
    config = RLTActorCriticConfig(
        total_episodes=args.total_episodes,
        warmup_episodes=args.warmup_episodes,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        batch_size_rl=args.batch_size,
        G=args.G,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        replay_buffer_capacity=args.buffer_capacity,
    )

    # --- WandB ---
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.job_name,
            config=vars(args),
        )

    # --- Load frozen VLA ---
    log.info(f"Loading RLT checkpoint (Phase 2) from {args.rlt_checkpoint}")
    vla = SmolVLAPolicy.from_pretrained(args.rlt_checkpoint)

    log.info(f"Patching in action head from VLA checkpoint (Phase 1) at {args.vla_checkpoint}")
    phase1 = SmolVLAPolicy.from_pretrained(args.vla_checkpoint)
    action_head_modules = ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]
    for name in action_head_modules:
        getattr(vla.model, name).load_state_dict(getattr(phase1.model, name).state_dict())
    del phase1

    vla.eval().to(device)
    for p in vla.parameters():
        p.requires_grad_(False)

    # --- Tokenize task description once ---
    from lerobot.envs.metaworld import TASK_DESCRIPTIONS
    task_desc = TASK_DESCRIPTIONS.get(args.task, args.task)
    log.info(f"Task: {args.task} | Description: '{task_desc}'")
    lang_tokens, lang_masks = tokenize_task(vla, task_desc, device)

    # --- Environment ---
    env = MetaworldEnv(task=args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")

    # --- Actor & Critics ---
    actor = RLTActor(config).to(device)
    critic = RLTCritic(config, seed1=0, seed2=1).to(device)
    target_actor = make_target(actor)
    target_critic = make_target(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    # --- Replay buffer ---
    replay = ReplayBuffer(
        capacity=config.replay_buffer_capacity,
        device=str(device),
        state_keys=["rl_state"],
        use_drq=False,
        storage_device="cpu",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Warmup: fill replay buffer with VLA rollouts before RL updates
    # -------------------------------------------------------------------
    log.info(f"Warmup: collecting {config.warmup_episodes} VLA episodes...")
    warmup_successes = 0
    for ep in range(config.warmup_episodes):
        info = collect_episode(
            env, vla, actor=None, replay=replay,
            lang_tokens=lang_tokens, lang_masks=lang_masks,
            image_key=args.image_key, config=config,
            device=device, use_actor=False,
        )
        warmup_successes += int(info["success"])
        log.info(f"  Warmup ep {ep+1}/{config.warmup_episodes} | success={info['success']} | "
                 f"steps={info['steps']} | buffer={replay.size}")

    log.info(f"Warmup done. Buffer size: {replay.size} | "
             f"Success rate: {warmup_successes}/{config.warmup_episodes}")

    # -------------------------------------------------------------------
    # Online RL training
    # -------------------------------------------------------------------
    global_grad_step = 0
    best_success_rate = 0.0

    for episode in range(config.total_episodes):
        t0 = time.time()

        # Collect one episode with actor
        ep_info = collect_episode(
            env, vla, actor=actor, replay=replay,
            lang_tokens=lang_tokens, lang_masks=lang_masks,
            image_key=args.image_key, config=config,
            device=device, use_actor=True,
        )

        # G gradient updates per chunk-transition collected this episode
        n_updates = config.G * ep_info["transitions"]
        critic_losses, actor_losses = [], []

        for _ in range(n_updates):
            batch = replay.sample(config.batch_size_rl)

            rl_state = batch["state"]["rl_state"].to(device)
            action_flat = batch["action"].to(device)
            reward = batch["reward"].to(device)
            next_rl_state = batch["next_state"]["rl_state"].to(device)
            done = batch["done"].float().to(device)

            # --- Critic update (every step) ---
            c_loss = compute_td3_critic_loss(
                rl_state, action_flat, reward, next_rl_state, done,
                critic, target_critic, target_actor, config,
            )
            critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            critic_losses.append(c_loss.item())

            # --- Actor + target update (every 2nd step) ---
            if global_grad_step % 2 == 0:
                # VLA ref is read from the replay buffer (stored at collection time)
                vla_ref = batch["complementary_info"]["vla_ref_action"].to(device)

                a_loss = compute_td3_actor_loss(
                    rl_state, vla_ref, actor, critic, config
                )
                actor_optimizer.zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
                actor_losses.append(a_loss.item())

                polyak_update(actor, target_actor, config.tau)
                polyak_update(critic, target_critic, config.tau)

            global_grad_step += 1

        ep_time = time.time() - t0
        mean_c_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        mean_a_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0

        log.info(
            f"Ep {episode+1}/{config.total_episodes} | "
            f"success={ep_info['success']} | steps={ep_info['steps']} | "
            f"c_loss={mean_c_loss:.4f} | a_loss={mean_a_loss:.4f} | "
            f"buf={replay.size} | t={ep_time:.1f}s"
        )

        if args.wandb:
            wandb.log({
                "episode": episode,
                "ep_success": int(ep_info["success"]),
                "critic_loss": mean_c_loss,
                "actor_loss": mean_a_loss,
                "buffer_size": replay.size,
                "grad_steps": global_grad_step,
            })

        # --- Evaluation ---
        if (episode + 1) % config.eval_freq == 0:
            success_rate = evaluate(
                env, vla, actor, lang_tokens, lang_masks,
                args.image_key, config, device, n_episodes=config.eval_episodes,
            )
            log.info(f"  [EVAL] Episode {episode+1} | success_rate={success_rate:.2f}")

            if args.wandb:
                wandb.log({"eval/success_rate": success_rate, "episode": episode})

            # Save checkpoint
            ckpt = {
                "episode": episode,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_actor": target_actor.state_dict(),
                "target_critic": target_critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "config": config,
            }
            ckpt_path = output_dir / f"checkpoint_ep{episode+1}.pt"
            torch.save(ckpt, ckpt_path)
            log.info(f"  Saved checkpoint: {ckpt_path}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save(ckpt, output_dir / "best_checkpoint.pt")
                log.info(f"  New best: {best_success_rate:.2f}")

    log.info(f"Training complete. Best success rate: {best_success_rate:.2f}")
    env.close()
    if args.wandb:
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="RLT Phase 3: TD3 actor-critic training")

    # Required
    p.add_argument("--vla_checkpoint", type=str, required=True, help="Path to Phase 1 SmolVLA checkpoint (trained action head)")
    p.add_argument("--rlt_checkpoint", type=str, required=True, help="Path to Phase 2 SmolVLA checkpoint (trained TransformerRLT)")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save actor-critic checkpoints")

    # Environment
    p.add_argument("--task", type=str, default="peg-insert-side-v3")
    p.add_argument("--image_key", type=str, default="observation.images.corner2", help="Image key in SmolVLA batch dict — check vla.config.image_features")

    # Training budget
    p.add_argument("--total_episodes", type=int, default=1000)
    p.add_argument("--warmup_episodes", type=int, default=20)
    p.add_argument("--eval_freq", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=10)

    # Hyperparameters
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_capacity", type=int, default=10_000)
    p.add_argument("--G", type=int, default=5, help="Gradient steps per chunk transition (paper)")
    p.add_argument("--beta", type=float, default=0.1, help="VLA regularization weight")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate")
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)

    # WandB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="rlt-smolvla")
    p.add_argument("--wandb_entity", type=str, default="idl_34")
    p.add_argument("--job_name", type=str, default="rlt-ac")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
