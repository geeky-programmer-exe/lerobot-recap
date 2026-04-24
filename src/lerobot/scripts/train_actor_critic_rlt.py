"""
Loads a frozen SmolVLA + TransformerRLT checkpoint,
then trains a lightweight TD3 actor-critic using z_rl as the RL state representation.

Usage (see scripts/train_rlt_ac.sh for SLURM wrapper):
    python train_actor_critic_rlt.py \
        --vla_checkpoint /path/to/peg-sft \
        --output_dir /path/to/output \
        --task peg-insert-side-v3
"""

import time
import torch
import wandb
import argparse
import logging
import math
from pathlib import Path

from lerobot.envs.metaworld import MetaworldEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.actor_critic_rlt import (
    RLTActor,
    RLTActorCriticConfig,
    RLTCritic,
    compute_td3_actor_loss,
    compute_td3_critic_loss,
    make_target,
    polyak_update,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.io_utils import write_video
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)


@torch.no_grad()
def extract_rl_state_and_vla_ref(
    vla: SmolVLAPolicy,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VLM forward returning (z_rl, proprio, vla_ref_chunk).

    z_rl      : (1, rlt_d_model)
    proprio   : (1, max_state_dim)
    vla_ref   : (1, chunk_size, action_dim)
    """
    images, img_masks = vla.prepare_images(batch)
    state = vla.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    images = [img.to(device) for img in images]
    img_masks = [m.to(device) for m in img_masks]
    state = state.to(device)

    z_rl, actions = vla.model.get_rl_state_and_actions(images, img_masks, lang_tokens, lang_masks, state)
    original_action_dim = vla.config.action_feature.shape[0]
    vla_ref = actions[:, :, :original_action_dim]
    return z_rl, state, vla_ref


def get_q_loss_weight(
    episode: int,
    curriculum_start_episode: int | None,
    config: RLTActorCriticConfig,
) -> float:
    if curriculum_start_episode is None:
        return 0.0

    if config.q_loss_weight_increment <= 0 or config.q_loss_weight_max <= 0:
        return 0.0

    jumps_needed = max(1, math.ceil(config.q_loss_weight_max / config.q_loss_weight_increment))
    remaining_episodes = max(1, config.total_episodes - curriculum_start_episode)
    progressed_episodes = min(remaining_episodes, episode - curriculum_start_episode + 1)
    completed_jumps = max(1, math.ceil(progressed_episodes * jumps_needed / remaining_episodes))
    return min(config.q_loss_weight_max, completed_jumps * config.q_loss_weight_increment)


def get_ref_action_dropout_prob(
    q_loss_weight: float,
    config: RLTActorCriticConfig,
) -> float:
    if config.ref_action_dropout_prob <= 0 or config.q_loss_weight_max <= 0:
        return 0.0

    progress = min(1.0, max(0.0, q_loss_weight / config.q_loss_weight_max))
    return progress * config.ref_action_dropout_prob


def collect_episode(
    env: MetaworldEnv,
    vla: SmolVLAPolicy,
    actor: RLTActor | None,
    replay: ReplayBuffer,
    preprocessor,
    postprocessor,
    config: RLTActorCriticConfig,
    device: torch.device,
    use_actor: bool = True,
    sparse_reward: bool = False,
    time_penalty: bool = False,
) -> dict:
    """Run one full episode and push chunk-level transitions into the replay buffer.

    Uses the lerobot preprocessor pipeline for state normalization and tokenization.
    During warmup (use_actor=False) the VLA reference action is executed directly.
    During RL training (use_actor=True) the actor produces the executed chunk.
    """
    raw_obs, _ = env.reset()
    vla.reset()
    cached_current = None

    episode_success = False
    total_steps = 0
    total_reward = 0.0
    transitions_added = 0

    while True:
        # Reuse the already-computed next-state VLA features from the previous chunk when available. 
        if cached_current is None:
            obs = preprocess_observation(raw_obs)
            obs["task"] = env.task_description
            obs = preprocessor(obs)
            z_rl, proprio, vla_ref = extract_rl_state_and_vla_ref(vla, obs, device)
        else:
            z_rl, proprio, vla_ref = cached_current
            cached_current = None
        rl_state = torch.cat([z_rl, proprio], dim=-1)
        vla_ref_flat = vla_ref.flatten(1)                       # (1, 40)

        if use_actor and actor is not None:
            action_chunk = actor.select_action(z_rl, proprio, vla_ref_flat, add_noise=True)
        else:
            action_chunk = vla_ref  # (1, C, action_dim)

        # Unnormalize for env execution; store normalized copy in replay buffer
        action_chunk_exec = postprocessor(action_chunk)

        # --- Execute C steps in environment ---
        chunk_reward = 0.0
        done = False
        truncated = False

        for step_i in range(config.chunk_size_rl):
            action_np = action_chunk_exec[0, step_i].detach().cpu().numpy()
            raw_next_obs, reward, terminated, truncated, info = env.step(action_np)
            if time_penalty:
                reward = 0.0 if bool(info.get("is_success", False)) else -1.0
            elif sparse_reward:
                reward = float(info.get("is_success", False))
            chunk_reward += (config.gamma ** step_i) * reward
            total_reward += reward
            total_steps += 1
            done = terminated or truncated

            if done:
                episode_success = bool(info.get("is_success", False))
                break

        # --- Compute next rl_state ---
        if done:
            next_rl_state = torch.zeros_like(rl_state)
            next_vla_ref_flat = torch.zeros_like(vla_ref_flat)
        else:
            next_obs = preprocess_observation(raw_next_obs)
            next_obs["task"] = env.task_description
            next_obs = preprocessor(next_obs)

            next_z_rl, next_proprio, next_vla_ref = extract_rl_state_and_vla_ref(vla, next_obs, device)
            next_rl_state = torch.cat([next_z_rl, next_proprio], dim=-1)
            next_vla_ref_flat = next_vla_ref.flatten(1)
            cached_current = (next_z_rl, next_proprio, next_vla_ref)

        replay.add(
            state={"rl_state": rl_state.squeeze(0).cpu()},
            action=action_chunk.flatten(1).squeeze(0).detach().cpu(),
            reward=float(chunk_reward),
            next_state={"rl_state": next_rl_state.squeeze(0).cpu()},
            done=done,
            truncated=truncated,
            complementary_info={
                "vla_ref_action": vla_ref_flat.squeeze(0).cpu(),
                "next_vla_ref_action": next_vla_ref_flat.squeeze(0).cpu(),
            },
        )
        transitions_added += 1

        if done:
            break
        raw_obs = raw_next_obs

    return {
        "success": episode_success,
        "steps": total_steps,
        "reward": float(total_reward),
        "transitions": transitions_added,
    }


@torch.no_grad()
def evaluate(
    env: MetaworldEnv,
    vla: SmolVLAPolicy,
    actor: RLTActor,
    preprocessor,
    postprocessor,
    config: RLTActorCriticConfig,
    device: torch.device,
    n_episodes: int = 10,
    max_videos: int = 0,
    videos_dir: Path | None = None,
    video_prefix: str = "eval",
) -> tuple[float, list[Path]]:
    """Run n_episodes with the actor (no noise), optionally save videos, and return success rate."""
    actor.eval()
    successes = 0
    saved_videos: list[Path] = []

    for episode_idx in range(n_episodes):
        raw_obs, _ = env.reset()
        vla.reset()
        done = False
        frames = []

        if episode_idx < max_videos:
            frames.append(env.render())

        while not done:
            obs = preprocess_observation(raw_obs)
            obs["task"] = env.task_description
            obs = preprocessor(obs)

            z_rl, proprio, vla_ref = extract_rl_state_and_vla_ref(vla, obs, device)
            vla_ref_flat = vla_ref.flatten(1)

            action_chunk = actor.select_action(z_rl, proprio, vla_ref_flat, add_noise=False)
            action_chunk_exec = postprocessor(action_chunk)

            for step_i in range(config.chunk_size_rl):
                action_np = action_chunk_exec[0, step_i].cpu().numpy()
                raw_obs, _, terminated, truncated, info = env.step(action_np)
                if episode_idx < max_videos:
                    frames.append(env.render())
                done = terminated or truncated
                if done:
                    successes += int(info.get("is_success", False))
                    break

        if episode_idx < max_videos and videos_dir is not None and len(frames) > 0:
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / f"{video_prefix}_episode_{episode_idx}.mp4"
            write_video(str(video_path), frames, fps=env.metadata["render_fps"])
            saved_videos.append(video_path)

    actor.train()
    return successes / n_episodes, saved_videos


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
        ref_action_dropout_prob=args.ref_action_dropout_prob,
        actor_output_variance=args.actor_output_variance,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        replay_buffer_capacity=args.buffer_capacity,
        q_loss_weight_max=args.q_loss_weight_max,
        q_loss_weight_increment=args.q_loss_weight_increment,
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

    # --- Load preprocessor / postprocessor from VLA checkpoint ---
    preprocessor, postprocessor = make_pre_post_processors(
        vla.config, pretrained_path=args.vla_checkpoint
    )

    # --- Environment ---
    env = MetaworldEnv(task=args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")
    log.info(f"Env task: {args.task} | description: '{env.task_description}'")

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
            preprocessor=preprocessor, postprocessor=postprocessor,
            config=config, device=device, use_actor=False,
            sparse_reward=args.sparse_reward,
            time_penalty=args.time_penalty,
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
    q_curriculum_start_episode = None
    q_curriculum_triggered = False

    for episode in range(config.total_episodes):
        t0 = time.time()

        ep_info = collect_episode(
            env, vla, actor=actor, replay=replay,
            preprocessor=preprocessor, postprocessor=postprocessor,
            config=config, device=device, use_actor=True,
            sparse_reward=args.sparse_reward,
            time_penalty=args.time_penalty,
        )
        q_loss_weight = get_q_loss_weight(episode, q_curriculum_start_episode, config)
        ref_action_dropout_prob = get_ref_action_dropout_prob(q_loss_weight, config)

        # G gradient updates per chunk-transition collected this episode
        n_updates = config.G * ep_info["transitions"]
        critic_losses, actor_losses, beta_losses = [], [], []
        actor_q_terms, delta_abs_means, delta_abs_maxes, ref_keep_fracs = [], [], [], []
        critic_grad_norms, actor_grad_norms = [], []

        for _ in range(n_updates):
            batch = replay.sample(config.batch_size_rl)

            rl_state = batch["state"]["rl_state"].to(device)
            action_flat = batch["action"].to(device)
            reward = batch["reward"].to(device)
            next_rl_state = batch["next_state"]["rl_state"].to(device)
            next_vla_ref = batch["complementary_info"]["next_vla_ref_action"].to(device)
            done = batch["done"].float().to(device)

            # --- Critic update (every step) ---
            c_loss = compute_td3_critic_loss(
                rl_state, action_flat, reward, next_rl_state, next_vla_ref, done,
                critic, target_critic, target_actor, config,
            )
            critic_optimizer.zero_grad()
            c_loss.backward()
            c_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            critic_losses.append(c_loss.item())
            critic_grad_norms.append(c_grad_norm.item())

            # --- Actor + target update (every 2nd step) ---
            if global_grad_step % 2 == 0:
                # VLA ref is read from the replay buffer (stored at collection time)
                vla_ref = batch["complementary_info"]["vla_ref_action"].to(device)

                actor_stats = compute_td3_actor_loss(
                    rl_state,
                    vla_ref,
                    actor,
                    critic,
                    config,
                    q_loss_weight=q_loss_weight,
                    ref_action_dropout_prob=ref_action_dropout_prob,
                )
                actor_optimizer.zero_grad()
                actor_stats["loss"].backward()
                a_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
                actor_losses.append(actor_stats["loss"].item())
                beta_losses.append(actor_stats["beta_loss"])
                actor_q_terms.append(actor_stats["actor_q_term"])
                delta_abs_means.append(actor_stats["delta_abs_mean"])
                delta_abs_maxes.append(actor_stats["delta_abs_max"])
                ref_keep_fracs.append(actor_stats["ref_keep_frac"])
                actor_grad_norms.append(a_grad_norm.item())

                polyak_update(actor, target_actor, config.tau)
                polyak_update(critic, target_critic, config.tau)

            global_grad_step += 1

        ep_time = time.time() - t0
        mean_c_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        mean_a_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
        mean_actor_q_term = sum(actor_q_terms) / len(actor_q_terms) if actor_q_terms else 0.0
        mean_beta = sum(beta_losses) / len(beta_losses) if beta_losses else 0.0
        mean_delta_abs = sum(delta_abs_means) / len(delta_abs_means) if delta_abs_means else 0.0
        max_delta_abs = max(delta_abs_maxes) if delta_abs_maxes else 0.0
        mean_ref_keep = sum(ref_keep_fracs) / len(ref_keep_fracs) if ref_keep_fracs else 0.0
        mean_c_gnorm = sum(critic_grad_norms) / len(critic_grad_norms) if critic_grad_norms else 0.0
        mean_a_gnorm = sum(actor_grad_norms) / len(actor_grad_norms) if actor_grad_norms else 0.0

        log.info(
            f"Ep {episode+1}/{config.total_episodes} | "
            f"success={ep_info['success']} | steps={ep_info['steps']} | reward={ep_info['reward']:.3f} | "
            f"c_loss={mean_c_loss:.4f} | a_loss={mean_a_loss:.4f} | "
            f"q_w={q_loss_weight:.2f} | ref_drop={ref_action_dropout_prob:.2f} | "
            f"actor_q={mean_actor_q_term:.4f} | actor_reg={mean_beta:.4f} | "
            f"delta_abs={mean_delta_abs:.4f}/{max_delta_abs:.4f} | keep={mean_ref_keep:.3f} | "
            f"buf={replay.size} | t={ep_time:.1f}s"
        )

        if args.wandb:
            wandb.log({
                "episode": episode,
                "ep_success": int(ep_info["success"]),
                "reward_per_episode": ep_info["reward"],
                "episode_length": ep_info["steps"],
                "critic_loss": mean_c_loss,
                "actor_loss": mean_a_loss,
                "actor_q_weight": q_loss_weight,
                "actor_ref_dropout_prob": ref_action_dropout_prob,
                "actor_q_term": mean_actor_q_term,
                "actor_reg_term": mean_beta,
                "delta_abs_mean": mean_delta_abs,
                "delta_abs_max": max_delta_abs,
                "ref_keep_frac": mean_ref_keep,
                "grad_norm_critic": mean_c_gnorm,
                "grad_norm_actor": mean_a_gnorm,
                "buffer_size": replay.size,
                "grad_steps": global_grad_step,
            })

        # --- Evaluation ---
        if (episode + 1) % config.eval_freq == 0:
            videos_dir = output_dir / "eval_videos" / f"episode_{episode+1}"
            success_rate, saved_videos = evaluate(
                env, vla, actor,
                preprocessor, postprocessor,
                config, device, n_episodes=config.eval_episodes,
                max_videos=args.eval_videos_to_save,
                videos_dir=videos_dir,
                video_prefix=f"eval_ep{episode+1}",
            )
            log.info(f"  [EVAL] Episode {episode+1} | success_rate={success_rate:.2f}")

            if (
                not q_curriculum_triggered
                and success_rate >= args.q_curriculum_start_success_rate
            ):
                q_curriculum_start_episode = episode
                q_curriculum_triggered = True
                log.info(
                    f"  [CURRICULUM] Starting Q/dropout curriculum at episode {episode+1} "
                    f"(eval success_rate={success_rate:.2f} >= {args.q_curriculum_start_success_rate:.2f})"
                )

            if args.wandb:
                log_payload = {"eval/success_rate": success_rate, "episode": episode}
                if saved_videos:
                    for i, video_path in enumerate(saved_videos):
                        log_payload[f"eval/video_{i}"] = wandb.Video(str(video_path), format="mp4")
                    artifact = wandb.Artifact(f"eval-videos-{args.job_name}-ep{episode+1}", type="evaluation")
                    for video_path in saved_videos:
                        artifact.add_file(str(video_path))
                    wandb.log_artifact(artifact)
                wandb.log(log_payload)

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

    # Training budget
    p.add_argument("--total_episodes", type=int, default=1000)
    p.add_argument("--warmup_episodes", type=int, default=20)
    p.add_argument("--eval_freq", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_videos_to_save", type=int, default=1, help="Number of evaluation episodes to save as videos each eval")

    # Hyperparameters
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_capacity", type=int, default=10_000)
    p.add_argument("--G", type=int, default=5, help="Gradient steps per chunk transition (paper)")
    p.add_argument("--beta", type=float, default=0.1, help="VLA regularization weight")
    p.add_argument("--ref_action_dropout_prob", type=float, default=0.5, help="Probability of dropping the VLA reference action during actor training")
    p.add_argument("--actor_output_variance", type=float, default=0.1, help="Exploration noise variance added to actor outputs during rollout")
    p.add_argument("--q_loss_weight_max", type=float, default=1.0, help="Maximum weight on the critic term in actor loss")
    p.add_argument("--q_loss_weight_increment", type=float, default=0.1, help="Size of each staircase jump in critic-term weight after first success")
    p.add_argument("--q_curriculum_start_success_rate", type=float, default=0.3, help="Evaluation success-rate threshold required to start the Q/dropout curriculum")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate")
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--sparse_reward", action="store_true", help="Use binary success indicator as reward instead of dense Metaworld reward")
    p.add_argument("--time_penalty", action="store_true", help="Use -1 per step and 0 on success instead of the environment reward")

    # WandB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="rlt-smolvla")
    p.add_argument("--wandb_entity", type=str, default="idl_34")
    p.add_argument("--job_name", type=str, default="rlt-ac")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
