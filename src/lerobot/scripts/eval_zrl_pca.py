"""
Collect z_rl tokens from successful and failed episodes with controlled peg/hole positions,
then visualize with PCA and t-SNE.

Strategy:
  1. Reset once with a fixed base seed to obtain MetaWorld's internal random vector
     (encodes initial peg position + goal/hole position).
  2. For each episode, add a small Gaussian perturbation to the base vector and inject
     it directly via env._env._last_rand_vec + _freeze_rand_vec = True.
     This keeps peg/hole roughly in the same place across episodes while giving
     slight variation so the policy produces both successes and failures.

Usage:
    python eval_zrl_pca.py \
        --vla_checkpoint /path/to/phase1 \
        --rlt_checkpoint /path/to/phase2 \
        --task peg-insert-side-v3 \
        --n_episodes 40 \
        --perturb_std 0.02 \
        --output_dir ./zrl_pca_out
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lerobot.envs.metaworld import MetaworldEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.train_actor_critic_rlt import extract_rl_state_and_vla_ref
from lerobot.utils.io_utils import write_video


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_patched_vla(vla_checkpoint: str, rlt_checkpoint: str, device: torch.device) -> SmolVLAPolicy:
    """Phase 2 VLA with Phase 1 action head patched back in."""
    vla = SmolVLAPolicy.from_pretrained(rlt_checkpoint)
    phase1 = SmolVLAPolicy.from_pretrained(vla_checkpoint)
    for name in ["action_in_proj", "action_out_proj", "action_time_mlp_in",
                 "action_time_mlp_out", "state_proj"]:
        getattr(vla.model, name).load_state_dict(getattr(phase1.model, name).state_dict())
    del phase1
    vla.eval().to(device)
    return vla


# ---------------------------------------------------------------------------
# Position control
# ---------------------------------------------------------------------------

def get_base_rand_vec(env: MetaworldEnv, base_seed: int) -> np.ndarray:
    """Reset once with base_seed and return MetaWorld's random position vector."""
    env._env._freeze_rand_vec = False
    env._env.reset(seed=base_seed)
    base_vec = env._env._last_rand_vec.copy()
    env._env._freeze_rand_vec = True  # freeze for subsequent resets
    return base_vec


def set_rand_vec(env: MetaworldEnv, vec: np.ndarray) -> None:
    """Inject a specific random vector so the next reset uses exact positions."""
    env._env._last_rand_vec = vec.copy()
    env._env._freeze_rand_vec = True


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(vla, preprocessor, postprocessor, env, rand_vec, device, save_frames=False):
    """Run one episode with the given rand_vec controlling peg/hole positions."""
    set_rand_vec(env, rand_vec)
    raw_obs, _ = env.reset()
    vla.reset()

    z_rl_list = []
    frames = [] if save_frames else None
    done = False

    if save_frames:
        frames.append(env.render())

    while not done:
        obs = preprocess_observation(raw_obs)
        obs["task"] = env.task_description
        obs = preprocessor(obs)

        z_rl, _, vla_ref = extract_rl_state_and_vla_ref(vla, obs, device)
        z_rl_list.append(z_rl.squeeze(0).cpu().numpy())  # (rlt_d_model,)

        action_chunk = postprocessor(vla_ref)

        for step_i in range(vla.config.chunk_size):
            raw_obs, _, terminated, truncated, info = env.step(
                action_chunk[0, step_i].cpu().numpy()
            )
            if save_frames:
                frames.append(env.render())
            done = terminated or truncated
            if done:
                break

    success = bool(info.get("is_success", False))
    return {
        "z_rl": np.stack(z_rl_list),   # (T, rlt_d_model)
        "success": success,
        "rand_vec": rand_vec.copy(),
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_embeddings(coords, labels, title, out_path, dim_names=("Dim 0", "Dim 1")):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["tab:blue" if s else "tab:red" for s in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colours, alpha=0.65, s=25)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="Success"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",  markersize=8, label="Failure"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_title(title)
    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vla = load_patched_vla(args.vla_checkpoint, args.rlt_checkpoint, device)
    preprocessor, postprocessor = make_pre_post_processors(
        vla.config, pretrained_path=args.vla_checkpoint
    )
    env = MetaworldEnv(task=args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")

    # --- Determine base peg/hole configuration ---
    base_vec = get_base_rand_vec(env, args.base_seed)
    print(f"Base rand_vec (len={len(base_vec)}): {base_vec}")
    np.save(out_dir / "base_rand_vec.npy", base_vec)

    rng = np.random.default_rng(args.rng_seed)

    # --- Collect episodes ---
    all_episodes = []
    for ep_idx in range(args.n_episodes):
        # Perturb base vector with small Gaussian noise
        perturbation = rng.normal(0.0, args.perturb_std, size=base_vec.shape)
        rand_vec = np.clip(
            base_vec + perturbation,
            env._env._random_reset_space.low,
            env._env._random_reset_space.high,
        )

        save = ep_idx < args.max_videos
        print(f"Episode {ep_idx+1}/{args.n_episodes} ...", end=" ", flush=True)
        ep = run_episode(vla, preprocessor, postprocessor, env,
                         rand_vec=rand_vec, device=device, save_frames=save)
        ep["episode_idx"] = ep_idx
        all_episodes.append(ep)
        status = "SUCCESS" if ep["success"] else "failure"
        print(status)

        if save and ep["frames"] and args.max_videos > 0:
            vid_path = out_dir / f"ep_{ep_idx:03d}_{status}.mp4"
            write_video(str(vid_path), ep["frames"], fps=env.metadata["render_fps"])
            print(f"  Video: {vid_path}")

    env.close()

    n_success = sum(e["success"] for e in all_episodes)
    print(f"\nTotal: {len(all_episodes)} | Successes: {n_success} | Rate: {n_success/len(all_episodes):.2f}")

    # --- Save metadata ---
    meta = [{"episode": e["episode_idx"], "success": e["success"],
              "rand_vec": e["rand_vec"].tolist()} for e in all_episodes]
    with open(out_dir / "episode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Build feature matrices ---
    z_first = np.stack([e["z_rl"][0]  for e in all_episodes])   # (N, D) — initial state
    z_last  = np.stack([e["z_rl"][-1] for e in all_episodes])   # (N, D) — final state
    z_mean  = np.stack([e["z_rl"].mean(0) for e in all_episodes]) # (N, D) — episode mean
    success_arr = np.array([e["success"] for e in all_episodes])

    # All time-steps
    z_all       = np.concatenate([e["z_rl"] for e in all_episodes], axis=0)
    success_all = np.concatenate([[e["success"]] * len(e["z_rl"]) for e in all_episodes])

    np.save(out_dir / "z_first.npy", z_first)
    np.save(out_dir / "z_last.npy",  z_last)
    np.save(out_dir / "z_mean.npy",  z_mean)
    np.save(out_dir / "z_all.npy",   z_all)
    np.save(out_dir / "success_arr.npy", success_arr)

    # --- PCA ---
    for z_mat, tag in [(z_last, "last"), (z_first, "first"), (z_mean, "mean")]:
        n_comp = min(20, z_mat.shape[0], z_mat.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(z_mat)
        z2 = pca.transform(z_mat)[:, :2]
        vr = pca.explained_variance_ratio_

        plot_embeddings(
            z2, success_arr.tolist(),
            f"PCA of z_rl ({tag} step) — PC1 {vr[0]:.1%} / PC2 {vr[1]:.1%}",
            out_dir / f"pca_{tag}_success.png",
            dim_names=[f"PC1 ({vr[0]:.1%})", f"PC2 ({vr[1]:.1%})"],
        )

        # Cumulative variance plot (only once)
        if tag == "last":
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(1, n_comp + 1), np.cumsum(vr))
            ax.axhline(0.9, color="red", linestyle="--", linewidth=0.8, label="90%")
            ax.set_xlabel("Number of PCs")
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title("z_rl PCA explained variance (last step)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "pca_variance.png", dpi=150)
            plt.close(fig)
            print(f"Saved {out_dir / 'pca_variance.png'}")

    # PCA on all time-steps
    if len(z_all) >= 4:
        pca_all = PCA(n_components=2)
        z_pca_all = pca_all.fit_transform(z_all)
        vr_all = pca_all.explained_variance_ratio_
        plot_embeddings(
            z_pca_all, success_all.tolist(),
            f"PCA z_rl (all steps) — PC1 {vr_all[0]:.1%} / PC2 {vr_all[1]:.1%}",
            out_dir / "pca_all_steps_success.png",
            dim_names=[f"PC1 ({vr_all[0]:.1%})", f"PC2 ({vr_all[1]:.1%})"],
        )

    # --- t-SNE ---
    for z_mat, tag in [(z_last, "last"), (z_first, "first"), (z_mean, "mean")]:
        n = len(z_mat)
        if n < 4:
            continue
        perp = min(args.tsne_perplexity, max(2, n // 2))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=0)
        z_tsne = tsne.fit_transform(z_mat)
        plot_embeddings(
            z_tsne, success_arr.tolist(),
            f"t-SNE z_rl ({tag} step, perp={perp})",
            out_dir / f"tsne_{tag}_success.png",
            dim_names=["t-SNE 1", "t-SNE 2"],
        )

    # t-SNE on all time-steps (subsample if large)
    max_pts = 2000
    if len(z_all) >= 4:
        if len(z_all) > max_pts:
            idx = np.random.default_rng(0).choice(len(z_all), max_pts, replace=False)
            z_sub, s_sub = z_all[idx], success_all[idx]
        else:
            z_sub, s_sub = z_all, success_all
        perp_all = min(args.tsne_perplexity, max(2, len(z_sub) // 2))
        z_tsne_all = TSNE(n_components=2, perplexity=perp_all, random_state=0).fit_transform(z_sub)
        plot_embeddings(
            z_tsne_all, s_sub.tolist(),
            f"t-SNE z_rl (all steps, n={len(z_sub)}, perp={perp_all})",
            out_dir / "tsne_all_steps_success.png",
            dim_names=["t-SNE 1", "t-SNE 2"],
        )

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vla_checkpoint", required=True, help="Phase 1 BC checkpoint path")
    p.add_argument("--rlt_checkpoint", required=True, help="Phase 2 RLT checkpoint path")
    p.add_argument("--task",           default="peg-insert-side-v3")
    p.add_argument("--n_episodes",     type=int,   default=40)
    p.add_argument("--base_seed",      type=int,   default=42,
                   help="Seed used once to determine the base peg/hole configuration")
    p.add_argument("--rng_seed",       type=int,   default=0,
                   help="Seed for the perturbation RNG")
    p.add_argument("--perturb_std",    type=float, default=0.02,
                   help="Std-dev of Gaussian noise added to the base rand_vec each episode")
    p.add_argument("--output_dir",     default="./zrl_pca_out")
    p.add_argument("--max_videos",     type=int,   default=0,
                   help="Save up to this many episode videos")
    p.add_argument("--tsne_perplexity", type=int,  default=10)
    main(p.parse_args())
