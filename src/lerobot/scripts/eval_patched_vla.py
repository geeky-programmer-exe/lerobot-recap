"""
Verify that patching the Phase 1 action head onto the Phase 2 checkpoint
produces the same eval performance as the Phase 1 BC baseline.
TransformerRLT is not used; actions come purely from the VLA.
"""
import argparse
import torch
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.train_actor_critic_rlt import extract_rl_state_and_vla_ref
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Phase 2 checkpoint (has RLT weights)
    vla = SmolVLAPolicy.from_pretrained(args.rlt_checkpoint)

    # Patch in Phase 1 action head
    phase1 = SmolVLAPolicy.from_pretrained(args.vla_checkpoint)
    for name in ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]:
        getattr(vla.model, name).load_state_dict(getattr(phase1.model, name).state_dict())
    del phase1

    vla.eval().to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        vla.config, pretrained_path=args.vla_checkpoint
    )
    env = MetaworldEnv(task=args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")

    successes = 0
    for ep in range(args.n_episodes):
        raw_obs, _ = env.reset()
        vla.reset()
        done = False
        while not done:
            obs = preprocess_observation(raw_obs)
            obs["task"] = env.task_description
            obs = preprocessor(obs)

            # VLA reference action — RLT output (z_rl) is discarded
            _, _, vla_ref = extract_rl_state_and_vla_ref(vla, obs, device)
            action_chunk_exec = postprocessor(vla_ref)

            for step_i in range(vla.config.chunk_size):
                raw_obs, _, terminated, truncated, info = env.step(
                    action_chunk_exec[0, step_i].cpu().numpy()
                )
                done = terminated or truncated
                if done:
                    successes += int(info.get("is_success", False))
                    break

        print(f"Episode {ep+1}/{args.n_episodes} | success={info.get('is_success', False)}")

    print(f"\nSuccess rate: {successes}/{args.n_episodes} = {successes/args.n_episodes:.2f}")
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vla_checkpoint", required=True)
    p.add_argument("--rlt_checkpoint", required=True)
    p.add_argument("--task", default="peg-insert-side-v3")
    p.add_argument("--n_episodes", type=int, default=20)
    main(p.parse_args())
