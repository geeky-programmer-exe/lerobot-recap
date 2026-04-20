"""
Actor-Critic networks and losses for RLT Phase 3 (TD3). All design decisions are documented in ACTOR_CRITIC_DESIGN.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.sac.modeling_sac import CriticHead, MLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTActorCriticConfig:
    # Network architecture
    actor_hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_dim: int = 256
    critic_hidden_layers: int = 2

    # RL hyperparameters
    tau: float = 0.005       # Polyak rate
    gamma: float = 0.99      # Discount factor
    beta: float = 0.1        # VLA reg. weight
    ref_action_dropout_prob: float = 0.5   # Paper
    actor_output_variance: float = 0.1     # Exploration noise variance

    # Dimensions — must match SmolVLAConfig
    z_rl_dim: int = 2048     # rlt_d_model
    proprio_dim: int = 32    # max_state_dim
    action_dim: int = 4      # Metaworld Sawyer: xyz + gripper
    chunk_size_rl: int = 10  # Paper

    # Training loop
    G: int = 5               # Gradient steps per env chunk-transition  (paper)
    batch_size_rl: int = 256
    replay_buffer_capacity: int = 10_000
    warmup_episodes: int = 20
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    total_episodes: int = 1_000
    eval_freq: int = 50
    eval_episodes: int = 10

    # Derived (read-only)
    @property
    def rl_state_dim(self) -> int:
        return self.z_rl_dim + self.proprio_dim  # 2080

    @property
    def action_flat_dim(self) -> int:
        return self.chunk_size_rl * self.action_dim  # 40

    @property
    def actor_input_dim(self) -> int:
        return self.rl_state_dim + self.action_flat_dim  # 2120

    @property
    def critic_input_dim(self) -> int:
        return self.rl_state_dim + self.action_flat_dim  # 2120


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class RLTActor(nn.Module):
    """Deterministic actor for TD3.

    Input : concat(z_rl [2048], proprio [32], vla_ref_flat [40]) → 2120-dim
    Output: action chunk mean reshaped to (C=10, action_dim=4)

    At rollout time, Gaussian noise (variance = actor_output_variance) is added
    to the mean to form the executed action. The mean is used directly for the
    target actor in critic-loss computation.
    """

    def __init__(self, config: RLTActorCriticConfig):
        super().__init__()
        self.config = config
        hidden_dims = [config.actor_hidden_dim] * config.actor_hidden_layers
        self.trunk = MLP(input_dim=config.actor_input_dim, hidden_dims=hidden_dims)
        self.output_layer = nn.Linear(config.actor_hidden_dim, config.action_flat_dim)

    def forward(
        self,
        z_rl: torch.Tensor,          # (B, 2048)
        proprio: torch.Tensor,       # (B, 32)
        vla_ref_flat: torch.Tensor,  # (B, 40)  — zeros if dropout applied upstream
    ) -> torch.Tensor:
        """Returns action chunk mean of shape (B, C, action_dim)."""
        x = torch.cat([z_rl, proprio, vla_ref_flat], dim=-1)
        mean = self.output_layer(self.trunk(x))
        return mean.view(-1, self.config.chunk_size_rl, self.config.action_dim)

    def select_action(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
        vla_ref_flat: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Sample action for environment rollout; clamps to [-1, 1]."""
        mean = self.forward(z_rl, proprio, vla_ref_flat)
        if add_noise:
            std = self.config.actor_output_variance ** 0.5
            mean = mean + torch.randn_like(mean) * std
        return mean.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class RLTCritic(nn.Module):
    """Twin Q-network critic for TD3.

    Two independent CriticHead instances initialised with different seeds.
    Input : concat(rl_state [2080], actor_action_flat [40]) → 2120-dim
    Output: Q1, Q2 — each scalar per batch element.
    """

    def __init__(self, config: RLTActorCriticConfig, seed1: int = 0, seed2: int = 1):
        super().__init__()
        hidden_dims = [config.critic_hidden_dim] * config.critic_hidden_layers

        with torch.random.fork_rng():
            torch.manual_seed(seed1)
            self.critic1 = CriticHead(input_dim=config.critic_input_dim, hidden_dims=hidden_dims)

        with torch.random.fork_rng():
            torch.manual_seed(seed2)
            self.critic2 = CriticHead(input_dim=config.critic_input_dim, hidden_dims=hidden_dims)

    def forward(
        self,
        rl_state: torch.Tensor,    # (B, 2080)
        action_flat: torch.Tensor, # (B, 40)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1, q2), each of shape (B,)."""
        x = torch.cat([rl_state, action_flat], dim=-1)
        return self.critic1(x).squeeze(-1), self.critic2(x).squeeze(-1)

    def q1(self, rl_state: torch.Tensor, action_flat: torch.Tensor) -> torch.Tensor:
        """Returns Q1 only — used for actor loss gradient."""
        x = torch.cat([rl_state, action_flat], dim=-1)
        return self.critic1(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Target network utilities
# ---------------------------------------------------------------------------


def polyak_update(online_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Soft Polyak update: target ← τ·online + (1−τ)·target."""
    for p, tp in zip(online_net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def make_target(net: nn.Module) -> nn.Module:
    """Deep-copy a network and freeze all its parameters."""
    target = copy.deepcopy(net)
    target.requires_grad_(False)
    return target


# ---------------------------------------------------------------------------
# TD3 losses
# ---------------------------------------------------------------------------


def compute_td3_critic_loss(
    rl_state: torch.Tensor,        # (B, 2080)
    action_flat: torch.Tensor,     # (B, 40)
    reward: torch.Tensor,          # (B,)
    next_rl_state: torch.Tensor,   # (B, 2080)
    done: torch.Tensor,            # (B,) float
    critic: RLTCritic,
    target_critic: RLTCritic,
    target_actor: RLTActor,
    config: RLTActorCriticConfig,
) -> torch.Tensor:
    """TD3 Bellman loss for both critics.

    Q_target = r + γ^C · (1−done) · min(Q1_tgt, Q2_tgt)(s', ã')
    L = MSE(Q1(s,a), Q_target) + MSE(Q2(s,a), Q_target)

    For the target actor at next states we use zeros for the VLA reference input
    (equivalent to applying dropout with p=1). This avoids re-running the VLA
    for every next state in the batch while staying consistent with the dropout
    training distribution.
    """
    with torch.no_grad():
        next_z_rl = next_rl_state[:, : config.z_rl_dim]
        next_proprio = next_rl_state[:, config.z_rl_dim :]
        next_vla_ref = torch.zeros(
            next_z_rl.shape[0], config.action_flat_dim, device=rl_state.device
        )
        next_action = target_actor(next_z_rl, next_proprio, next_vla_ref)
        next_action_flat = next_action.flatten(1).clamp(-1.0, 1.0)

        q1_tgt, q2_tgt = target_critic(next_rl_state, next_action_flat)
        q_next = torch.min(q1_tgt, q2_tgt)

        gamma_c = config.gamma ** config.chunk_size_rl
        q_target = reward + gamma_c * (1.0 - done) * q_next

    q1, q2 = critic(rl_state, action_flat)
    return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)


def compute_td3_actor_loss(
    rl_state: torch.Tensor,    # (B, 2080)
    vla_ref_flat: torch.Tensor,  # (B, 40)  VLA reference stored in replay buffer
    actor: RLTActor,
    critic: RLTCritic,
    config: RLTActorCriticConfig,
) -> torch.Tensor:
    """TD3 actor loss with VLA regularization.

    L = -Q1(s, a) + β · ||a − ã||²

    Reference action dropout is applied here (not at rollout time) so that
    gradients see both the conditioned and unconditioned actor paths.
    """
    z_rl = rl_state[:, : config.z_rl_dim]
    proprio = rl_state[:, config.z_rl_dim :]

    if torch.rand(1).item() < config.ref_action_dropout_prob:
        vla_ref_input = torch.zeros_like(vla_ref_flat)
    else:
        vla_ref_input = vla_ref_flat

    action = actor(z_rl, proprio, vla_ref_input)
    action_flat = action.flatten(1)

    q1 = critic.q1(rl_state, action_flat)
    beta_loss = F.mse_loss(action_flat, vla_ref_flat)
    return -q1.mean() + config.beta * beta_loss
