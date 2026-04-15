import numpy as np
import time
import os
import pickle
from typing import Callable, Optional

from core.rl_agent import BootstrappedDQNAgent
from core.environment import ClinicalTrialEnvironment


def train(
    agent: BootstrappedDQNAgent,
    env: ClinicalTrialEnvironment,
    n_episodes: int = 400,
    progress_callback: Optional[Callable] = None,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """
    Train the BDQL++ agent.

    progress_callback(ep, total, reward, loss, epsilon, arm_counts) called every episode.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float("-inf")
    start_time  = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        agent.rotate_head()
        ep_reward = 0.0
        ep_losses = []

        # Each episode = one patient, multiple treatment steps
        for _ in range(20):
            action = agent.selectAction(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward
            agent.arm_counts[action] = agent.arm_counts[action] + 1
            agent.steps_done += 1

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

        agent.decay_epsilon()

        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        agent.episode_rewards.append(ep_reward)
        agent.episode_losses.append(mean_loss)
        agent.epsilons.append(agent.epsilon)

        # Head certainty (avg std across heads as proxy for uncertainty)
        q_sample = agent.online.predict(np.random.randn(10, agent.state_dim).astype(np.float32))
        q_stack  = np.vstack([q.mean(axis=1) for q in q_sample])
        certainty = float(1.0 - q_stack.std() / (np.abs(q_stack).mean() + 1e-6))
        agent.head_certainty.append(np.clip(certainty, 0, 1))

        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pkl"))

        # Progress callback for Streamlit live update
        if progress_callback:
            progress_callback(
                ep=ep,
                total=n_episodes,
                reward=ep_reward,
                loss=mean_loss,
                epsilon=agent.epsilon,
                arm_counts=agent.arm_counts[:],
                elapsed=time.time() - start_time,
            )

    elapsed = time.time() - start_time
    agent.save(os.path.join(checkpoint_dir, "final_model.pkl"))

    return {
        "best_reward"  : best_reward,
        "total_time_s" : elapsed,
        "episodes"     : n_episodes,
        "final_epsilon": agent.epsilon,
        "rewards"      : agent.episode_rewards,
        "losses"       : agent.episode_losses,
    }


def randomBaseline(env: ClinicalTrialEnvironment, n_episodes: int = 400) -> dict:
    """Run a random assignment baseline for comparison."""
    import random
    rewards = []
    arm_counts = [0] * env.n_actions

    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0.0
        for _ in range(20):
            action = random.randrange(env.n_actions)
            state, reward, done, _ = env.step(action)
            ep_r += reward
            arm_counts[action] += 1
        rewards.append(ep_r)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward" : float(np.std(rewards)),
        "arm_counts" : arm_counts,
        "rewards"    : rewards,
    }
