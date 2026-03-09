
import time

import torch
import numpy as np

from homework2 import Hw2Env
from homework_2v2 import DQN


def evaluate(model_path: str = "dqn_highlevel:new_reward.pth", n_episodes: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load policy network
    policy_net = DQN(input_size=6, hidden_size=128, output_size=8).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = Hw2Env(n_actions=8, render_mode="gui")

    episode_rewards = []

    for ep in range(n_episodes):
        env.reset()
        state = torch.from_numpy(env.high_level_state().astype(np.float32)).to(device).unsqueeze(0)

        ep_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                qvals = policy_net(state)
                action = qvals.argmax(dim=1).item()

            _, reward, terminal, truncated = env.step(action)
            done = terminal or truncated
            ep_reward += reward

            next_state = torch.from_numpy(env.high_level_state().astype(np.float32)).to(device).unsqueeze(0)
            state = next_state

            # small delay so GUI updates are visible
            time.sleep(0.01)

        episode_rewards.append(ep_reward)
        print(f"[Test] Episode {ep+1:2d}/{n_episodes} reward {ep_reward:.2f}")

    print(f"Average reward over {n_episodes} episodes: {sum(episode_rewards)/len(episode_rewards):.2f}")


if __name__ == "__main__":
    evaluate()
