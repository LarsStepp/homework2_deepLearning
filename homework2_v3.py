import time
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import environment

class Hw2Env_adapted(environment.BaseEnv):
    def __init__(self, n_actions=8, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._n_actions = n_actions
        self._delta = 0.05

        # currently we only control x/y translation of the end-effector.
        # Rotation actions can be added later if desired.
        theta = np.linspace(0, 2*np.pi, n_actions)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}

        self._goal_thresh = 0.01
        self._max_timesteps = 50

        # used to compute object velocity for reward shaping
        self._prev_obj_pos = None

    def reset(self):
        super().reset()
        # remember object position so we can compute object velocity for shaping
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        # Reward combines (a) closeness to object, (b) closeness of object to goal,
        # and (c) whether the object is actually moving toward the goal.
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)

        base_reward = 1.0/(ee_to_obj) + 1.0/(obj_to_goal)

        # velocity reward: positive if the object moves toward the goal, negative if it moves away
        reward_dir = 0.0
        if self._prev_obj_pos is not None:
            obj_vel = obj_pos - self._prev_obj_pos
            dir_goal = goal_pos - obj_pos
            dist_goal = np.linalg.norm(dir_goal)
            if dist_goal > 1e-6:
                dir_goal /= dist_goal
                reward_dir = float(np.dot(obj_vel, dir_goal))

        # scale the direction reward to be on a comparable scale
        reward_dir = np.clip(reward_dir, -1.0, 1.0) * 7.0
        return base_reward + reward_dir

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action_id):
        # does this have to be changed for the rotational actions?
        action = self._actions[action_id] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])

        # include rotation in the action execution
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()

        # update previous object position for the next step
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()

        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state/next_state expected to be torch tensors with shape (1, obs_dim)
        self.buffer.append((state.detach(), action, reward, next_state.detach(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # --- hyperparameters ---
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_DECAY = 0.999
    EPSILON_DECAY_ITER = 10
    MIN_EPSILON = 0.1
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    UPDATE_FREQ = 4
    TARGET_NETWORK_UPDATE_FREQ = 100
    BUFFER_LENGTH = 10000
    NUM_EPISODES = 1000

    N_ACTIONS = 8
    OBS_DIM = 6
    HIDDEN_SIZE = 128

    env = Hw2Env_adapted(n_actions=N_ACTIONS, render_mode="offscreen")

    policy_net = DQN(input_size=OBS_DIM, hidden_size=HIDDEN_SIZE, output_size=N_ACTIONS).to(device)
    target_net = DQN(input_size=OBS_DIM, hidden_size=HIDDEN_SIZE, output_size=N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(BUFFER_LENGTH)

    epsilon = EPSILON
    step_count = 0

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        env.reset()
        state = torch.from_numpy(env.high_level_state().astype(np.float32)).to(device=device).unsqueeze(0)
        episode_reward = 0.0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    qvals = policy_net(state)
                    action = qvals.argmax(dim=1).item()

            _, reward, terminal, truncated = env.step(action)
            done = terminal or truncated
            episode_reward += reward

            next_state = torch.from_numpy(env.high_level_state().astype(np.float32)).to(device=device).unsqueeze(0)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            step_count += 1

            if step_count % UPDATE_FREQ == 0 and len(replay_buffer) >= BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(BATCH_SIZE)

                q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_b).max(dim=1)[0]
                    targets = rewards_b + GAMMA * next_q * (1.0 - dones_b)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if step_count % EPSILON_DECAY_ITER == 0:
                epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        episode_rewards.append(episode_reward)

        if episode % 1 == 0:
            print(f"Episode {episode:4d} reward {episode_reward:.2f} epsilon {epsilon:.3f}")

    # --- after training: save model & plot rewards ---
    torch.save(policy_net.state_dict(), "dqn_highlevel_new_reward.pth")
    print("Saved model to dqn_highlevel_new_reward.pth")

    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label="episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN training rewards")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_rewards_reward_adapted.png")
    print("Saved reward plot to training_rewards_reward_adapted.png")

    print("training finished")
