import gym
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(),
                                     nn.Linear(256, nb_actions), )

    def forward(self, x):
        return self.network(x / 255.)


def Deep_Q_Learning(env, replay_memory_size=1_000_000, nb_epochs=30_000_000, update_frequency=4, batch_size=32,
                    discount_factor=0.99, replay_start_size=80_000, initial_exploration=1, final_exploration=0.01,
                    exploration_steps=1_000_000, device='cuda'):
    # Initialize replay memory D to capacity N
    rb = ReplayBuffer(replay_memory_size, env.observation_space, env.action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    # Initialize action-value function Q with random weights
    q_network = DQN(env.action_space.n).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4)

    epoch = 0
    smoothed_rewards = []
    rewards = []

    progress_bar = tqdm(total=nb_epochs)
    while epoch <= nb_epochs:

        dead = False
        total_rewards = 0

        # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
        obs = env.reset()

        for _ in range(random.randint(1, 30)):  # Noop and fire to reset environment
            obs, _, _, info = env.step(1)

        while not dead:
            current_life = info['lives']

            epsilon = max((final_exploration - initial_exploration) / exploration_steps * epoch + initial_exploration,
                          final_exploration)
            if random.random() < epsilon:  # With probability ε select a random action a
                action = np.array(env.action_space.sample())
            else:  # Otherwise select a = max_a Q∗(φ(st), a; θ)
                q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            # Execute action a in emulator and observe reward rt and image xt+1
            next_obs, reward, dead, info = env.step(action)

            done = True if (info['lives'] < current_life) else False

            # Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)  # Reward clipping

            # Store transition (φt, at, rt, φt+1) in D
            rb.add(obs, real_next_obs, action, reward, done, info)

            obs = next_obs

            if epoch > replay_start_size and epoch % update_frequency == 0:
                # Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
                data = rb.sample(batch_size)
                with torch.no_grad():
                    max_q_value, _ = q_network(data.next_observations).max(dim=1)
                    y = data.rewards.flatten() + discount_factor * max_q_value * (1 - data.dones.flatten())
                current_q_value = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.huber_loss(y, current_q_value)

                # Perform a gradient descent step according to equation 3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch += 1
            if (epoch % 50_000 == 0) and epoch > 0:
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title("Average Reward on Breakout")
                plt.xlabel("Training Epochs")
                plt.ylabel("Average Reward per Episode")
                plt.savefig('Imgs/average_reward_on_breakout.png')
                plt.close()

            progress_bar.update(1)
        rewards.append(total_rewards)


if __name__ == "__main__":
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = MaxAndSkipEnv(env, skip=4)

    Deep_Q_Learning(env, device='cuda')
    env.close()
