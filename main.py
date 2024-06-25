import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import os
from time import time

from gym.utils.save_video import save_video
from collections import deque

# Hyperparameters
episodes = 3000
max_steps = 700
batch_size = 32
buffer_size = 100000
learning_rate = 1e-03
starting_exp_rate = 1.
min_exp_rate = .05
gamma = .99
sync_every = 1000
render_every = 50

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

class Buffer:
    '''
    Buffer class to save and sample agent exploration for training
    '''
    def __init__(self, buffer_size):
        """
        Initialize the buffer with a maximum size.

        Args:
            buffer_size (int): Maximum size of the buffer.
        """
        self.memory = deque(maxlen=buffer_size)


    def save(self, state, action, next_state, reward, done):
        """
        Save a transition (state, action, next_state, reward, done) in the buffer.

        Args:
            state (array): Current state.
            action (array): Action taken.
            next_state (array): Next state.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """

        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        self.memory.append((state, action, next_state, reward, done))


    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Size of the batch to sample.

        Returns:
            tuple of tensors: Batch of states, actions, next_states, rewards, and dones.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.stack(rewards).to(device)
        dones = torch.stack(dones).to(device)

        return states, actions, next_states, rewards, dones

    def clear(self):
        """
        Clear the buffer.
        """
        self.memory.clear()


buffer = Buffer(buffer_size)


class DQN(nn.Module):
    '''
    DQN model with a reshaped output to discretize the action space
    with for each articulation either max speed in a direction or the other or neutral
    '''

    def __init__(self, n_observation, n_actions):
        """
        Initialize the DQN model.

        Args:
            n_observation (int): Number of observations.
            n_actions (int): Number of actions.
        """
        super().__init__()
        self.layer1 = nn.Linear(n_observation,48)
        self.layer2 = nn.Linear(48,64)
        self.layer3 = nn.Linear(64,n_actions)

    def forward(self, states):
        """
        Forward pass of the DQN model.

        Args:
            states (torch.Tensor): Input states.

        Returns:
            torch.Tensor: Output Q-values.
        """
        x = F.relu(self.layer1(states))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1,3,4)


class DQNAgent:
    '''
    DQN Agent class
    '''
    def __init__(self, n_observation, n_actions):
        """
        Initialize the DQN agent with value network, target network, and parameters.

        Args:
            n_observation (int): Number of observations.
            n_actions (int): Number of actions.
        """

        self.net = DQN(n_observation, n_actions).to(device)
        self.opt = torch.optim.Adam(params=self.net.parameters(), lr=learning_rate)
        self.target = DQN(n_observation, n_actions).to(device)
        self.target.load_state_dict(self.net.state_dict())

        self.exploration_rate = starting_exp_rate
        self.min_exp_rate = min_exp_rate
        self.gamma = gamma
        self.sync_every = sync_every

        self.total_steps = 0
        self.episodes = episodes
        self.max_steps = max_steps

        self.batch_size = batch_size

    @torch.no_grad()
    def act(self, state, validation_mode=False):
        """
        Select an action based on the current state using an epsilon-greedy policy.

        Args:
            state (array): Current state.
            validation_mode (bool): Whether to use validation mode.

        Returns:
            array: Selected action.
        """

        self.total_steps += 1

        if self.total_steps % self.sync_every == 0 :
            self.target.load_state_dict(self.net.state_dict())
            print('Target updated ლ(╹◡╹ლ)', '\n')


        if random.random() < self.exploration_rate and not validation_mode:
            action = np.random.randint(0,3,size=4)
        else :
            state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            action = self.net(state).squeeze(0).argmax(dim=0).numpy()

        self.exploration_rate = max(min_exp_rate, self.exploration_rate - 1/(self.episodes * self.max_steps/2))

        return action

    def optimize(self, states, actions, next_states, rewards, dones):
        """
        Optimize the DQN using a batch of transitions.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            next_states (torch.Tensor): Batch of next states.
            rewards (torch.Tensor): Batch of rewards.
            dones (torch.Tensor): Batch of done flags.
        """

        current_values = self.net(states).gather(1,actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            expected_values = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * self.target(next_states).max(1).values

        criterion = nn.SmoothL1Loss()
        loss=criterion(current_values, expected_values)

        print(current_values, expected_values)

        self.opt.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.opt.step()

    def save(self, reward, episode):
        """
        Save the current model.

        Args:
            reward (float): Reward achieved.
            episode (int): Current episode.
        """
        torch.save(self.net.state_dict(), f"agents/saved_model_{round(reward)}_episode_{episode}.pt")

    def load(self, path):
        """
        Load a model from a file.

        Args:
            path (str): Path to the model file.
        """
        data = torch.load(path, map_location=device)
        self.net.load_state_dict(data)


if __name__=='__main__':

    choice = 0

    while choice not in [1,2]:
        choice = int(input("Choose between : 1. Training  2. Load and evaluate"))

    if choice == 1:

        env = gym.make("BipedalWalker-v3",hardcore=False, render_mode='rgb_array_list')

        agent = DQNAgent(24,12)

        history = {'total_rwds':[],
                'max_heights':[],
                'max_velocities':[]}

        for episode in range(episodes):

            start = time()

            ep_rwds = []
            ep_velocities = []

            state, _ = env.reset()


            for step in range(max_steps):

                action = agent.act(state)

                next_state, reward, terminated, truncated, _ = env.step(action  - np.array([1.,1.,1.,1.]))

                done = terminated or truncated

                buffer.save(state, action, next_state, reward, done)

                ep_rwds.append(reward)

                state = next_state

                if episode > 5:
                    states, actions, next_states, rewards, dones = buffer.sample(batch_size)
                    agent.optimize(states, actions, next_states, rewards, dones)


                if done or step == max_steps-1:
                    save_video(
                        env.render(),
                        "replays",
                        fps=env.metadata["render_fps"],
                        episode_trigger= lambda x: x%render_every == 0,
                        episode_index=episode
                    )
                    break

            total_rwd = sum(ep_rwds)

            if total_rwd > 5:
                agent.save(total_rwd, episode)

            if (episode+1) % 10 == 0 or total_rwd>0:
                print('-'*100)
                print(f'Episode {episode+1}/{episodes} | total reward = {total_rwd} | exp rate = {agent.exploration_rate}', '\n')


            history['total_rwds'].append(total_rwd)

            print(f'ep duration : {time()-start:.2f}')

        print('Mean reward :')
        print(np.mean(history['total_rwds']))
        print('Max reward :')
        print(np.max(history['total_rwds']))


        plt.plot(history['total_rwds'])
        plt.show()


    else:
        agents = os.listdir("agents")

        if 'archive' in agents:
            agents.remove('archive')

        print("Agents available:")

        for i, a in enumerate(agents):
            print(f'{i}: {a}')

        c = int(input("Choose :"))

        agent = DQNAgent(24,12)

        agent.load(f'agents/{agents[c]}')

        env = gym.make("BipedalWalker-v3", hardcore=False, render_mode='human')

        state, _ = env.reset()

        ep_rwds = []
        actions = []

        for step in range(max_steps +900):

            action = agent.act(state, validation_mode = True)

            next_state, reward, terminated, truncated, _ = env.step(action- np.array([1.,1.,1.,1.]))

            state = next_state

            actions.append(action)
            ep_rwds.append(reward)

            done = terminated or truncated

            if done :
                break

        total_rwd = sum(ep_rwds)


        print('-'*100)
        print(f'Final Results ( ͡°( ͡° ͜ʖ( ͡° ͜ʖ ͡°)ʖ ͡°) ͡°) | total reward = {total_rwd}', '\n')
        # print(actions)
