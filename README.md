# DQN Agent for Bipedal Walker

This project aims to adapt a Deep Q-Network (DQN) agent to learn to walk in the Bipedal Walker environment from OpenAI Gym. The continuous action space of the environment is discretized to allow the learning process using DQN.

## Overview

In this project, we implement and train a DQN agent to navigate and walk in the Bipedal Walker environment. The Bipedal Walker environment presents a challenging task as it requires the agent to control a two-legged robot to walk across a terrain. This task involves balancing, coordinating both legs, and handling uneven surfaces.

## Features

- **Discretization of Action Space**: The continuous action space of the Bipedal Walker is discretized into discrete actions, allowing the DQN to handle the environment effectively.
- **DQN Implementation**: Utilizes a neural network to approximate Q-values for state-action pairs.
- **Experience Replay Buffer**: Stores past experiences to break the correlation between consecutive samples and to improve learning stability.
- **Target Network**: Helps stabilize training by reducing the divergence between the Q-network and the target values.
- **Training and Evaluation**: Includes training scripts and evaluation modes to visualize and assess the agent's performance.

## Implementation Details

### Network Architecture

The DQN model consists of:
- **Input Layer**: Accepts the state representation from the environment.
- **Hidden Layers**: Two fully connected layers with ReLU activations.
- **Output Layer**: Outputs Q-values for each possible action.

### Training Procedure

1. **Initialization**:
   - Initialize the replay buffer, Q-network, and target network.
   - Set exploration parameters (starting exploration rate, minimum exploration rate).

2. **Action Selection**:
   - Use an epsilon-greedy policy to balance exploration and exploitation. With probability ε, select a random action; otherwise, select the action with the highest Q-value.

3. **Experience Replay**:
   - Store transitions (state, action, reward, next state, done) in the replay buffer.
   - Sample mini-batches of transitions from the replay buffer to train the Q-network.

4. **Optimization**:
   - Calculate the loss between the current Q-values and the target Q-values.
   - Perform backpropagation and update the Q-network parameters.
   - Periodically update the target network to match the Q-network.

5. **Training Episodes**:
   - Train the agent over multiple episodes, each consisting of multiple steps in the environment.
   - Save the trained model periodically and monitor the agent's performance.

### Results

The DQN agent managed to walk approximately, particularly around episodes 2100-2500. Pre-trained agents are available in the `agents` directory for evaluation.

![DQN Agent Walking](https://github.com/Paul-Chupot/Bipede_DQN/blob/main/assets/rl-video-episode-2400.gif)

## Usage

### Training and validation

To train or evaluate the DQN agent, run the following command: python main.py
