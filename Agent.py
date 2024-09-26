from Environment import *
from Model import *
import torch # type: ignore
import random
import pandas as pd # type: ignore
from collections import deque
import tracemalloc

# Load trade data

# Constants for memory and training
MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.5

class Agent:
    def __init__(self, environment):
        self.env = environment
        self.n_trades = 0
        self.epsilon = 1.0  # Initial randomness (exploration)
        self.gamma = 0.9    # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory
        self.model = LSTM_Q_Net(10, 256, 2).to(device)  # Adjust input size as necessary
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, BATCH_SIZE=BATCH_SIZE)
        
        self.buy_signals = []  # This should remain a list
        self.sell_signals = []  # This should remain a list # Access sell_signals from the environment

    def get_action(self, state):
        state_tensor = torch.tensor(state.values, dtype=torch.float).unsqueeze(0).to(device)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough experiences to sample

        mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to Tensors and reshape
        states = torch.tensor(np.array(states), dtype=torch.float).to(device).view(BATCH_SIZE, 60, 10)  # Adjust as necessary
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device).view(BATCH_SIZE, 60, 10)  # Adjust as necessary
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def update_epsilon(self):
        if self.epsilon > 0.01:  # Min epsilon
            self.epsilon *= 0.995  # Decay epsilon

    def run(self, episodes):
        for episode in range(episodes):
            print(f"Episode {episode + 1}")
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action0 = self.get_action(state)  # Select action
                match action0:
                    case 0:
                        action1 = Actions.Buy
                    case 1:
                        action1 = Actions.Sell
                

                state, reward, realprofit, unrealprofit, done = self.env.forward(action1)  # Interact with the environment
                self.store_experience(state.values, action0, reward, state.values, done)  # Store experience
                self.train()  # Train the agent
                total_reward += reward
            
            self.env.close()
            self.update_epsilon()  # Update exploration rate after each episode
            print(f"Episode: {episode}\nReward: {total_reward}\nProfit: {realprofit}")



if __name__ == "__main__":
    df = pd.read_csv('trade_data.csv')
    env = environment(df, window_size=60)  # Assuming your environment class is named `Environment`
    agent = Agent(env)
    agent.run(episodes=100)