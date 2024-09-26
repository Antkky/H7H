from Environment import *
from Model import *
import torch # type: ignore
import random
import pandas as pd # type: ignore
from collections import deque
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore
import tracemalloc

# Constants for memory and training
MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.05

class Agent:
    def __init__(self, df):
        self.window_size = 30
        

        self.n_trades = 0
        self.epsilon = 1.0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LSTM_Q_Net(input_size=10, hidden_size=256, output_size=2).to(device)   # Initializes  Neural Network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, BATCH_SIZE=BATCH_SIZE) # Initializes         Trainer
        self.env = Environment(df=df, window_size=self.window_size)                         # Initializes     Environment
        
        self.buy_signals = []
        self.sell_signals = []

    def get_action(self, state):
        state_tensor = torch.tensor(state.values, dtype=torch.float).unsqueeze(0).to(device)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
            
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            print("Not enough experiences to sample")
            return  # Not enough experiences to sample

        mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to Tensors and reshape
        states = torch.tensor(np.array(states), dtype=torch.float).to(device).view(BATCH_SIZE, self.window_size, 10)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device).view(BATCH_SIZE, self.window_size, 10)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def update_epsilon(self):
        if self.epsilon > 0.01:  # Min epsilon
            self.epsilon *= 0.995  # Decay epsilon

    def save(self, episode):
        name = "Episode-" + str(episode) + ".pth"
        self.model.save(name)

    def run(self, episodes):
        for episode in range(episodes):
            print(f"Episode {episode + 1}")
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action0 = self.get_action(state)
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
            self.update_epsilon()
            self.save(episode)
            print(f"Episode: {episode}\nReward: {total_reward}\nProfit: {unrealprofit}")



if __name__ == "__main__":
    df = pd.read_csv('trade_data.csv')
    # vv Data preprocessing vv

    features = df.drop(columns=["Time"])

    min_max_scaler = MinMaxScaler()
    scaled_minmax = min_max_scaler.fit_transform(features)

    df_minmax = pd.DataFrame(scaled_minmax, columns=features.columns)

    print(df_minmax)

    # ^^ Data preprocessing ^^
    agent = Agent(df_minmax)
    agent.run(episodes=100)