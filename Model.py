import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
import os

class LSTM_Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 3, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)

        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # This should work if lstm_out is 3D

        # Pass through fully connected layer
        x = self.fc(lstm_out)

        return x

    def save(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma, BATCH_SIZE):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.batch_size = BATCH_SIZE

    def train_step(self, state, action, reward, next_state, done):
        # Ensure correct shape for LSTM (batch_size, sequence_length, input_size)
        if state.dim() == 2:  # Assuming input shape (batch_size, input_size)
            state = state.view(-1, 30, 11)  # Adjust based on your actual input sizes
        if next_state.dim() == 2:
            next_state = next_state.view(-1, 30, 11)  # Adjust accordingly

        # Q values for current state
        pred = self.model(state)  # Output: [batch_size, num_actions]
        target = pred.clone()

        for idx in range(len(done)):
            with torch.no_grad():
                Q_next = self.model(next_state[idx].unsqueeze(0)).max()  # Add unsqueeze to create a batch dimension
            if done[idx]:
                target[idx][action[idx]] = reward[idx]  # No future reward if done
            else:
                target[idx][action[idx]] = reward[idx] + self.gamma * Q_next

        # Loss
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

