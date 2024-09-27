import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
import os

class LSTM_Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3, bidirectional=True):
        super(LSTM_Q_Net, self).__init__()


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=5, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional)
        
        self.ln = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        # Pass input through LSTM
        lstm_out, _ = self.lstm(x)

        # Get the last time step's output (many-to-one approach)
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size * 2) if bidirectional, else (batch_size, hidden_size)

        # Apply layer normalization
        lstm_out = self.ln(lstm_out)

        # Pass through the fully connected layer
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
        self.target_model = LSTM_Q_Net(11, 64, 2) # Target model
        self.update_target()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.MSELoss()
        self.batch_size = BATCH_SIZE

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        # Ensure correct shape for LSTM (batch_size, sequence_length, input_size)
        if state.dim() == 2:
            state = state.view(-1, 30, 11)
        if next_state.dim() == 2:
            next_state = next_state.view(-1, 30, 11)


        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            with torch.no_grad():
                Q_next = self.target_model(next_state[idx].unsqueeze(0)).max()
            if done[idx]:
                target[idx][action[idx]] = reward[idx]
            else:
                target[idx][action[idx]] = reward[idx] + self.gamma * Q_next

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

