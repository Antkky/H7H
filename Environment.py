import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore
import time
import numpy as np
from enum import Enum

class Actions(Enum):
    Neutral = 0
    Buy = 1
    Sell = 2

class Positions(Enum):
    Long = 0
    Short = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class environment:
    def __init__(self, df, window_size):
        # set variables
        self.window_size = window_size
        self.step = 0
        self.position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None

        # Convert 'Time' column to datetime format
        features = ['Time', 'Open', 'High', 'Low', 'Close', 'Buys', 'Sells', 'Buy_Amount', 'Sell_Amount', 'Bid', 'Ask']
        df1 = df.copy()[features]
        df1.columns = features
        df1['Time'] = pd.to_datetime(df1['Time'])
        df1 = df1.set_index('Time')
        self.data = df1

        # Initialize a plot figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Past Hour / Close Price')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Close Price')
        self.line, = self.ax.plot([], [], color='blue')
        self.ax.grid(True)

    # returns state based on steps
    def current_data(self):
        start = self.step
        end = start + self.window_size
        features = self.data.to_numpy()[start:end]
        state = pd.DataFrame(features, columns=self.data.columns)
        return state
    
    # get entire dataframe
    def all_data(self):
        features = self.data.to_numpy()
        state = pd.DataFrame(features, columns=self.data.columns)
        return state


    def _render_frame(self):
        self.render()

    # render
    def render(self):
        start_time = time.time()
        window_size = 100
        if self._first_rendering:
            self._first_rendering = False
            plt.ioff()  # Turn on interactive mode
            self.line.set_data(self.data.index[:self.step], self.data['Close'][:self.step])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.0005)

        # Update the plot with the new step
        self.line.set_data(self.data.index[max(0, self.step - window_size):self.step], 
        self.data['Close'][max(0, self.step - window_size):self.step])
        self.ax.relim()
        self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.0005)

        end_time = time.time()

    # render entire chart
    def render_all(self, title=None):
        plt.plot(self.data['Close'])

    # moves forward and returns info
    def forward(self, action):
        self.step += 1
        new_state = self.current_data()

        reward = 10
        done = self.step >= len(self.data) - self.window_size  # Stop when reaching the end

        self._render_frame()

        return new_state, self.calculate_reward(action), done
    
    # returns current state
    def state(self):
        return self.current_data()

    # returns state after reseting steps
    def reset(self):
        self._truncated = False
        self.step = 0
        self._total_profit = 1
        self._total_reward = 0
        self._first_rendering = True
        self._render_frame()
        
        return self.current_data()
    
    def close(self):
        plt.close()
    
    def calculate_reward(self, action):
        return 5

    def update_profit(self, action):
        self._total_profit = self._total_profit + 5
        