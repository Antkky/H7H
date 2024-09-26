import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore
import time
import numpy as np
from enum import Enum

class Actions(Enum):
    Buy = 0
    Sell = 1

class Positions(Enum):
    Long = 0
    Short = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class Environment:
    def __init__(self, df, window_size):
        # Set variables
        self._first_rendering = None
        self.window_size = window_size
        self.step = 0

        self.position = None
        self.entry_price = None  # Track the price at which a position is opened

        self._total_reward = 0
        self._realized_pnl = 0

        self._unrealized_pnl = 0  # Add tracking for unrealized PnL
        self.reward_history = []

        self.profit_history = []  # To track total profit over time
        self.unrealized_pnl_history = []  # Track unrealized PnL for plotting

        self.buy_signals = []  # Track Buy signals
        self.sell_signals = []  # Track Sell signals

        #########################################################################################################################################

        # Data Preprocessing
        features = ['Open', 'High', 'Low', 'Close', 'Buys', 'Sells', 'Buy_Amount', 'Sell_Amount', 'Bid', 'Ask']
        self.data = df.copy()[features] # Clone Features
        self.data.columns = features # sets columns attribute

        #########################################################################################################################################

        # Initialize a plot figure for PnL
        self.fig_pnl, self.ax_pnl = plt.subplots()
        self.ax_pnl.set_title('Realized and Unrealized PnL')
        self.ax_pnl.set_xlabel('Time')
        self.ax_pnl.set_ylabel('PnL', color='green')


        # Initialize a plot figure for Price
        self.fig_price, self.ax_price = plt.subplots()
        self.ax_price.set_title('Price')
        self.ax_price.set_xlabel('Time')
        self.ax_price.set_ylabel('Price', color='blue')
        

        # Create lines for realized PnL, unrealized PnL, and close price
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.ax_pnl.grid(True)
        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_price, = self.ax_price.plot([], [], color='blue', label='Close Price')

        # Initialize scatter plots for buy and sell markers
        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=150)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=150)

        # Make sure to add the legend for better visualization
        self.ax_pnl.legend()

        #########################################################################################################################################

    # Re-Render
    def render(self):
        # Check if this is the first rendering
        if self._first_rendering:
            self._first_rendering = False
            plt.ion()

            # Set initial x and y limits for PnL
            self.ax_pnl.set_xlim(self.data.index[0], self.data.index[-1])
            initial_y_min = self.data['Close'].min() * 0.95
            initial_y_max = self.data['Close'].max() * 1.05 
            self.ax_pnl.set_ylim(initial_y_min, initial_y_max)

        if self.step > 0:
            # Update PnL lines
            self.line_realized_pnl.set_data(self.data.index[:self.step], self.profit_history[:self.step])
            self.line_unrealized_pnl.set_data(self.data.index[:self.step], self.unrealized_pnl_history[:self.step])

            # Update Price line
            self.line_price.set_data(self.data.index[:self.step], self.data['Close'].iloc[:self.step])

            # Update the axes dynamically for PnL
            self.ax_pnl.relim()
            self.ax_pnl.autoscale_view()

            # Dynamically update y-limits for PnL
            current_min_y = min(self.profit_history + self.unrealized_pnl_history) * 1.05
            current_max_y = max(self.profit_history + self.unrealized_pnl_history) * 1.05
            self.ax_pnl.set_ylim(current_min_y, current_max_y)

            # Update the axes dynamically for Price
            self.ax_price.relim()
            self.ax_price.autoscale_view()

            # Plot buy and sell signals (optional)
            buy_indices = [i for i in range(len(self.buy_signals)) if self.buy_signals[i][1] > 0]
            sell_indices = [i for i in range(len(self.sell_signals)) if self.sell_signals[i][1] > 0]
            # Prepare y-values for signals
            buy_y = [self.buy_signals[i][1] for i in buy_indices] if buy_indices else []
            sell_y = [self.sell_signals[i][1] for i in sell_indices] if sell_indices else []
            # Plot buy signals
            if buy_indices:
                self.buy_scatter.set_offsets(np.column_stack((self.data.index[buy_indices].astype(np.int64) // 10**9, buy_y)))
            # Plot sell signals
            if sell_indices:
                self.sell_scatter.set_offsets(np.column_stack((self.data.index[sell_indices].astype(np.int64) // 10**9, sell_y)))

            plt.draw()
            plt.pause(0.0005)

    # Move forward
    def forward(self, action):
        self.step += 1
        new_state = self.current_data()

        reward = self.calculate_reward(action)  # Calculate the reward based on action
        done = self.step + self.window_size >= len(self.data)  # Stop when reaching the end

        # Store PnL & Reward
        self.profit_history.append(self._realized_pnl)
        self.unrealized_pnl_history.append(self._unrealized_pnl)
        self.reward_history.append(reward)

        self.render()

        return new_state, reward, self._realized_pnl, self._unrealized_pnl, done
    
    # Calculate reward and PnL
    def calculate_reward(self, action):
        current_price = self.data['Close'].iloc[self.step]
        current_time = self.data.index[self.step]

        if self.position is None:
            # No position held
            if action == Actions.Buy:
                self.position = Positions.Long
                self.entry_price = current_price
                self._unrealized_pnl = 0
                self.buy_signals.append((current_time, current_price))  # Append buy signal
                return 0
            elif action == Actions.Sell:
                self.position = Positions.Short
                self.entry_price = current_price
                self._unrealized_pnl = 0
                self.sell_signals.append((current_time, current_price))  # Append sell signal
                return 0

        elif self.position == Positions.Long:
            # If holding a long position, calculate unrealized PnL
            self._unrealized_pnl = current_price - self.entry_price

            if action == Actions.Sell:
                profit = current_price - self.entry_price
                self._realized_pnl += profit
                reward = profit
                self.sell_signals.append((current_time, current_price))  # Mark sell signal
                print(f"Closed Long at {current_price}, Realized Profit: {profit}")
                self.position = None  # Position closed
                self.entry_price = None
                self._unrealized_pnl = 0  # Reset unrealized PnL when position is closed
            else:
                reward = self._unrealized_pnl

        elif self.position == Positions.Short:
            # If holding a short position, calculate unrealized PnL
            self._unrealized_pnl = self.entry_price - current_price

            if action == Actions.Buy:
                profit = self.entry_price - current_price
                self._realized_pnl += profit
                reward = profit
                self.buy_signals.append((current_time, current_price))  # Mark buy signal
                print(f"Closed Short at {current_price}, Realized Profit: {profit}")
                self.position = None  # Position closed
                self.entry_price = None
                self._unrealized_pnl = 0  # Reset unrealized PnL when position is closed
            else:
                reward = self._unrealized_pnl

        self._total_reward += reward
        return reward
    
    #########################################################################################################################################

    def reset(self):
        self._truncated = False
        self.step = 0
        self._realized_pnl = 0
        self._unrealized_pnl = 0
        self.profit_history = [0]  # Initialize with a starting point for profit
        self._total_reward = 0
        self.position = None
        self.entry_price = None
        self._first_rendering = True
        self.render();
        
        return self.current_data()
    
    def current_data(self):
        # returns state based on steps
        start = self.step
        end = start + self.window_size
        features = self.data.to_numpy()[start:end]
        state = pd.DataFrame(features, columns=self.data.columns)
        return state
    
    def all_data(self):
        # get entire dataframe
        features = self.data.to_numpy()
        state = pd.DataFrame(features, columns=self.data.columns)
        return state
    
    def render_all(self, title=None):
        # render entire chart
        plt.plot(self.data['Close'])

    def state(self):
        # returns current state
        return self.current_data()
   
    def close(self):
        plt.close()
   
    #########################################################################################################################################