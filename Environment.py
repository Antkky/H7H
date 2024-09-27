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

        self.fig_reward, self.ax_reward = plt.subplots()
        self.ax_reward.set_title('Reward & Profit')
        self.ax_reward.set_xlabel('Time')
        self.ax_reward.set_ylabel('PnL', color='green')
        

        # Create lines for realized PnL, unrealized PnL, and close price
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_price, = self.ax_price.plot([], [], color='blue', label='Close Price')

        # Initialize scatter plots for buy and sell markers
        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=25)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=25)

        self.line_realized_pnl_reward, = self.ax_reward.plot([], [], color='green', label="Realized PnL")
        self.line_reward, = self.ax_reward.plot([], [], color='red', label='Reward')

        self.ax_reward.grid(True)
        self.ax_pnl.grid(True)
        self.ax_price.grid(True)

        # Make sure to add the legend for better visualization
        self.ax_pnl.legend()

        #########################################################################################################################################

    def render(self):
        # Check if this is the first rendering
        if self._first_rendering:
            self._first_rendering = False
            plt.ion()
            plt.show(block=False)

        if self.step > 0:
            # Update PnL lines
            self.line_realized_pnl.set_data(self.data.index[:self.step], self.profit_history[:self.step])
            self.line_unrealized_pnl.set_data(self.data.index[:self.step], self.unrealized_pnl_history[:self.step])

            # Update Price line
            self.line_price.set_data(self.data.index[:self.step], self.data['Close'].iloc[:self.step])

            self.line_reward.set_data(self.data.index[:self.step], self.reward_history[:self.step])
            self.line_realized_pnl_reward.set_data(self.data.index[:self.step], self.profit_history[:self.step])

            # Update the axes dynamically for PnL
            self.ax_pnl.relim()
            self.ax_pnl.autoscale_view()

            # Dynamically update y-limits for PnL
            current_min_y = min(self.profit_history + self.unrealized_pnl_history) * 1.05
            current_max_y = max(self.profit_history + self.unrealized_pnl_history) * 1.05
            if current_min_y == current_max_y:
                current_max_y += 1
            self.ax_pnl.set_ylim(current_min_y, current_max_y)

            # Update the axes dynamically for Price
            self.ax_price.relim()
            self.ax_price.autoscale_view()

             # Update the axes dynamically for PnL
            self.ax_reward.relim()
            self.ax_reward.autoscale_view()

            # Plot buy and sell signals
            buy_indices = [i for i in range(len(self.buy_signals)) if self.buy_signals[i][1] > 0]
            sell_indices = [i for i in range(len(self.sell_signals)) if self.sell_signals[i][1] > 0]

            # Extract timestamps and y-values for buy/sell signals
            buy_times = [self.buy_signals[i][0] for i in buy_indices]
            buy_prices = [self.buy_signals[i][1] for i in buy_indices]

            sell_times = [self.sell_signals[i][0] for i in sell_indices]
            sell_prices = [self.sell_signals[i][1] for i in sell_indices]

            # Set offsets for buy and sell signals (timestamps and corresponding price)
            if buy_indices:
                self.buy_scatter.set_offsets(np.column_stack((buy_times, buy_prices)))
            if sell_indices:
                self.sell_scatter.set_offsets(np.column_stack((sell_times, sell_prices)))

            # Draw and pause for dynamic updates
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
                reward = self._realized_pnl
                self.sell_signals.append((current_time, current_price))  # Mark sell signal
                print(f"Closed Long at {current_price}, Realized Profit: {profit}")
                self.position = None  # Position closed
                self.entry_price = None
                self._unrealized_pnl = 0  # Reset unrealized PnL when position is closed
            else:
                reward = self._realized_pnl

        elif self.position == Positions.Short:
            # If holding a short position, calculate unrealized PnL
            self._unrealized_pnl = self.entry_price - current_price

            if action == Actions.Buy:
                profit = self.entry_price - current_price
                self._realized_pnl += profit
                reward = self._realized_pnl
                self.buy_signals.append((current_time, current_price))  # Mark buy signal
                print(f"Closed Short at {current_price}, Realized Profit: {profit}")
                self.position = None  # Position closed
                self.entry_price = None
                self._unrealized_pnl = 0  # Reset unrealized PnL when position is closed
            else:
                reward = self._realized_pnl

        self._total_reward += reward
        return reward
    
    #########################################################################################################################################

    def reset(self):
        plt.close(self.fig_pnl)
        plt.close(self.fig_price)
        plt.close(self.fig_reward)
        # Reset internal variables
        self._truncated = False
        self.step = 0
        self._realized_pnl = 0
        self._unrealized_pnl = 0
        self.profit_history = []  # Initialize with a starting point for profit
        self.reward_history = []
        self.unrealized_pnl_history = []
        self._total_reward = 0
        self.position = None
        self.entry_price = None
        self._first_rendering = True  # Ensure chart is re-initialized
        self.render();

        # Recreate figures for new episode
        self.fig_pnl, self.ax_pnl = plt.subplots()
        self.ax_pnl.set_title('Realized and Unrealized PnL')
        self.ax_pnl.set_xlabel('Time')
        self.ax_pnl.set_ylabel('PnL', color='green')

        self.fig_price, self.ax_price = plt.subplots()
        self.ax_price.set_title('Price')
        self.ax_price.set_xlabel('Time')
        self.ax_price.set_ylabel('Price', color='blue')

        self.fig_reward, self.ax_reward = plt.subplots()
        self.ax_reward.set_title('Reward & Profit')
        self.ax_reward.set_xlabel('Time')
        self.ax_reward.set_ylabel('PnL', color='green')

        # Recreate line and scatter objects for new episode
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.ax_pnl.grid(True)
        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_price, = self.ax_price.plot([], [], color='blue', label='Close Price')

        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=25)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=25)

        self.line_realized_pnl_reward, = self.ax_reward.plot([], [], color='green', label="Realized PnL")
        self.line_reward, = self.ax_reward.plot([], [], color='red', label='Reward')

        self.ax_reward.grid(True)

        # Ensure the legends are added
        self.ax_pnl.legend()

        self.render()
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
        plt.close(self.fig_pnl)
        plt.close(self.fig_price)
    #########################################################################################################################################