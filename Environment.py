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

        self.winning_streak = 0
        self.losing_streak = 0
        self.streak_rewarded = False

        #########################################################################################################################################

        # Data Preprocessing
        features = ['Open', 'High', 'Low', 'Close', 'Buys', 'Sells', 'Buy_Amount', 'Sell_Amount', 'Bid', 'Ask']
        self.data = df.copy()[features] # Clone Features
        self.data.columns = features # sets columns attribute

        #########################################################################################################################################

        # Initialize a plot figure for Price
        self.fig_price, self.ax_price = plt.subplots()

        self.ax_price.set_title('Price')
        self.ax_price.set_xlabel('Time')
        self.ax_price.set_ylabel('Price', color='blue')

        self.ax_pnl = self.ax_price.twinx()

        # Create lines for realized PnL, unrealized PnL, and close price
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_price, = self.ax_price.plot([], [], color='black', label='Close Price')

        # Initialize scatter plots for buy and sell markers
        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=25)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=25)

        self.line_reward, = self.ax_pnl.plot([], [], color='red', label='Reward')

        self.ax_price.grid(True)

        # Make sure to add the legend for better visualizaton
        self.ax_price.legend()
        self.ax_pnl.legend()
        #########################################################################################################################################

    def render(self):
        # Check if this is the first rendering
        if self._first_rendering:
            self._first_rendering = False
            plt.ion()
            #plt.show(block=False)

        if self.step > 0:
            done = False
            while not done:
                try:
                    # Update PnL lines
                    self.line_realized_pnl.set_data(self.data.index[:self.step], self.profit_history[:self.step])
                    self.line_unrealized_pnl.set_data(self.data.index[:self.step], self.unrealized_pnl_history[:self.step])

                    # Update Price line
                    self.line_price.set_data(self.data.index[:self.step], self.data['Close'].iloc[:self.step])

                    self.line_reward.set_data(self.data.index[:self.step], self.reward_history[:self.step])


                    # Update the axes dynamically for Price
                    self.ax_price.relim()
                    self.ax_price.autoscale_view()
                    self.ax_pnl.relim()
                    self.ax_pnl.autoscale_view()

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
                    plt.pause(0.005)
                    done = True

                except ValueError:
                    continue
            

    def forward(self, action):
        self.step += 1
        new_state = self.current_data()

        reward = self.execute(action) # Execute Action
        done = self.step + self.window_size >= len(self.data)  # Stop when reaching the end

        # Store PnL & Reward
        self.profit_history.append(self._realized_pnl)
        self.unrealized_pnl_history.append(self._unrealized_pnl)
        self.reward_history.append(self._total_reward)

        self.render()

        return new_state, reward, self._realized_pnl, self._unrealized_pnl, done

    def execute(self, action):
        current_price = self.data['Close'].iloc[self.step]
        current_time = self.data.index[self.step]

        reward = 0

        if self.position is None:
            if action == Actions.Buy:
                self.long_entry(current_price, current_time)

            elif action == Actions.Sell:
                self.short_entry(current_price, current_time)

        elif self.position == Positions.Long:
            self._unrealized_pnl = current_price - self.entry_price
            if action == Actions.Sell:
                reward += self.long_exit(current_price, current_time)
        
        elif self.position == Positions.Short:
            self._unrealized_pnl = self.entry_price - current_price
            if action == Actions.Buy:
                reward += self.short_exit(current_price, current_time)
            
        self._total_reward += reward
        return reward
    
    #########################################################################################################################################

    def reset(self):
        plt.close(self.fig_price)
        # Reset internal variables
        self._truncated = False
        self.step = 0
        self._realized_pnl = 0
        self._unrealized_pnl = 0
        self.profit_history = []
        self.reward_history = []
        self.unrealized_pnl_history = []
        self._total_reward = 0
        self.position = None
        self.entry_price = None
        self._first_rendering = True
        self.winning_streak = 0
        self.losing_streak = 0
        self.streak_rewarded = False

        # Initialize a plot figure for Price
        self.fig_price, self.ax_price = plt.subplots()

        self.ax_price.set_title('Price')
        self.ax_price.set_xlabel('Time')
        self.ax_price.set_ylabel('Price', color='blue')

        self.ax_pnl = self.ax_price.twinx()

        # Create lines for realized PnL, unrealized PnL, and close price
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_price, = self.ax_price.plot([], [], color='black', label='Close Price')

        # Initialize scatter plots for buy and sell markers
        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=25)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=25)

        self.line_reward, = self.ax_pnl.plot([], [], color='red', label='Reward')

        self.ax_price.grid(True)

        # Make sure to add the legend for better visualizaton
        self.ax_price.legend()
        self.ax_pnl.legend()

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
        print("closing")

    #########################################################################################################################################
    
    def long_entry(self, price, time):
        self.position = Positions.Long
        self.entry_price = price
        self._unrealized_pnl = 0
        self.buy_signals.append((time, price))
        print(f"Opened Long")

    def short_entry(self, price, time):
        self.position = Positions.Short
        self.entry_price = price
        self._unrealized_pnl = 0
        self.sell_signals.append((time, price))
        print(f"Opened Short")

    def long_exit(self, price, time):
        profit = price - self.entry_price
        self._realized_pnl += profit
        reward = 0

        if profit > 0:
            reward += self.winning_trade() # 10
            if profit > 25:
                reward += self.big_winning_trade() # 20
            elif profit > 50:
                reward += 50 
                
        elif profit < 0:
            reward += self.losing_trade() # -10
            if profit < -25:
                reward += self.big_losing_trade() # -20
            elif profit < -50:
                reward += -50
    
        # Log
        self.sell_signals.append((time, price))  # Mark sell signal
        print(f"Closed Long, Realized Profit: {profit}, Total Profit: {self._realized_pnl}, Total Reward: {self._total_reward}")

        # Close Position
        self.position = None
        self.entry_price = None
        self._unrealized_pnl = 0

        return reward

    def short_exit(self, price, time):
        profit = self.entry_price - price
        self._realized_pnl += profit
        reward = 0
        
        if profit > 0:
            reward += self.winning_trade() # 10
            if profit > 25:
                reward += self.big_winning_trade() # 20
            elif profit > 50:
                reward += 50 
                
        elif profit < 0:
            reward += self.losing_trade() # -10
            if profit < -25:
                reward += self.big_losing_trade() # -20
            elif profit < -50:
                reward += -50

        # Log
        self.buy_signals.append((time, price))
        print(f"Closed Short, Realized Profit: {profit}, Total Profit: {self._realized_pnl}, Total Reward: {self._total_reward}")

        # Close Position
        self.position = None
        self.entry_price = None
        self._unrealized_pnl = 0

        return reward
    
    #########################################################################################################################################

    def winning_trade(self):
        return 10

    def losing_trade(self):
        return -10
    
    def big_winning_trade(self):
        return 20
    
    def big_losing_trade(self):
        return -20