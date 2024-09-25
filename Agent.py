from Environment import environment
import time
import pandas as pd # type: ignore

df = pd.read_csv('trade_data.csv')

if __name__ == "__main__":
    env = environment(df=df, window_size=60)
    model = None
    state0 = env.reset()

    while True:
        # action = model.forward(state0)
        action = 0

        state1, reward, done = env.forward(action)
        
        if done:
            break