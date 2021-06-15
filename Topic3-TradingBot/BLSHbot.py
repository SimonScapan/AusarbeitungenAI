'''
Buy Low Sell High or momentum strategy bot for educational purposes.
The project is based on: https://www.datacamp.com/community/tutorials/finance-python-trading

The aim is to find a model for trading with a "buy low sell high" heuristic to compare later on with a high level intelligence

More Detailed Information in scientific Paper linked to this Repository (in folder LaTex)
'''


############################
# import required packages #
############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                 # required for plotting results at the end
from pandas_datareader import data as pdr       # required to read stock data
import fix_yahoo_finance as yf                  # stock information source


##################
# fetch the data #
##################

# fetch data from yahoo finance - stock: INFY, duration: since 2018
yf.pdr_override()
df_full = pdr.get_data_yahoo("INFY", start="2018-01-01", end="2021-01-01").reset_index()
df_full.to_csv('INFY.csv',index=False)
df_full.head()


###############################
# Define BuyLowSellHigh Agent #
###############################

short_window = 30                                                                                                                   # Initialize the short and long windows
long_window = 100

signals = pd.DataFrame(index=df_full.index)                                                                                         
signals['signal'] = 0.0                                                                                                             # Initialize the `signals` DataFrame with the `signal` column
signals['short_mavg'] = df_full['Close'].rolling(window=short_window, min_periods=1, center=False).mean()                           # Create short simple moving average over the short window
signals['long_mavg'] = df_full['Close'].rolling(window=long_window, min_periods=1, center=False).mean()                             # Create long simple moving average over the long window
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   # Create signals
signals['positions'] = signals['signal'].diff()                                                                                     # Generate trading orders
print(signals)                                                                                                                      # Print `signals`


####################
# Plot the results #
####################


fig = plt.figure(figsize = (15,5))                                  # Initialize the plot figure
plt.title('Buy Low Sell High Bot')
ax1 = fig.add_subplot(111,  ylabel='Price in $')                    # Add a subplot and label for y-axis
df_full['Close'].plot(ax=ax1, color='r', lw=2.)                     # Plot the closing price
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)            # Plot the short and long moving averages
ax1.plot(signals.loc[signals.positions == 1.0].index,               # Plot the buy signals
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(signals.loc[signals.positions == -1.0].index,              # Plot the sell signals
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.savefig('plots/BLSH-bot.png')                                   # Save and show the plot