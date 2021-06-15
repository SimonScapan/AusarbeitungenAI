'''
Buy Low Sell High or momentum strategy bot for educational purposes.
The project is based on: https://www.datacamp.com/community/tutorials/finance-python-trading

The aim is to find a model for trading with a "buy low sell high" heuristic to compare later on with a high level intelligence
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


########

# Initialize the short and long windows
short_window = 40
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=df_full.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = df_full['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = df_full['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)


########

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
df_full['Close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
         
# Show the plot
plt.show()