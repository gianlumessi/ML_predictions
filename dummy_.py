import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


signal = [1,1,1,-1,-1,1,-1,-1,-1]

df = pd.DataFrame(data=signal, columns=['signal'])
df['signal_shifted'] = df['signal'].shift(1)
df['diff'] = df['signal'] - df['signal_shifted']
sig_arr = np.where(df['diff'] != 0, -1, 1)

if df['signal'].iloc[0] == 1:
    sig_arr[0] = 1
else:
    sig_arr[0] = -1

print(df)
print(sig_arr)
