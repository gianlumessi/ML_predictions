from Data_handler import Data_manager
import pandas as pd
pd.set_option('display.max_columns', 500)

pair = 'BTC-USD'
start_date = '2015-01-01'
search_term = 'bitcoin'
date_split = '2021-01-01'

lags_p_daily_rets = [1, 7] #[1, 7, 28]
lags_rets = [5, 60]
lags_sma = [5] # [7, 28, 90]
lags_std = None #[5, 7]
lags_rsi = [28, 60] #[20, 28, 60] #
lags_search = [5, 20, 60]
lags_search_sma = [5, 20, 60]

dm = Data_manager(pair, start_date)
dm.download_price_data()
dm.merge_search_with_price_data(search_term)
dm.features_engineering(lags_p_drets=lags_p_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std,
                             lags_rsi=lags_rsi, lags_search=lags_search, lags_search_sma=lags_search_sma)

print('Features:', dm.feature_cols)
print(dm.df.tail())

X_train, Y_train, X_test, Y_test = dm.split_train_test(date_split)