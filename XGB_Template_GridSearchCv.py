'''
Info: this is a template script where it is shown how to do a GridsearchCV with early stopping with XGBoost
'''


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data_manager import Data_manager
from pylab import plt, mpl
from Utils import Utils
from sklearn.inspection import permutation_importance
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'
where_ = 'yahoo'
min_ret = tc

seed = np.random.randint(0, 1000)

scoring_ = 'roc_auc'
eval_metric_ = 'auc'

## inputs for features creation
vals_rsi = [7, 30, 90, 180]#[30, 180]#[7, 30, 90, 180]
vals_sma = [14, 30, 90, 120, 180] #[30, 180]#
vals_pctChange = [7, 30, 90, 180] # [30, 180]#
vals_std = [30, 90, 120, 180] #[30, 180]#
vals_so = [30, 90, 120, 180] # [30, 180]#[30, 90, 120, 180]

dm = Data_manager(coin, s_date, where=where_)
dm.download_price_data()
#dm.features_engineering_RSI(vals_rsi) #([val1, val2])
#dm.features_engineering_SMA(vals_sma) #([val1, val2])
dm.features_engineering_PCTCHANGE(vals_pctChange) #([val1, val2])
dm.features_engineering_ONBALVOL()
dm.features_engineering_STD(vals_std)
dm.features_engineering_STOCHOSC(vals_so)

dm.add_returns_to_data()
dm.dropna_in_data()
data = dm.get_data()
feature_names = dm.feature_names
ret_1 = dm.ret_1d

X = data[feature_names]
y = np.where(data[ret_1] - min_ret > 0, 1, 0)
#y = np.sign(data[ret_1] - min_ret)  # including min_ret

##print('Shape of X:', X.shape)

train_size = int(len(X)*0.8)
##print('Train size:', train_size)

X_train_val = X.iloc[:train_size, :]
y_train_val = y[:train_size]

X_test = X.iloc[train_size:, :]
y_test = y[train_size:]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric=eval_metric_)

# Define the hyperparameters to be searched over
param_grid = {
    'random_state': [seed],
    'eta': [0.3], #learning rate
    'gamma': [5, 10],
    'max_depth': [3, 4], #, 4, 5, 6],
    'colsample_bytree': [1], ## want to use all features to help with features importance
    'alpha': [5, 10], ##L1 regularization
    'lambda': [5, 10], ##L2 regularization
    'min_child_weight': [10, 20],
    'subsample': [1]
}

# Create a GridSearchCV object and fit it to the data
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring=scoring_)
grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

# Print the best parameters found by the grid search
print('Best parameters:', grid_search.best_params_)

# Make predictions on the validation set
y_pred = grid_search.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Score:', accuracy)

y_pred = pd.Series(y_pred, index=X_test.index, name='prediction')
returns = data.loc[X_test.index, ret_1]
result_df = pd.concat([returns, y_pred], axis=1)
result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

plt.show()