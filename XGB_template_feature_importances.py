'''
Info: this is a template script where it is shown how to do a features importances with XGBoost
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
from xgboost import plot_importance
pd.set_option('display.max_columns', 500)
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
train_size_ = 0.8 # train-test split
val_size_ = 0.2 #train-validation split

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

n_repeats_ = 20

n_best_features = 3

## inputs for features creation
ll_ = [7, 14, 30, 90, 180]
vals_rsi = ll_ #[7, 30, 90, 180]#[30, 180]#[7, 30, 90, 180]
vals_sma = ll_ #[14, 30, 90, 120, 180] #[30, 180]#
vals_pctChange = ll_ #[7, 30, 90, 180] # [30, 180]#
vals_std = ll_ #[30, 90, 120, 180] #[30, 180]#
vals_so = ll_ #[30, 90, 120, 180] # [30, 180]#[30, 90, 120, 180]

dm = Data_manager(coin, s_date, where=where_)
dm.download_price_data()

lag_ls_ = [1]#, 2, 3, 4, 5, 6, 7]
dm.features_engineering_RSI(vals_rsi, lag_ls=lag_ls_) #([val1, val2])
dm.features_engineering_SMA(vals_sma) #([val1, val2])
dm.features_engineering_PCTCHANGE(vals_pctChange, lag_ls=lag_ls_) #([val1, val2])
dm.features_engineering_ONBALVOL()
dm.features_engineering_STD(vals_std, lag_ls=lag_ls_)
dm.features_engineering_STOCHOSC(vals_so, lag_ls=lag_ls_)
dm.features_engineering_HIGHLOW(lag_ls=lag_ls_)

dm.add_returns_to_data()
dm.dropna_in_data()
data = dm.get_data()
feature_names = dm.feature_names
ret_1 = dm.ret_1d

X = data[feature_names]
y = np.where(data[ret_1] - min_ret > 0, 1, 0)

train_size = int(len(X)*train_size_)

X_train_val = X.iloc[:train_size, :]
y_train_val = y[:train_size]

X_test = X.iloc[train_size:, :]
y_test = y[train_size:]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_)

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric=eval_metric_, eta=0.3, max_depth=3,
                              colsample_bytree=0.8, reg_alpha=10, reg_lambda=10, gamma=5, min_child_weight=10,
                              subsample=0.7)

# Create a GridSearchCV object and fit it to the data
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)


# ******************************************************************************************************** #
# ******************** Feat importance by Mean Decrease impurity on train data *************************** #
plot_importance(xgb_model, max_num_features=15, title='Feature importances by MDI')
print('MDI performances:', xgb_model.get_booster().get_score(importance_type='weight'))

# ******************************************************************************************** #
# ******************** Feat importance by permutation on train data *************************** #
#calculate permutation importance for training data (see https://towardsdatascience.com/best-practice-to-calculate-and-interpret-model-feature-importance-14f0e11ee660)
result_train = permutation_importance(xgb_model, X_train, y_train, n_repeats=5, random_state=seed, scoring=scoring_)

sorted_importances_idx_train = result_train.importances_mean.argsort()
ordered_features_train = X.columns[sorted_importances_idx_train]
importances_train = pd.DataFrame(
    result_train.importances[sorted_importances_idx_train].T,
    columns=ordered_features_train,
)

# Make predictions on the validation set
y_pred = xgb_model.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score:', accuracy)

y_pred = pd.Series(y_pred, index=X_test.index, name='prediction')
returns = data.loc[X_test.index, ret_1]
result_df = pd.concat([returns, y_pred], axis=1)
result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

# ******************************************************************************************** #
# ******************** Feat importance by permutation on test data *************************** #
result_test = permutation_importance(xgb_model, X_test, y_test, n_repeats=n_repeats_, random_state=seed)

sorted_importances_idx_test = result_test.importances_mean.argsort()
importances_test = pd.DataFrame(
    result_test.importances[sorted_importances_idx_test].T,
    columns=X.columns[sorted_importances_idx_test],
)

f, axs = plt.subplots(1,2,figsize=(15,5))

importances_train.plot.box(vert=False, whis=10, ax = axs[0])
axs[0].set_title("Permutation Importances (train set)")
axs[0].axvline(x=0, color="k", linestyle="--")
axs[0].set_xlabel("Decrease in accuracy score")
axs[0].figure.tight_layout()

importances_test.plot.box(vert=False, whis=10, ax = axs[1])
axs[1].set_title("Permutation Importances (test set)")
axs[1].axvline(x=0, color="k", linestyle="--")
axs[1].set_xlabel("Decrease in accuracy score")
axs[1].figure.tight_layout()

######## Re-initialise model for best features
# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric=eval_metric_, eta=0.3, max_depth=3,
                              colsample_bytree=1, reg_alpha=5, reg_lambda=5, gamma=5, min_child_weight=10, subsample=1)

# Create a GridSearchCV object and fit it to the data
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

best_features = list(ordered_features_train[-n_best_features:])
print('best features on train set:', best_features)
##print(list(importances_train.columns.values[:n_best_features]))
X_test_best_features = X_test[best_features]
X_train_best_features = X_train[best_features]
X_val_best_features = X_val[best_features]
# Create a GridSearchCV object and fit it to the data
xgb_model.fit(X_train_best_features, y_train, eval_set=[(X_val_best_features, y_val)], verbose=0)
# Make predictions on the validation set
y_pred = xgb_model.predict(X_test_best_features)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy score:', accuracy)

y_pred = pd.Series(y_pred, index=X_test.index, name='prediction')
returns = data.loc[X_test.index, ret_1]
result_df = pd.concat([returns, y_pred], axis=1)
result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results using best features')


plt.show()