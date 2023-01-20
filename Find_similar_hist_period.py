'''
Info: the objective of this script is to look at the most recent n-month period and find a similar period in the past.
- Train a ML model on the most recent n-month period and run a features importances analysis
- Use the best features to make predictions on past n-month historical periods
- See which historical periods the model makes good predictions
'''


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import quantstats
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from pylab import mpl, plt
from Data_manager import Data_manager
from sklearn.base import clone
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('display.max_columns', 500)
from datetime import datetime
import Train_test_split as For_test_split


# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")#("%d-%m-%Y_%H-%M-%S")

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'
where_ = 'yahoo'
min_ret = tc

recent_period = ['2022-01-01', '2022-12-31']

########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## inputs for cross-=validation
num_folds = 4
scoring = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'


## inputs for features creation
vals_rsi = [30, 180]#[7, 30, 90, 180]
vals_sma = [30, 180]#[14, 30, 90, 120, 180]
vals_pctChange = [30, 180]#[7, 30, 90, 180]
vals_std = [30, 180]#[30, 90, 120, 180]
vals_so = [30, 180]#[30, 90, 120, 180]

fold_size_y = 5
fold_size = int(fold_size_y*365) #Walk-forward test fold size
train_pct = 0.75 #For walk forward testing


########### Model / Estimator ###############
model_ = 'Random_forest_classifier'
model_dict = {model_: [RandomForestClassifier(), 'Random forest']}
seed = np.random.randint(0, 1000)
n_best_features = 4

dm = Data_manager(coin, s_date, where=where_)

dm.download_price_data()

dm.features_engineering_RSI(vals_rsi) #([val1, val2])
dm.features_engineering_SMA(vals_sma) #([val1, val2])
dm.features_engineering_PCTCHANGE(vals_pctChange) #([val1, val2])
dm.features_engineering_ONBALVOL()
dm.features_engineering_STD(vals_std)
dm.features_engineering_STOCHOSC(vals_so)

dm.add_returns_to_data()
dm.dropna_in_data()
data = dm.get_data()
feature_names = dm.feature_names
ret_1 = dm.ret_1d

param_grid = {
    'n_estimators': [800],#, 800],#, 400, 800],
    'max_depth': [3],#, 3],#, 4],
    'criterion': ['gini'],  # , 'entropy'],
    'random_state': [seed],
    #'max_features': [len(feature_names)], #use all features since they are only 2
    'min_samples_leaf': [0.1],
    'max_samples': [0.8]
}

print('Feature names:')
print(dm.feature_names, '\n')

print(data.head())

frames = []
outer_i = 0


psp = For_test_split.Period_search_split(data, recent_period)
train_dates = psp.get_train_period()
X_train = data.loc[train_dates, feature_names]
Y_train = np.sign(data.loc[train_dates, ret_1] - min_ret)  # including min_ret
Y_train = Y_train.replace(-1, 0)  # to have labels otherwise sklearn thinks you are doing a multi-label classification
print(X_train.head())
print('train_period:')
print(train_dates)

## 1. Standard GRidSearch
grid = GridSearchCV(estimator=model_dict[model_][0], param_grid=param_grid, scoring=scoring, cv=num_folds)
grid.fit(X_train, Y_train)

# optimized_GBM.best_estimator_.feature_importances_
importances = grid.best_estimator_.feature_importances_  # clone_grid.feature_importances_
std = np.std([tree.feature_importances_ for tree in grid.best_estimator_.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print('Feature importances:')
print(forest_importances, '\n')
best_features = forest_importances[:n_best_features]
print('Best features:')
print(best_features, '\n')
X_train_best_features = X_train.loc[:, best_features.index.values]
grid_best_feat = grid.fit(X_train_best_features, Y_train)

for test_dates in psp.get_test_periods():
    print('\nFold', outer_i)
    print('Test dates:')
    print(test_dates)
    X_test_fold_best_features = data.loc[test_dates, best_features.index.values]
    Y_test_fold = np.sign(data.loc[test_dates, ret_1] - min_ret)  # including min_ret
    Y_test_fold = Y_test_fold.replace(-1, 0)  # to have labels otherwise sklearn thinks you are doing a multi-label classification

    y_pred = grid_best_feat.predict(X_test_fold_best_features)

    y_pred = pd.Series(y_pred, index=X_test_fold_best_features.index, name='prediction')
    returns = data.loc[Y_test_fold.index, ret_1]
    tmp_df = pd.concat([returns, y_pred], axis=1)
#    frames.append(tmp_df)

    result_data = dm.run_simple_backtest(tmp_df, trading_costs=tc, strategy_type=st)
    fig = Utils.plot_oos_results(result_data, 'Out of sample results')
    plt.show()

    outer_i +=1