'''
Info: this script runs a walk forward analysis, where the features used to make predictions on the test set
are extracted from a feature importances analysis carried out on the previous training set
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
    'n_estimators': [400],#, 800],#, 400, 800],
    'max_depth': [2],#, 3],#, 4],
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
fwts = For_test_split.Walk_forward_nonanchored_ts_split(fold_size, train_pct, data)
for train_index, test_index in fwts.split(leftover=0.25):
    print('\nFold', outer_i)
    train_dates = data.index[train_index]
    test_dates = data.index[test_index]
    print('Train fold:', train_dates[0], train_dates[-1], '. # of samples:', len(train_index))
    print('Test fold:', test_dates[0], test_dates[-1],  '. # of samples:', len(test_index))
    fwts.plot_sets(train_dates, test_dates, outer_i)

    X = data[feature_names]
    Y = np.sign(data[ret_1] - min_ret)  # including min_ret
    Y = Y.replace(-1, 0)  # to have labels otherwise sklearn thinks you are doing a multi-label classification
    X_train_fold = X.loc[train_dates]
    y_train_fold = Y.loc[train_dates]
    X_test_fold = X.loc[test_dates]
    y_test_fold = Y.loc[test_dates]

    ## 1. Standard GRidSearch
    grid = GridSearchCV(estimator=model_dict[model_][0], param_grid=param_grid, scoring=scoring, cv=num_folds)

    clone_grid = clone(grid)
    clone_grid.fit(X_train_fold, y_train_fold)

    #optimized_GBM.best_estimator_.feature_importances_
    importances = clone_grid.best_estimator_.feature_importances_ # clone_grid.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clone_grid.best_estimator_.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print('Feature importances:')
    print(forest_importances, '\n')
    best_features = forest_importances[:n_best_features]
    print('Best features:')
    print(best_features, '\n')

    #fig, ax = plt.subplots()
    #forest_importances.plot.bar(yerr=std, ax=ax)
    #ax.set_title("Feature importances using MDI")
    #ax.set_ylabel("Mean decrease in impurity")
    #fig.tight_layout()
    #plt.show()

    clone_grid = clone(grid)
    X_train_fold_best_features = X_train_fold.loc[:, best_features.index.values]
    clone_grid_best_feat = clone_grid.fit(X_train_fold_best_features, y_train_fold)

    X_test_fold_best_features = X_test_fold.loc[:, best_features.index.values]
    y_pred = clone_grid_best_feat.predict(X_test_fold_best_features)

    y_pred = pd.Series(y_pred, index=X_test_fold.index, name='prediction')
    returns = data.loc[y_test_fold.index, ret_1]
    tmp_df = pd.concat([returns, y_pred], axis=1)
    frames.append(tmp_df)

    outer_i+=1

result_df = pd.concat(frames)

result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

plt.show()