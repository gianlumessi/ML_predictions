'''
Info: this script runs a walk forward analysis, where the features used to make predictions on the test set
are extracted from a feature importances analysis carried out on the previous training set. The xgb boost runs
an automatic early stopping on the number of boosting rounds.
'''

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
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
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from datetime import datetime
#from imblearn.over_sampling import RandomOverSampler ## NOT WORKING ATM

start_time = datetime.now()
print('Process started at: ', start_time)


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
num_folds = 3
scoring = 'roc_auc'  # 'accuracy', 'precision', 'recall', 'f1'

## inputs for features creation
vals_rsi = [7, 30, 90, 180]#[30, 180]#[7, 30, 90, 180]
vals_sma = [14, 30, 90, 120, 180] #[30, 180]#
vals_pctChange = [7, 30, 90, 180] # [30, 180]#
vals_std = [30, 90, 120, 180] #[30, 180]#
vals_so = [30, 90, 120, 180] # [30, 180]#[30, 90, 120, 180]


## params for the walk-forward testing
fold_size_y = 4
fold_size = int(fold_size_y*365) #Walk-forward test fold size
train_pct_wf = 0.75 #For walk forward testing
train_pct_early_stop = 0.8 # Used  for the early stopping


########### Model / Estimator ###############
early_stopping_rounds_ = 15
eval_metric_ = 'auc'
## For available eval_metrics see: https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
#“rmse” for root mean squared error.
#“mae” for mean absolute error.
#“logloss” for binary logarithmic loss and “mlogloss” for multi-class log loss (cross entropy).
#“error” for classification error.
#“auc” for area under ROC curve.
xgb_model = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds_, eval_metric=eval_metric_)
seed = np.random.randint(0, 1000)

n_best_features = 3

##check_oversample = True #to allow resampling if classes are imbalanced

dm = Data_manager(coin, s_date, where=where_)

dm.download_price_data()


lag_ls_ = [1, 2, 3, 4, 5, 6, 7]
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

param_grid = {
    'random_state': [seed],
    'eta': [0.3], #learning rate
    'gamma': [10],# [0, 1, 2, 5, 10],
    'max_depth': [3], #[2, 3, 4, 5]
    'colsample_bytree': [1], ## want to use all features to help with features importance
    'alpha': [10], #[0, 1, 2, 5, 10], ##L1 regularization
    'lambda': [10], #[0, 1, 2, 5, 10], ##L2 regularization
    'min_child_weight': [10], # [1, 5, 10, 20],
    'subsample': [0.9]
    }


print('Feature names:')
print(dm.feature_names, '\n')

print(data.head())

frames = []
outer_i = 0
fwts = For_test_split.Walk_forward_nonanchored_ts_split(fold_size, train_pct_wf, data)
for train_index, test_index in fwts.split(leftover=0.25):
    print('\nFold', outer_i)
    train_dates = data.index[train_index]
    test_dates = data.index[test_index]
    print('Train fold:', train_dates[0], train_dates[-1], '. # of samples:', len(train_index))
    print('Test fold:', test_dates[0], test_dates[-1],  '. # of samples:', len(test_index))
    fwts.plot_sets(train_dates, test_dates, outer_i)

    # Reinitialise X and Y since they may be changed during the loop (sure they are changed???)
    X = data[feature_names]
    Y = np.sign(data[ret_1] - min_ret)  # including min_ret
    Y = Y.replace(-1, 0)  # to have labels otherwise sklearn thinks you are doing a multi-label classification
    X_train_val_fold = X.loc[train_dates]
    y_train_val_fold = Y.loc[train_dates]

    #if check_oversample:
    #    print('Resampling training population...')
    #    ros = RandomOverSampler()  # (random_state=seed)
    #    X_train_val_fold, y_train_val_fold = ros.fit_resample(X_train_val_fold, y_train_val_fold)

    # Split the data into training, validation and test sets
    split_point = int(len(X_train_val_fold) * train_pct_early_stop)
    X_train_fold = X_train_val_fold.iloc[:split_point, :] #, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
    y_train_fold = y_train_val_fold[:split_point]
    ## Validation set for early stopping
    X_val_fold = X_train_val_fold.iloc[split_point:, :]
    y_val_fold = y_train_val_fold[split_point:]

    ## Test fold: unseen data by the model
    X_test_fold = X.loc[test_dates]
    y_test_fold = Y.loc[test_dates]

    ## 1. Standard GRidSearch
    grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring,
                        cv=TimeSeriesSplit(n_splits=num_folds).get_n_splits([X_train_fold, y_train_fold]))
    #cv_results_df = xgb.cv(dtrain=dtrain_data, params=param_grid, metrics=scoring, as_pandas=True, nfold=num_folds, seed=seed)

    clone_grid = clone(grid)
    clone_grid.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=0)

    #optimized_GBM.best_estimator_.feature_importances_
    importances = clone_grid.best_estimator_.feature_importances_ # clone_grid.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in clone_grid.best_estimator_.estimators_], axis=0)

    importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print('Feature importances:')
    print(importances, '\n')
    best_features = importances[:n_best_features]
    print('Best features:')
    print(best_features, '\n')

    #fig, ax = plt.subplots()
    #forest_importances.plot.bar(yerr=std, ax=ax)
    #ax.set_title("Feature importances using MDI")
    #ax.set_ylabel("Mean decrease in impurity")
    #fig.tight_layout()
    #plt.show()

    X_train_fold_best_features = X_train_fold.loc[:, best_features.index.values]
    X_val_fold_best_features = X_val_fold.loc[:, best_features.index.values]
    X_test_fold_best_features = X_test_fold.loc[:, best_features.index.values]

    #grid_best_feat = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring,
    #                    cv=TimeSeriesSplit(n_splits=num_folds).get_n_splits([X_train_fold_best_features, y_train_fold]))

    #grid_best_feat.fit(X_train_fold_best_features, y_train_fold, eval_set=[(X_val_fold_best_features, y_val_fold)])

    model = clone_grid.best_estimator_
    model.fit(X_train_fold_best_features, y_train_fold, eval_set=[(X_val_fold_best_features, y_val_fold)], verbose=0)
    y_pred = model.predict(X_test_fold_best_features)

    y_pred = pd.Series(y_pred, index=X_test_fold.index, name='prediction')
    returns = data.loc[y_test_fold.index, ret_1]
    tmp_df = pd.concat([returns, y_pred], axis=1)
    frames.append(tmp_df)

    outer_i+=1

result_df = pd.concat(frames)

result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

end_time = datetime.now()
print('Process ended at: ', start_time)
print('Elapsed time:', end_time - start_time)

plt.show()