'''
Info: this script runs a walk forward analysis, where at each iteration of the walk-forward, you loop through the
params of the features over the train fold in order to find the optimal ones to be used to make preditions over the
test fold.
'''

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import quantstats
import seaborn as sns
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 500)
import Train_test_split as For_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
from itertools import product
import sys
import logging
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")#("%d-%m-%Y_%H-%M-%S")

########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## inputs for features creation
val1_list = [7]#, 14, 30, 60, 90, 120, 180, 360]
val2_list = [30]#, 60, 90, 180, 360, 720]
param_value_ls = [val1_list, val2_list]
constraint = lambda v1, v2: v2 <= v1

## inputs for cross-=validation
num_folds = 4
scoring = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'

## inputs for walk-forward testing
fold_size_y = 6
fold_size = int(fold_size_y*365) #Walk-forward test fold size
train_pct = 0.75 #For walk forward testing
first_test_index = max(val2_list) + int(fold_size * train_pct) + 1

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'
where_ = 'yahoo'
min_ret = tc

########### Model / Estimator ###############
model_ = 'Random_forest_classifier'
model_dict = {model_: [RandomForestClassifier(), 'Random forest']}
seed = np.random.randint(0, 1000)
param_grid = {
    'n_estimators': [200],#, 400, 800],
    'max_depth': [2],#, 3, 4],
    'criterion': ['gini'],  # , 'entropy'],
    'random_state': [seed],
    'max_features': [len(param_value_ls)], #use all features since they are only 2
    'min_samples_leaf': [0.05],
    'max_samples': [0.9]
}

verbose = True

dm = Data_manager(coin, s_date, where=where_)

output_folder = 'C:/Users/Gianluca/Desktop/strategy_results/Approach_2/' + coin + '/' + model_ + '/' + dt_string + \
                '_foldsz_' + str(fold_size_y) + 'y/'


if os.path.exists(output_folder):
    answer = input('Folder ' + output_folder + ' already exists. Do you want to overwrite its content? ...')
    if answer.lower() == 'no' or answer.lower() == 'n':
        print('Script terminated, please use a new folder')
        sys.exit()
else:
    os.mkdir(output_folder)


logging.basicConfig(filename=output_folder+'log_file.txt',
                    filemode='a',  # set it to append rather than overwrite
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    # determine the format of the output message (format=...)
                    datefmt='%H:%M:%S',
                    level=logging.INFO)  # and determine the minimum message level it will accept

logging.info('PROGRAM STARTED ...')
logging.info('Pair used for testing = ' + coin)
logging.info('param 1 values: ' + str(val1_list))
logging.info('param 2 values: ' + str(val2_list))
logging.info('# of folds for cross validation = ' + str(num_folds))
logging.info('Score to select fittest model ' + scoring)
logging.info('Fold size for walk-forward testing ' + str(fold_size))
logging.info('Train pct = ' + str(train_pct))
logging.info('First test index (when to start the first test fold of the walk-forward) = ' + str(first_test_index))
logging.info('Transaction costs (binance commissions + slippage) = ' + str(tc))
logging.info('Strategy type = ' + st)
logging.info('Model = ' + model_)
logging.info('Param grid for cross validation = ' + str(param_grid))

print('Model = ' + model_)
print('Param grid for cross validation = ' + str(param_grid))

########## Download data ###############
dm.download_price_data(output_folder + coin + '_raw_data.xlsx')
logging.info('Data for '+ coin +' downloaded from: ' + where_)

raw_data = dm.get_data()
fwts = For_test_split.Walk_forward_nonanchored_ts_split(fold_size, train_pct, raw_data, first_test_index=first_test_index)
print('Test starts at:', raw_data.index[first_test_index])
logging.info('First test fold of walkforward starts at: ' + str(raw_data.index[first_test_index]))

frames = []
outer_i = 0
for train_index, test_index in fwts.split(leftover=0.25):
    print('\nFold', outer_i)
    logging.info('Fold:' + str(outer_i))
    train_dates = raw_data.index[train_index]
    test_dates = raw_data.index[test_index]
    print('Train fold:', train_dates[0], train_dates[-1], '. # of samples:', len(train_index))
    print('Test fold:', test_dates[0], test_dates[-1],  '. # of samples:', len(test_index))
    logging.info('Train fold: ' + str(train_dates[0]) + ' ' + str(train_dates[-1]) + '. # of samples:' + str(len(train_index)))
    logging.info('Test fold: ' + str(test_dates[0]) + ' ' + str(test_dates[-1]) + '. # of samples:' + str(len(test_index)))


    fwts.plot_sets(train_dates, test_dates, outer_i)

    params_ls = []
    estimator_ls = []
    score_ls = []
    #X_test_fold_ls = []

    for p_vals in product(*param_value_ls):
        if constraint(p_vals[0], p_vals[1]):
            continue
        print('Parameters of the strategy:', p_vals[0], p_vals[1])
        logging.info('Parameters of the strategy:' + str(p_vals[0]) +' '+ str(p_vals[1]))

        # set the master_df = raw_df
        dm.reset_data()
        ########## Create features ###############
        dm.features_engineering_SMA([p_vals[0], p_vals[1]])
        dm.add_returns_to_data()
        dm.dropna_in_data()
        feature_cols = dm.feature_names

        ## For a given [train, test] set:
            ## For a given set of params:
                ## Calculate features and split data set into train and test set.
        data = dm.get_data()
        ret_1 = dm.ret_1d
        X = data[feature_cols]
        Y = np.sign(data[ret_1]-min_ret) #including min_ret
        Y = Y.replace(-1, 0) # to have labels otherwise sklearn thinks you are doing a multi-label classification
        X_train_fold = X.loc[train_dates]
        y_train_fold = Y.loc[train_dates]
        X_test_fold = X.loc[test_dates]
        y_test_fold = Y.loc[test_dates]

        ## 1. Standard GRidSearch
        grid = GridSearchCV(estimator=model_dict[model_][0], param_grid=param_grid, scoring=scoring, cv=num_folds)

        clone_grid = clone(grid)
        clone_grid.fit(X_train_fold, y_train_fold)

        params_ls.append([p_vals[0], p_vals[1]])
        estimator_ls.append(clone_grid.best_estimator_)
        score_ls.append(clone_grid.best_score_)

    tmp_d = {'Params': params_ls, 'Estimator': estimator_ls, 'Score': score_ls}
    df = pd.DataFrame(tmp_d)
    #Sort by best score
    df = df.sort_values(by='Score', axis=0, ascending=False)
    best_model = df.iloc[0].loc['Estimator']
    best_score = df.iloc[0].loc['Score']
    best_params = df.iloc[0].loc['Params']
    #best_X_test_fold = X_test_fold_ls[df.index[0]]

    y_pred = best_model.predict(X_test_fold)

    if verbose:
        ## Results:
        print('Fold ' + str(outer_i) + '. Test dates:', X_test_fold.index[0], ' - ', X_test_fold.index[-1])
        print('Best estimator on train set:', best_model) #, ', best score:', best_score)
        print('Best params:', best_params)
        print('Accuracy on test set:\t', accuracy_score(y_test_fold, y_pred))
        print('Recall on test set:\t', recall_score(y_test_fold, y_pred))
        print('Precision on test set:\t', precision_score(y_test_fold, y_pred))
        print('f1 on test set:\t', f1_score(y_test_fold, y_pred), '\n')

    logging.info('Best estimator on train set:' + str(best_model)) #, ', best score:', best_score)
    logging.info('Best params:' + str(best_params))
    logging.info('Accuracy on test set:\t' + str(accuracy_score(y_test_fold, y_pred)))
    logging.info('Recall on test set:\t' + str(recall_score(y_test_fold, y_pred)))
    logging.info('Precision on test set:\t' + str(precision_score(y_test_fold, y_pred)))
    logging.info('f1 on test set:\t' + str(f1_score(y_test_fold, y_pred)))

    y_pred = pd.Series(y_pred, index=X_test_fold.index, name='prediction')
    returns = data.loc[y_test_fold.index, ret_1]
    tmp_df = pd.concat([returns, y_pred], axis=1)
    frames.append(tmp_df)

    outer_i +=1

logging.info('::: Summary results in overall test set :::')
print('::: Summary results in overall test set :::')

# Save figure with train and test sets
plt.savefig(output_folder + 'train_and_test_sets.png')

result_df = pd.concat(frames)

result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

fig.savefig(output_folder + 'strategy_performance_on_test_set.png')
plt.close()

realised_Y = Y.loc[result_df.index]
predictions = result_df.loc[:, 'prediction']

logging.info('Summary of predictions in overall test set:')
logging.info(str(predictions.value_counts()))

logging.info('Confusion matrix:')
logging.info(confusion_matrix(realised_Y, predictions))

print('- Confusion matrix:\n', confusion_matrix(realised_Y, predictions))

overall_accuracy = accuracy_score(realised_Y, predictions)
overall_recall = recall_score(realised_Y, predictions)
overall_precision = precision_score(realised_Y, predictions)
overall_f1 = f1_score(realised_Y, predictions)

strategy_rets = result_data['strategy']
cagr = quantstats.stats.cagr(strategy_rets)
strat_cum_ret = result_data['cum_strategy'].iloc[-1] / result_data['cum_strategy'].iloc[0] - 1
sharpe = quantstats.stats.sharpe(strategy_rets)
sortino = quantstats.stats.sortino(strategy_rets)
kelly = quantstats.stats.kelly_criterion(strategy_rets)
rruin = quantstats.stats.risk_of_ruin(strategy_rets)
drawdown_series = quantstats.stats.to_drawdown_series(strategy_rets)
drawdown_df = quantstats.stats.drawdown_details(drawdown_series)
max_drawdown = drawdown_df['max drawdown'].min()
longest_dd_time = drawdown_df['days'].max()

coin_price_ts = result_data.loc[:, 'cum_return']
coin_rets = result_data.loc[:, 'ret_']
coin_cagr = quantstats.stats.cagr(coin_rets)
coin_cum_ret = coin_price_ts[-1] / coin_price_ts[0] - 1
coin_sharpe = quantstats.stats.sharpe(coin_rets)
coin_sortino = quantstats.stats.sortino(coin_rets)
coin_kelly = quantstats.stats.kelly_criterion(coin_rets)
coin_rruin = quantstats.stats.risk_of_ruin(coin_rets)
coin_drawdown_series = quantstats.stats.to_drawdown_series(coin_rets)
coin_drawdown_df = quantstats.stats.drawdown_details(coin_drawdown_series)
coin_max_drawdown = coin_drawdown_df['max drawdown'].min()
coin_longest_dd_time = coin_drawdown_df['days'].max()


summary_dict = {'val1': [str(val1_list), coin], 'val2': [str(val2_list), 'Buy and hold'], 'accuracy': [overall_accuracy, 'N/A'],
                'f1': [overall_f1, 'N/A'],
                'recall': [overall_recall, 'N/A'], 'precision': [overall_precision, 'N/A'], 'seed': [seed, 'N/A'],
                'cagr': [cagr, coin_cagr],
                'strategy_cum_ret': [strat_cum_ret, coin_cum_ret], 'sharpe': [sharpe, coin_sharpe], 'sortino': [sortino, coin_sortino],
                'kelly_criterion': [kelly, coin_kelly], 'risk_of_ruin': [rruin, coin_rruin], 'max_drawdown': [max_drawdown, coin_max_drawdown],
                'longest_dd_time': [longest_dd_time, coin_longest_dd_time]}

static_summary_dict = {'binance_commissions': binance_commissions, 'slippage': slippage, 'strategy_type': st,
                       'model': model_dict[model_][1], 'fold_size': fold_size, 'train_pct': train_pct,
                       'num_folds_cv': num_folds, 'optimisation_metric_cv': scoring, 'min return': min_ret}

key_ls = param_grid.keys()
cols = []
for key in key_ls:
    cols.append(pd.Series(param_grid[key], name=key))
model_params_df = pd.concat(cols, axis=1)
summary_result_df = pd.DataFrame(summary_dict)
static_df = pd.DataFrame(static_summary_dict, index=[0])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(output_folder + 'summary_of_results.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
summary_result_df.to_excel(writer, sheet_name='Result summary')
static_df.to_excel(writer, sheet_name='Static data')
model_params_df.to_excel(writer, sheet_name='Model params')

writer.close()

result_data.to_excel(output_folder + 'strategy_predictions.xlsx')

logging.info('PROGRAM ENDED')

print('The end')
