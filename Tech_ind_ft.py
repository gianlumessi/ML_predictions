'''
Info: this script you loop through the parameters of the features and at each iteration (tuple of params) you run a
walk forward analysis ...
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## inputs for features creation
val1_list = [30]#, 90, 120, 180, 360]
val2_list = [60]#, 90]#, 90, 120, 180, 360, 720]
# Constraint for optimisation
constraint = lambda v1, v2: v2 <= v1


## inputs for cross-=validation
num_folds = 4
scoring = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'

## inputs for walk-forward testing
fold_size = int(6*360) #Walk-forward test fold size
train_pct = 0.75 # For walk forward testing

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'

dm = Data_manager(coin, s_date, where='yahoo')

output_folder = 'C:/Users/Gianluca/Desktop/strategy_results/Approach_1/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

accuracy_summary_ls = []
precision_summary_ls = []
recall_summary_ls = []
f1_summary_ls = []
val1_summary_ls = []
val2_summary_ls = []
seedModel_summary_ls = []

cagr_summary_ls = []
strat_cum_ret_summary_ls = []
sharpe_summary_ls = []
sortino_summary_ls = []
kelly_summary_ls = []
rruin_summary_ls = []
max_drawdown_summary_ls = []
longest_dd_time_summary_ls = []

model_ = 'Random_forest'
model_dict = {model_: [RandomForestClassifier(), 'Random forest']}

########## Download data ###############
dm.download_price_data(output_folder + coin + '_raw_data.xlsx')

for val1 in val1_list:
    for val2 in val2_list:

        feat_vals = [val1, val2]
        ## Constraints:
        if constraint(val1, val2):
            continue

        # set the master_df = raw_df
        dm.reset_data()
        ########## Create features ###############
        dm.features_engineering_SMA([feat_vals[0], feat_vals[1]])
        dm.add_returns_to_data()
        dm.dropna_in_data()
        data = dm.get_data()
        feature_cols = dm.feature_names
        print('Feature cols:', feature_cols)

        ## Split data into training and test set
        data = dm.data
        ret_1 = dm.ret_1d
        X = data[feature_cols]
        Y = np.sign(data[ret_1])
        Y = Y.replace(-1, 0) # to have labels otherwise sklearn thinks you are doing a multi-label classification

        seed = np.random.randint(0, 1000)
        param_grid = {
            'n_estimators': [200],#, 400], # 800],
            'max_depth': [2],#, 3, 4],
            'criterion': ['gini'], #, 'entropy'],
            'random_state': [seed]
        }

        ## 1. Standard GRidSearch
        grid = GridSearchCV(estimator=model_dict[model_][0], param_grid=param_grid, scoring=scoring, cv=num_folds)

        fwts = For_test_split.Walk_forward_nonanchored_ts_split(fold_size, train_pct, X)

        print('Score used:', scoring)

        result_df = fwts.run_analysis(grid, Y, data, ret_1)

        realised_Y = Y.loc[result_df.index]
        predictions = result_df.loc[:, 'prediction']
        overall_accuracy = accuracy_score(realised_Y, predictions)
        overall_recall = recall_score(realised_Y, predictions)
        overall_precision = precision_score(realised_Y, predictions)
        overall_f1 = f1_score(realised_Y, predictions)

        result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
        fig = Utils.plot_oos_results(result_data, 'Out of sample results')

        fig_name = ''
        for i in range(len(feature_cols)):
            fig_name = fig_name + feature_cols[i] + '=' + str(feat_vals[i]) + '_'

        fig.savefig(output_folder + fig_name + '.png')
        plt.close()

        val1_summary_ls.append(val1)
        val2_summary_ls.append(val2)
        accuracy_summary_ls.append(overall_accuracy)
        precision_summary_ls.append(overall_precision)
        recall_summary_ls.append(overall_recall)
        f1_summary_ls.append(overall_f1)

        seedModel_summary_ls.append(seed)

        strategy_rets = result_data['strategy']
        cagr_summary_ls.append(quantstats.stats.cagr(strategy_rets))
        strat_cum_ret_summary_ls.append(result_data['cum_strategy'].iloc[-1] / result_data['cum_strategy'].iloc[0] - 1)

        sharpe_summary_ls.append(quantstats.stats.sharpe(strategy_rets))
        sortino_summary_ls.append(quantstats.stats.sortino(strategy_rets))
        kelly_summary_ls.append(quantstats.stats.kelly_criterion(strategy_rets))
        rruin_summary_ls.append(quantstats.stats.risk_of_ruin(strategy_rets))

        drawdown_series = quantstats.stats.to_drawdown_series(strategy_rets)
        drawdown_df = quantstats.stats.drawdown_details(drawdown_series)
        max_drawdown_summary_ls.append(drawdown_df['max drawdown'].min())
        longest_dd_time_summary_ls.append(drawdown_df['days'].max())


summary_dict = {'val1': val1_summary_ls, 'val2': val2_summary_ls, 'accuracy': accuracy_summary_ls, 'f1': f1_summary_ls,
                'recall': recall_summary_ls, 'precision': precision_summary_ls, 'seed': seedModel_summary_ls, 'cagr': cagr_summary_ls,
                'strategy_cum_ret': strat_cum_ret_summary_ls, 'sharpe': sharpe_summary_ls, 'sortino': sortino_summary_ls,
                'kelly_criterion': kelly_summary_ls, 'risk_of_ruin': rruin_summary_ls, 'max_drawdown': max_drawdown_summary_ls,
                'longest_dd_time': longest_dd_time_summary_ls}

static_summary_dict = {'binance_commissions': binance_commissions, 'slippage': slippage, 'strategy_type': st,
                       'model': model_dict[model_][1], 'fold_size': fold_size, 'train_pct': train_pct,
                       'num_folds_cv': num_folds, 'optimisation_metric_cv': scoring}

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

plt.plot()

print('The end?')




