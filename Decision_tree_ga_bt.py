import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous, Categorical
from imblearn.over_sampling import RandomOverSampler
pd.set_option('display.max_columns', 500)

from Backtester import BacktestLongShort

def plot_tree_(model, cols):
    _ = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)


########## inputs ###############
pair_dict = {'BTC': ['BTC-USD', 'bitcoin'], 'ETH': ['ETH-USD', 'Ethereum'], 'XRP': ['XRP-USD', 'xrp']}
asset = 'BTC'
price_col_name = None

coin = pair_dict[asset][0]
search_term = pair_dict[asset][1] #values allowed so far: 'bitcoin', 'Ethereum', 'xrp'
s_date = '2015-01-01'
date_split = '2021-07-01'

oversample = True
# test options for classification
num_folds = 8
seed = 7
scoring = 'accuracy' #'average_precision' #

if scoring == 'average_precision':
  from sklearn.metrics import average_precision_score as score_meth
elif scoring == 'accuracy':
  from sklearn.metrics import accuracy_score as score_meth

np.random.seed(seed)

fee = 0.001
short_funding = 0.0002

lags_p_smas = [2, 3, 4, 5, 6, 7, 14, 28, 60, 90]
lags_smas = [7, 14, 28, 60, 90]
lags_rsi = [7, 14, 28, 60, 90]
#lags_std = [7, 14, 28, 60, 90]

#####
###########
#################
#######################

dm = Data_manager(coin, s_date, search_term=search_term, path='local_file') #path='local_file' looks for files in local folder
dm.download_price_data()
dm.merge_search_with_price_data()
dm.features_engineering_for_dec_tree(lags_p_smas, lags_smas, lags_rsi)
feature_cols = dm.feature_cols

print('Feature cols:', feature_cols)

## Split data into training and test set
data = dm.df
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

ret_1 = dm.ret_1d
X_train = training_data[feature_cols]
Y_train = np.sign(training_data[ret_1])

if oversample:
    print('Resampling training population...')
    Y_train_ = np.where(Y_train == 0, 1, Y_train)
    if not (Y_train == Y_train_).all():
        Y_train = Y_train_
        print('Zeros found in Y_train set to 1 for not screwing up the resampling ...')
    ros = RandomOverSampler(random_state=seed)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

X_test = test_data[feature_cols]
Y_test = np.sign(test_data[ret_1])

################################################## 1 ############################################################
##############################################################################################################
n_gens = 0
consec_stop = 3
pop_size = 1
tourn_size = 3

print('\n <--- Genetic search of features on Decision tree --->')
model = DecisionTreeClassifier(max_depth=4, criterion='entropy', min_samples_leaf=10, splitter='best')
evolved_tree = GAFeatureSelectionCV(model,
                                    cv=num_folds,
                                    generations=n_gens,
                                    population_size=pop_size,
                                    scoring=scoring,
                                    tournament_size=tourn_size,
                                    keep_top_k=1,  #n of best solutions to keep
                                    verbose=True,
                                    n_jobs=-1)

callback = ConsecutiveStopping(generations=consec_stop, metric='fitness')
evolved_tree.fit(X_train, Y_train, callbacks=callback)

plot_fitness_evolution(evolved_tree)

best_features = list(X_train.columns[evolved_tree.best_features_].values)
print('Best features:', best_features)
print('Best estimator:', evolved_tree.best_estimator_)

text_representation = tree.export_text(model, feature_names=best_features) #list(best_features.values)
print(text_representation)

X_test_best_features = X_test.loc[:, best_features]
X_train_best_features = X_train.loc[:, best_features]

print('\nShape of X_test_best_features', X_test_best_features.shape)
print('Shape of X_train_best_features', X_train_best_features.shape)

# Predict only with the subset of selected features
predictions = evolved_tree.predict(X_test_best_features)
print('\n- Accuracy score on test set (Decision tree):\t', score_meth(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'Decision tree with best features found via ga')
#result_data = dm.backtest_strategy_with_fees(test_data, predictions, fee, short_funding)

debug = False
if debug:
    predictions=-1

bt_data = pd.DataFrame(index=test_data.index)
bt_data['prediction'] = predictions
price_data = dm.get_price_data().loc[date_split:]
if price_col_name is None:
    price_col_name = pair_dict[asset][0]
bt_data[price_col_name] = price_data


cash_balance = 1000
min_order_size = 0.0001
col_name_dict = {'price_col': price_col_name, 'buy_sell_signal_col': 'prediction', 'net_wealth_col': 'net_wealth',
                 'pnl_col': 'pnl'}
bt_ls = BacktestLongShort(cash_balance, bt_data, col_name_dict, min_order_size, ftc=0.0, ptc=0.00, sfc=0.00,
                          rebalance_threshold=0, verbose=False)
bt_ls.bt_long_short_signal()
bt_ls.plot_strategy_vs_asset()


result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, '- Out of sample results, Hyper params optimisation via genetic algo ' + coin)

plt.show()

#Utils.plot_oos_results(result_data, 'Out of sample results using best features found via ga, Dec tree')
#plot_tree_(model, best_features)


