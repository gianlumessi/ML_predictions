import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous, Categorical
pd.set_option('display.max_columns', 500)


def plot_tree_(model, cols):
    fig = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'
search_term = 'bitcoin'

# test options for classification
num_folds = 6
seed = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# features
arr1 = [1, 5, 10, 20, 60, 7, 14, 28, 84]
arr2 = [5, 10, 20, 60, 7, 14, 28, 84]
lags_price = arr1.copy()
lags_price_daily_rets = arr2.copy()
lags_rets = arr1.copy()
lags_sma = arr2.copy()
lags_std = arr2.copy()
lags_rsi = arr2.copy()
lags_search = arr2.copy()
lags_search_sma = arr2.copy()

#####
###########
#################
#######################

dm = Data_manager(coin, s_date, search_term=search_term, path='local_file') #path='local_file' looks for files in local folder
dm.download_price_data()
dm.merge_search_with_price_data()
dm.features_engineering(lags_p_drets=lags_price_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std,
                             lags_rsi=lags_rsi, lags_search=lags_search, lags_search_sma=lags_search_sma, lags_price=lags_price)
dm.combine_features()

feature_cols = dm.feature_cols

print('Feature cols:', feature_cols)

## Split data into training and test set
data = dm.df
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

X_train = training_data[feature_cols]
Y_train = np.sign(training_data['return'])

X_test = test_data[feature_cols]
Y_test = np.sign(test_data['return'])

print('Genetic search of features on Gradient Booster')
model = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1)
evolved_gb_ = GAFeatureSelectionCV(model,
                                   cv=num_folds,
                                   generations=1,
                                   population_size=500,
                                   scoring=scoring,
                                   tournament_size=5,
                                   keep_top_k=1,  #n of best solutions to keep
                                   verbose=True,
                                   n_jobs=-1)

callback = ConsecutiveStopping(generations=5, metric='fitness')
evolved_gb_.fit(X_train, Y_train, callbacks=callback)

plot_fitness_evolution(evolved_gb_)

best_features = list(X_train.columns[evolved_gb_.best_features_].values)
print('Best features:', best_features)
print('Best estimator:', evolved_gb_.best_estimator_)


print('Genetic search on Gradient booster')

param_grid = {'learning_rate': Continuous(0.01, 0.1, distribution='uniform'),
              'sub_sample': Continuous(0.1, 1, distribution='uniform'),
              'max_depth': Integer(2, 4),
              'min_samples_leaf': Integer(2, 50),
              'n_estimators': Integer(100, 200),
              'max_features': Integer(2, len(best_features))}

# The base classifier to tune
clf = GradientBoostingClassifier()

# The main class from sklearn-genetic-opt
evolved_gb = GASearchCV(estimator=clf,
                        cv=num_folds,
                        scoring=scoring,
                        param_grid=param_grid,
                        population_size=800,
                        generations=1,
                        n_jobs=-1,
                        keep_top_k=1,
                        verbose=True)

callback = ConsecutiveStopping(generations=5, metric='fitness')
evolved_gb.fit(X_train[best_features], Y_train, callbacks=callback)

# Best parameters found
print('\nBest parameters found by Genetic algo:')
print(evolved_gb.best_params_)


########## Check results on test data ##############

predictions = evolved_gb.predict(X_test[best_features])
print('\n- Accuracy score on test set GB:\t', accuracy_score(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'GB after grid search')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results, GB after grid search')

plt.show()