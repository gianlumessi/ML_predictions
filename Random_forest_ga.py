import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous
pd.set_option('display.max_columns', 500)


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'
search_term = 'bitcoin'

# test options for classification
num_folds = 6
seed = 7

scoring = 'average_precision' #'accuracy'
if scoring == 'average_precision':
  from sklearn.metrics import average_precision_score as score_meth
elif scoring == 'accuracy':
  from sklearn.metrics import accuracy_score as score_meth


# features
lags_p_smas = [7, 14, 28, 60]
lags_smas = [7, 14, 28, 60]
lags_rsi = [7, 14, 28, 60]
lags_std = [7, 14, 28, 60]

#######
###########
###############

dm = Data_manager(coin, s_date, search_term=search_term, path='local_file')
dm.download_price_data()
dm.merge_search_with_price_data()
dm.features_engineering_for_dec_tree(lags_p_smas, lags_smas, lags_rsi, lags_std)
feature_cols = dm.feature_cols

print('Feature cols:', feature_cols)
#print(data.tail())

## Split data into training and test set
data = dm.df
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

ret_1 = dm.ret_1d
X_train = training_data[feature_cols]
Y_train = np.sign(training_data[ret_1])

X_test = test_data[feature_cols]
Y_test = np.sign(test_data[ret_1])

################################################## 1 ############################################################
##############################################################################################################
n_gens = 2

print('Feature search using GAs')
max_depth = 3
pop_size = 500
print('population_size = ', len(feature_cols))
print('n_features', len(feature_cols))
model = RandomForestClassifier(max_depth=max_depth, n_estimators=150, min_samples_leaf=10, max_features=max_depth+1)
evolved_rf = GAFeatureSelectionCV(model,
                                  cv=num_folds,
                                  generations=n_gens,
                                  population_size=pop_size,
                                  scoring=scoring,
                                  tournament_size=5,
                                  keep_top_k=1,  #n of best solutions to keep
                                  verbose=True,
                                  n_jobs=-1)

callback = ConsecutiveStopping(generations=5, metric='fitness')
evolved_rf.fit(X_train, Y_train, callbacks=callback)

plot_fitness_evolution(evolved_rf)

best_features = list(X_train.columns[evolved_rf.best_features_].values)
print('Best features:', best_features)
print('Best estimator:', evolved_rf.best_estimator_)

X_test_best_features = X_test.loc[:, best_features]
X_train_best_features = X_train.loc[:, best_features]

print('\nShape of X_test_best_features', X_test_best_features.shape)
print('Shape of X_train_best_features', X_train_best_features.shape)

# Predict only with the subset of selected features
predictions = evolved_rf.predict(X_test_best_features)
print('- Accuracy score on test set (RF after GA feature selection):\t', score_meth(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'RF after GA feature selection')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results, RF after GA feature selection')

if True:
    ######################################################################
    ##########   Genetic Search on random forest   #####################
    #####################################################################
    print('Genetic search on Random Forest')

    param_grid = {'min_weight_fraction_leaf': Continuous(0.01, 0.5, distribution='log-uniform'),
                  'max_depth': Integer(2, 4),
                  'min_samples_leaf': Integer(2, 50),
                  'n_estimators': Integer(100, 300),
                  'max_features': Integer(2, len(best_features))}

    # The base classifier to tune
    clf = RandomForestClassifier()

    # The main class from sklearn-genetic-opt
    evolved_rf = GASearchCV(estimator=clf,
                                  cv=num_folds,
                                  scoring=scoring,
                                  param_grid=param_grid,
                                  population_size=800,
                                  generations=n_gens,
                                  n_jobs=-1,
                                  keep_top_k=1,
                                  verbose=True)


    callback = ConsecutiveStopping(generations=5, metric='fitness')
    evolved_rf.fit(X_train_best_features, Y_train, callbacks=callback)

    # Best parameters found
    print('\nBest parameters found by Genetic algo:')
    print(evolved_rf.best_params_)
    # Use the model fitted with the best parameters
    predictions = evolved_rf.predict(X_test_best_features)
    print('- Accuracy score on test set RF:\t', score_meth(Y_test, predictions), '\n')
    Utils.show_confusion_matrix(Y_test, predictions, 'RF')
    result_data = dm.get_result_data(test_data, predictions)
    Utils.plot_oos_results(result_data, 'Out of sample results, RF')


plt.show()