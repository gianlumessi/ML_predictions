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
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous, Categorical
pd.set_option('display.max_columns', 500)


def plot_tree_(model, cols):
    _ = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)


########## inputs ###############
coin = 'ETH-USD'
s_date = '2015-01-01'
date_split = '2021-07-01'
search_term = 'Ethereum' #values allowed so far: 'bitcoin', 'Ethereum', 'xrp'

# test options for classification
num_folds = 8
seed = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

lags_p_smas = [7, 14, 28, 60]
lags_smas = [7, 14, 28, 60]
lags_rsi = [7, 14, 28, 60]
lags_std = [7, 14, 28, 60]

#####
###########
#################
#######################

dm = Data_manager(coin, s_date, search_term=search_term, path='local_file') #path='local_file' looks for files in local folder
dm.download_price_data()
dm.merge_search_with_price_data()
dm.features_engineering_for_dec_tree(lags_p_smas, lags_smas, lags_rsi, lags_std)
feature_cols = dm.feature_cols

print('Feature cols:', feature_cols)

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
n_gens = 30
consec_stop = 5
pop_size = 800
tourn_size = 5

print('\n <--- Genetic search of features on Decision tree --->')
model = DecisionTreeClassifier(max_depth=4, criterion='gini', min_samples_leaf=10, splitter='best')
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
print('\n- Accuracy score on test set (Decision tree):\t', accuracy_score(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'Decision tree with best features found via ga')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results using best features found via ga, Dec tree')
plot_tree_(model, best_features)


################################################# 2 #############################################################
##############################################################################################################
print('\n <--- Hyper params optimisation via genetic algo on Decision tree --->')

param_grid = {'criterion': Categorical(['gini', 'entropy']),
              'max_depth': Integer(2, 4),
              'min_samples_leaf': Integer(2, 100),
              'max_features': Integer(2, len(best_features))}

# The base classifier to tune
clf = DecisionTreeClassifier()

# The main class from sklearn-genetic-opt
evolved_clf = GASearchCV(estimator=clf,
                        cv=num_folds,
                        scoring=scoring,
                        param_grid=param_grid,
                        population_size=pop_size,
                        generations=n_gens,
                        tournament_size=tourn_size,
                        n_jobs=-1,
                        keep_top_k=1,
                        verbose=True)

callback = ConsecutiveStopping(generations=consec_stop, metric='fitness')
evolved_clf.fit(X_train_best_features, Y_train, callbacks=callback)

# Best parameters found
print(evolved_clf.best_params_)
# Use the model fitted with the best parameters
predictions = evolved_clf.predict(X_test_best_features)
print('- Accuracy score on test set Hyper params optimisation via genetic algo on Decision tree:\t', accuracy_score(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'Hyper params optimisation via genetic algo on Decision tree')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results, Hyper params optimisation via genetic algo on Decision tree')

if False:
    ######################################################################
    ##########   Grid Search: CART (decision tree)   #####################
    #####################################################################
    print('Grid search on Decision tree')

    ## Results on test data set
    # prepare model

    max_depth = [2, 3, 4, 5]
    min_samples_leaf = [11, 33, 66, 100]
    criterion = ["gini", "entropy"]
    splitter = ['best', 'random']
    grid = dict(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, splitter=splitter)
    grid_result_cart = Utils.grid_search_(X_train_best_features, Y_train, grid, scoring, kfold, DecisionTreeClassifier(), 'CART')

    model_cart = DecisionTreeClassifier(criterion=grid_result_cart.best_params_['criterion'],
                                   max_depth=grid_result_cart.best_params_['max_depth'],
                                   min_samples_leaf=grid_result_cart.best_params_['min_samples_leaf'],
                                    splitter = grid_result_cart.best_params_['splitter'])

    model_cart.fit(X_train_best_features, Y_train)

    ########## Check results on test data ##############

    predictions = model_cart.predict(X_test_best_features)
    print('\n- Accuracy score on test set (CART after grid search):\t', accuracy_score(Y_test, predictions), '\n')
    Utils.show_confusion_matrix(Y_test, predictions, 'CART after grid search')
    result_data = dm.get_result_data(test_data, predictions)
    Utils.plot_oos_results(result_data, 'Out of sample results, CART after grid search')
    plot_tree_(model_cart, best_features)


plt.show()