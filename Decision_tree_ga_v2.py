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

dm = Data_manager(coin, s_date, search_term=search_term)
dm.download_price_data()
dm.merge_search_with_price_data()
dm.features_engineering(lags_p_drets=lags_price_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std,
                             lags_rsi=lags_rsi, lags_search=lags_search, lags_search_sma=lags_search_sma, lags_price=lags_price)
dm.combine_features()

feature_cols = dm.feature_cols

print('Feature cols:', feature_cols)
#print(data.tail())

## Split data into training and test set
data = dm.df
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

X_train = training_data[feature_cols]
Y_train = np.sign(training_data['return'])

X_test = test_data[feature_cols]
Y_test = np.sign(test_data['return'])

############################################################
print('Genetic search of features on Decision tree')
model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=10, splitter='best')
evolved_tree = GAFeatureSelectionCV(model,
                                    cv=6,
                                    generations=1,
                                    population_size=500,
                                    scoring=scoring,
                                    tournament_size=5,
                                    keep_top_k=1,  #n of best solutions to keep
                                    verbose=True,
                                    n_jobs=-1)

callback = ConsecutiveStopping(generations=5, metric='fitness')
evolved_tree.fit(X_train, Y_train, callbacks=callback)

plot_fitness_evolution(evolved_tree)

best_features = list(X_train.columns[evolved_tree.best_features_].values)
print('Best features:', best_features)
print('Best estimator:', evolved_tree.best_estimator_)

text_representation = tree.export_text(model, feature_names=best_features) #list(best_features.values)
print(text_representation)


# Predict only with the subset of selected features
predictions = evolved_tree.predict(X_test[best_features])
print('- Accuracy score on test set (CART):\t', accuracy_score(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'CART')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results, CART')
plot_tree_(model, best_features)


#######################################################
print('Hyper params optimisation via genetic algo on Decision tree')

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
                        population_size=800,
                        generations=1,
                        n_jobs=-1,
                        keep_top_k=1,
                        verbose=True)

callback = ConsecutiveStopping(generations=5, metric='fitness')
evolved_clf.fit(X_train[best_features], Y_train, callbacks=callback)

# Best parameters found
print(evolved_clf.best_params_)
# Use the model fitted with the best parameters
predictions = evolved_clf.predict(X_test[best_features])
print('- Accuracy score on test set RHyper params optimisation via genetic algo on Decision tree:\t', accuracy_score(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'Hyper params optimisation via genetic algo on Decision tree')
result_data = dm.get_result_data(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results, Hyper params optimisation via genetic algo on Decision tree')


#######################################################
#print('Genetic search on Decision tree')

#param_grid = {'criterion': Categorical(['gini', 'entropy']),
#              'max_depth': Integer(2, 4),
#              'min_samples_leaf': Integer(2, 100),
#              'max_features': Integer(2, len(best_features))}

# The base classifier to tune
#clf = DecisionTreeClassifier()

# The main class from sklearn-genetic-opt
#evolved_clf = GASearchCV(estimator=clf,
#                        cv=num_folds,
#                        scoring=scoring,
#                        param_grid=param_grid,
#                        population_size=800,
#                        generations=1,
#                        n_jobs=-1,
#                        keep_top_k=1,
#                        verbose=True)

#callback = ConsecutiveStopping(generations=5, metric='fitness')
#evolved_clf.fit(X_train[best_features], Y_train, callbacks=callback)

# Best parameters found
#print(evolved_clf.best_params_)
# Use the model fitted with the best parameters
#predictions = evolved_clf.predict(X_test[best_features])
#print('- Accuracy score on test set RF:\t', accuracy_score(Y_test, predictions), '\n')
#Utils.show_confusion_matrix(Y_test, predictions, 'RF')
#result_data = dm.get_result_data(test_data, predictions)
#Utils.plot_oos_results(result_data, 'Out of sample results, RF')

plt.show()