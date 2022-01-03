import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from pylab import mpl, plt
from Data_handler import Data_manager
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
pd.set_option('display.max_columns', 500)


def grid_search_(X, Y, param_grid, scoring, kfold, model_, ttl):
    model = model_
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X, Y)

    #Print Results
    print(ttl + ". Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    ranks = grid_result.cv_results_['rank_test_score']
    for mean, stdev, param, rank in zip(means, stds, params, ranks):
        print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))

    return grid_result


def plot_oos_results(df, ttl):
    fig, axes = plt.subplots(nrows=2, figsize=(10, 6), gridspec_kw={'height_ratios': [5, 1]})
    df[['cum_return', 'cum_strategy']].plot(ax=axes[0])
    df['prediction'].plot(ax=axes[1])

    fig.suptitle(ttl)
    plt.tight_layout()


def show_confusion_matrix(Y, x, ttl):
    plt.figure()
    df_cm = pd.DataFrame(confusion_matrix(Y, x), columns=np.unique(Y), index = np.unique(Y))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font sizes
    plt.title(ttl)



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

model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=10, splitter='best')
evolved_tree = GAFeatureSelectionCV(model,
                                    cv=6,
                                    generations=20,
                                    population_size=500,
                                    scoring='accuracy',
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
show_confusion_matrix(Y_test, predictions, 'CART')
result_data = dm.get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, CART')
plot_tree_(model, best_features)

######################################################################
##########   Grid Search: CART (decision tree)   #####################
#####################################################################
print('Grid search on Decision tree')

## Results on test data set
# prepare model

max_depth = [2, 3, 4]
min_samples_leaf = [11, 33, 66, 100]
criterion = ["gini", "entropy"]
splitter = ['best', 'random']
grid = dict(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, splitter=splitter)
grid_result_cart = grid_search_(X_train[best_features], Y_train, grid, scoring, kfold, DecisionTreeClassifier(), 'CART')

model_cart = DecisionTreeClassifier(criterion=grid_result_cart.best_params_['criterion'],
                               max_depth=grid_result_cart.best_params_['max_depth'],
                               min_samples_leaf=grid_result_cart.best_params_['min_samples_leaf'],
                                splitter = grid_result_cart.best_params_['splitter'])

model_cart.fit(X_train[best_features], Y_train)

########## Check results on test data ##############

predictions = model_cart.predict(X_test[best_features])
print('- Accuracy score on test set (CART after grid search):\t', accuracy_score(Y_test, predictions), '\n')
show_confusion_matrix(Y_test, predictions, 'CART after grid search')
result_data = dm.get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, CART after grid search')
plot_tree_(model_cart, best_features)


plt.show()