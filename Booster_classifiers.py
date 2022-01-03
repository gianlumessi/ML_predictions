import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from Data_handler import Data_manager
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score

##################################################################

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


def get_result_data(df, preds):
    df_ = df.copy()

    df_['prediction'] = preds
    df_['strategy'] = df_['prediction'] * df_['return']
    df_[['cum_return', 'cum_strategy']] = df_[['return', 'strategy']].cumsum().apply(np.exp)

    return df_


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

####################################################################################
####################################################################################

########## inputs ###############
coin = 'BTC-USD'
s_date = '2014-01-01'
date_split = '2021-06-01'

# test options for classification
num_folds = 6
seed = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# features
lags_price_daily_rets = [1, 7] #[1, 7, 28]
lags_rets = [5, 60]
lags_sma = [5] # [7, 28, 90]
lags_std = None #[5, 7]
lags_rsi = [28, 60] #[20, 28, 60] #



#######
#########
###########
#############

lags_price = [1, 5, 10, 20, 60, 7, 14, 28, 84]
lags_price_daily_rets = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(1, 60, num=60)#
lags_rets = [1, 5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(2, 60, num=59)
lags_sma = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)
lags_std = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)
lags_rsi = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)

features_subset = ['feat_price_5_feat_sma_10', 'feat_price_20_feat_sma_20',
                   'feat_price_7_feat_sma_14', 'feat_price_14_feat_sma_7', 'feat_return_1_feat_return_60',
                   'feat_return_1_feat_return_28', 'feat_return_5_feat_return_28', 'feat_return_20_feat_return_14',
                   'feat_return_60_feat_return_28', 'feat_rsi_20']
#######
###########
###############

dm = Data_manager(coin, s_date)
dm.download_price_data()
dm.features_engineering(lags_p_drets=lags_price_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std,
                             lags_rsi=lags_rsi, lags_price=lags_price)
dm.combine_features()

ls = features_subset.copy()
ls.append('return')
data = dm.df[ls]
print(data.head())

## Split data into training and test set
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

X_train = training_data[features_subset]
Y_train = np.sign(training_data['return']) #np.where(training_data['return'] >= 0, 1.0, 0.0)

X_test = test_data[features_subset]
Y_test = np.sign(test_data['return']) # np.where(test_data['return'] >= 0, 1.0, 0.0) it's ok if it is not binary. The algo
#automatically switches to multinomial.


######################################################################
##########   Grid Search: CART (decision tree)   #####################
#####################################################################

## Results on test data set
# prepare model

max_depth = [2, 3, 4]
min_samples_leaf = [11, 33, 66, 100]
criterion = ["gini", "entropy"]
splitter = ['best', 'random']
grid = dict(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, splitter=splitter)
grid_result_cart = grid_search_(X_train, Y_train, grid, scoring, kfold, DecisionTreeClassifier(), 'CART')

model_cart = DecisionTreeClassifier(criterion=grid_result_cart.best_params_['criterion'],
                               max_depth=grid_result_cart.best_params_['max_depth'],
                               min_samples_leaf=grid_result_cart.best_params_['min_samples_leaf'],
                                splitter = grid_result_cart.best_params_['splitter'])

model_cart.fit(X_train, Y_train)

########## Check results on test data ##############

predictions = model_cart.predict(X_test)
print('- Accuracy score on test set (CART):\t', accuracy_score(Y_test, predictions), '\n')
show_confusion_matrix(Y_test, predictions, 'CART')
result_data = get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, CART')
plot_tree_(model_cart, features_subset)


######################################################################
###############   Grid Search: Ada Boost   ###################
#####################################################################

ada_clf = AdaBoostClassifier(base_estimator=model_cart, algorithm="SAMME.R")

n_estimators = [100, 250, 500]
learning_rate = [0.01]


grid = dict(n_estimators = n_estimators, learning_rate=learning_rate)
grid_result_ada = grid_search_(X_train, Y_train, grid, scoring, kfold, ada_clf, 'ADA Boost')

ada_clf = AdaBoostClassifier(base_estimator=model_cart, algorithm="SAMME.R",
                             n_estimators=grid_result_ada.best_params_['n_estimators'],
                            learning_rate=grid_result_ada.best_params_['learning_rate'])

ada_clf.fit(X_train, Y_train)

########## Check results on test data ##############

predictions = ada_clf.predict(X_test)
print('- Accuracy score on test set (Ada Boost):\t', accuracy_score(Y_test, predictions), '\n')
show_confusion_matrix(Y_test, predictions, 'ADA')
result_data = get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, ADA Boost optimised')


######################################################################
###############   Grid search: Gradient Boosting   ###################
#####################################################################

#GB Boost. Best: 0.557542 using {'learning_rate': 0.01, 'max_depth': 4, 'min_samples_leaf': 10, 'n_estimators': 100, 'subsample': 0.33}
gb_clf = GradientBoostingClassifier()

n_estimators = [200, 500] #[100, 150, 200]
learning_rate = [0.01]
max_depth = [3, 4]
min_samples_leaf = [10, 30]
subsample = [0.5]

grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, subsample=subsample)
grid_result_gb = grid_search_(X_train, Y_train, grid, scoring, kfold, gb_clf, 'GB Boost')

## Results on test data set
gb_clf = GradientBoostingClassifier(n_estimators=grid_result_gb.best_params_['n_estimators'],
                                  learning_rate=grid_result_gb.best_params_['learning_rate'],
                                  max_depth=grid_result_gb.best_params_['max_depth'],
                                  min_samples_leaf=grid_result_gb.best_params_['min_samples_leaf'],
                                  subsample=grid_result_gb.best_params_['subsample'])

gb_clf.fit(X_train, Y_train)

########## Check results on test data ##############

predictions = gb_clf.predict(X_test)
print('- Accuracy score on test set (Gradient boosting):\t', accuracy_score(Y_test, predictions), '\n')
show_confusion_matrix(Y_test, predictions, 'GB')
result_data = get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, Gradient Boosting')


plt.show()
