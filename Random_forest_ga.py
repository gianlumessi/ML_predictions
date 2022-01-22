import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from pylab import mpl, plt
from Data_manager import Data_manager
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous
pd.set_option('display.max_columns', 500)


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


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'

# test options for classification
num_folds = 6
seed = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# features
lags_price = [1, 5, 10, 20, 60, 7, 14, 28, 84]
lags_price_daily_rets = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(1, 60, num=60)#
lags_rets = [1, 5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(2, 60, num=59)
lags_sma = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)
lags_std = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)
lags_rsi = [5, 10, 20, 60, 7, 14, 28, 84] #np.linspace(5, 60, num=56)

#######
###########
###############

dm = Data_manager(coin, s_date)
dm.download_price_data()
dm.features_engineering(lags_p_drets=lags_price_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std,
                             lags_rsi=lags_rsi, lags_price=lags_price)
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

print('Feature search using GAs')
max_depth = 3
pop_size = 500
print('population_size = ', len(feature_cols))
print('n_features', len(feature_cols))
model = RandomForestClassifier(max_depth=max_depth, n_estimators=150, min_samples_leaf=10, max_features=max_depth+1)
evolved_rf = GAFeatureSelectionCV(model,
                                  cv=num_folds,
                                  generations=30,
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

# Predict only with the subset of selected features
predictions = evolved_rf.predict(X_test[best_features])
print('- Accuracy score on test set (RF after GA feature selection):\t', accuracy_score(Y_test, predictions), '\n')
show_confusion_matrix(Y_test, predictions, 'RF after GA feature selection')
result_data = dm.get_result_data(test_data, predictions)
plot_oos_results(result_data, 'Out of sample results, RF after GA feature selection')

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
                                  generations=30,
                                  n_jobs=-1,
                                  keep_top_k=1,
                                  verbose=True)


    callback = ConsecutiveStopping(generations=5, metric='fitness')
    evolved_rf.fit(X_train[best_features], Y_train, callbacks=callback)

    # Best parameters found
    print('\nBest parameters found by Genetic algo:')
    print(evolved_rf.best_params_)
    # Use the model fitted with the best parameters
    predictions = evolved_rf.predict(X_test[best_features])
    print('- Accuracy score on test set RF:\t', accuracy_score(Y_test, predictions), '\n')
    show_confusion_matrix(Y_test, predictions, 'RF')
    result_data = dm.get_result_data(test_data, predictions)
    plot_oos_results(result_data, 'Out of sample results, RF')


plt.show()