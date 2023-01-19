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
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 500)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import Train_test_split as For_test_split



########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2022-06-01'

## inputs for features creation
t1_ = 30
t2_ = 90

# test options for classification
num_folds = 10
seed = np.random.randint(0, 100)

scoring = 'accuracy' #'accuracy'
if scoring == 'average_precision':
  from sklearn.metrics import average_precision_score as score_meth
elif scoring == 'accuracy':
  from sklearn.metrics import accuracy_score as score_meth

dm = Data_manager(coin, s_date, where='yahoo')
dm.download_price_data()

print(dm.data.head())

dm.features_engineering_for_SMA(t1_, t2_)
feature_cols = dm.feature_names

print('Feature cols:', feature_cols)

## Split data into training and test set
data = dm.data
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

ret_1 = dm.ret_1d
X_train = training_data[feature_cols]
Y_train = np.asarray(np.sign(training_data[ret_1]))

X_test = test_data[feature_cols]
Y_test = np.asarray(np.sign(test_data[ret_1]))


param_grid = {
    'n_estimators': [200], # , 300], #, 400, 500],
    'max_depth': [2], #, 3], #, 4],
    'criterion': ['gini'],#, 'entropy'],
    'random_state': [seed]
    }

## 1. Standard GRidSearch
model = RandomForestClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=num_folds)
grid_result = grid.fit(X_train, Y_train)


best_parameters = grid_result.best_params_
print('Best params:', best_parameters)

model_opt = RandomForestClassifier(criterion=grid_result.best_params_['criterion'],
                               max_depth=grid_result.best_params_['max_depth'],
                               n_estimators=grid_result.best_params_['n_estimators'],
                               random_state = grid_result.best_params_['random_state'])


model_opt.fit(X_train, Y_train)
predictions = model_opt.predict(X_test)

print('- Accuracy score on test set:\t', score_meth(Y_test, predictions), '\n')
Utils.show_confusion_matrix(Y_test, predictions, 'RF after GA feature selection')
print('- Classification report:\n', classification_report(Y_test, predictions))

result_data = dm.run_simple_backtest(test_data, predictions)
Utils.plot_oos_results(result_data, 'Out of sample results')


## Walkforward testing
fwts = For_test_split.Walk_forward_nonanchored_nonoverlapping_ts_split(n_splits=num_folds, train_pct=0.75)

model = RandomForestClassifier()

## 2. GridSearch forward testing optimisation
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=fwts)
grid_result = grid.fit(X_train, Y_train)

best_parameters = grid_result.best_params_
best_score = grid_result.best_score_

print('Best params', best_parameters, ' - Best score:', best_score)


## 3. GridSearch forward testing optimisation retaining the estimator per each fold
scores = cross_validate(grid, X_train, Y_train, cv=fwts, return_estimator=True)
print('Test score:', scores['test_score'])
print('Estimator:', scores['estimator'])

predictions = cross_val_predict(grid, X_train, Y_train, cv=5)
print('Predictions', predictions)
pd.Series(predictions).to_excel('C:/Users/Gianluca/Desktop/predictions.xlsx')

plt.show()