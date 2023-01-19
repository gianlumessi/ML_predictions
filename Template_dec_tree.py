import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
from pylab import mpl, plt
from Data_manager import Data_manager
from sklearn import tree
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_columns', 500)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os

def plot_tree_(model, cols):
    _ = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)

########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## inputs for features creation
val1 = 7#, 90, 120, 180, 360]
val2 = 30#, 90, 120, 180, 360, 720]

## inputs for cross-=validation
num_folds = 10
scoring = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'

dm = Data_manager(coin, s_date, where='yahoo')

output_folder = 'C:/Users/Gianluca/Desktop/strategy_results/Approach_1/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


########## Download data ###############
dm.download_price_data(output_folder + coin + '_raw_data.xlsx')

########## Create features ###############
dm.features_engineering_SMA([val1, val2])
dm.add_returns_to_data()
dm.dropna_in_data()
data = dm.get_data()
feature_cols = dm.feature_names
print('Feature cols:', feature_cols)

## Split data into training and test set
ret_1 = dm.ret_1d
X = data[feature_cols]
Y = np.sign(data[ret_1]-0.005)
Y = Y.replace(-1, 0) # to have labels otherwise sklearn thinks you are doing a multi-label classification

max_depth = [2, 3]
min_samples_leaf = [11, 33, 66, 100]
criterion = ['gini', 'entropy'] #"gini",
splitter = ['best', 'random']
param_grid = dict(max_depth=max_depth, splitter=splitter, criterion = criterion, min_samples_leaf = min_samples_leaf)
model = DecisionTreeClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=num_folds)
grid_result = grid.fit(X, Y)

model_cart = DecisionTreeClassifier(criterion=grid_result.best_params_['criterion'],
                               max_depth=grid_result.best_params_['max_depth'],
                               min_samples_leaf=grid_result.best_params_['min_samples_leaf'],
                               splitter = grid_result.best_params_['splitter'])

model_cart.fit(X, Y)

y_pred = model_cart.predict(X)
print(accuracy_score(Y, y_pred))

plot_tree_(model_cart, feature_cols)

text_representation = tree.export_text(model_cart, feature_names=feature_cols)
print(text_representation)

y_pred = pd.Series(y_pred, index=X.index, name='prediction')
returns = data.loc[Y.index, ret_1]
result_df = pd.concat([returns, y_pred], axis=1)
result_data = dm.run_simple_backtest(result_df, trading_costs=tc, strategy_type=st)
fig = Utils.plot_oos_results(result_data, 'Out of sample results')

print('The end?')




