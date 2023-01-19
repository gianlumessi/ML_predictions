'''
Info: the objective of this script is to look at the most recent n-month period and find a similar period in the past.
- Train a ML model on the most recent n-month period and run a features importances analysis
- Use the best features to make predictions on past n-month historical periods
- See which historical periods the model makes good predictions
'''


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import quantstats
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from pylab import mpl, plt
from Data_manager import Data_manager
from sklearn.base import clone
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('display.max_columns', 500)
from datetime import datetime
import Train_test_split as For_test_split


# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")#("%d-%m-%Y_%H-%M-%S")

## Other inputs
binance_commissions = 0.00075
slippage = 0.001
tc = binance_commissions + slippage
st = 'long_only'
where_ = 'yahoo'
min_ret = tc

########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'

## inputs for cross-=validation
num_folds = 4
scoring = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'


## inputs for features creation
vals_rsi = [30, 180]#[7, 30, 90, 180]
vals_sma = [30, 180]#[14, 30, 90, 120, 180]
vals_pctChange = [30, 180]#[7, 30, 90, 180]
vals_std = [30, 180]#[30, 90, 120, 180]
vals_so = [30, 180]#[30, 90, 120, 180]

fold_size_y = 5
fold_size = int(fold_size_y*365) #Walk-forward test fold size
train_pct = 0.75 #For walk forward testing


########### Model / Estimator ###############
model_ = 'Random_forest_classifier'
model_dict = {model_: [RandomForestClassifier(), 'Random forest']}
seed = np.random.randint(0, 1000)
n_best_features = 4

dm = Data_manager(coin, s_date, where=where_)

dm.download_price_data()

dm.features_engineering_RSI(vals_rsi) #([val1, val2])
dm.features_engineering_SMA(vals_sma) #([val1, val2])
dm.features_engineering_PCTCHANGE(vals_pctChange) #([val1, val2])
dm.features_engineering_ONBALVOL()
dm.features_engineering_STD(vals_std)
dm.features_engineering_STOCHOSC(vals_so)

dm.add_returns_to_data()
dm.dropna_in_data()
data = dm.get_data()
feature_names = dm.feature_names
ret_1 = dm.ret_1d

param_grid = {
    'n_estimators': [400],#, 800],#, 400, 800],
    'max_depth': [2],#, 3],#, 4],
    'criterion': ['gini'],  # , 'entropy'],
    'random_state': [seed],
    #'max_features': [len(feature_names)], #use all features since they are only 2
    'min_samples_leaf': [0.1],
    'max_samples': [0.8]
}

print('Feature names:')
print(dm.feature_names, '\n')

print(data.head())

frames = []
outer_i = 0

