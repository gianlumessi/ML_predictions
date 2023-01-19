import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import quantstats
import seaborn as sns
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 500)
import Train_test_split as For_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'


dm = Data_manager(coin, s_date, search_term=search_term, path='local_file') #path='local_file' looks for files in local folder
dm.download_price_data()
#dm.merge_search_with_price_data()
dm.features_engineering_RSI(l4)

