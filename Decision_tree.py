import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
from pylab import mpl, plt
from Data_manager import Data_manager
from Utils import Utils
from sklearn import tree
import seaborn as sns
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def plot_tree_(model, cols):
    _ = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)

coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'
search_term = 'bitcoin'

# test options for classification
num_folds = 6
seed = 7
scoring = 'accuracy'

lags_p_smas = [7, 14, 28, 60]
lags_smas = [7, 14, 28, 60]
lags_rsi = [7, 14, 28, 60]
lags_std = [7, 14, 28, 60]

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


######################################################################
##########   Grid Search: CART (decision tree)   #####################
#####################################################################

#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
max_depth= [2, 3, 4]
min_samples_leaf = [11, 33, 66, 100]
criterion = ['gini', 'entropy'] #"gini",
splitter = ['best', 'random']
param_grid = dict(max_depth=max_depth, splitter=splitter, criterion = criterion, min_samples_leaf = min_samples_leaf)
model = DecisionTreeClassifier()
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)

#Print Results
print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
ranks = grid_result.cv_results_['rank_test_score']
for mean, stdev, param, rank in zip(means, stds, params, ranks):
    print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))

## Results on test data set
# prepare model
model_cart = DecisionTreeClassifier(criterion=grid_result.best_params_['criterion'],
                               max_depth=grid_result.best_params_['max_depth'],
                               min_samples_leaf=grid_result.best_params_['min_samples_leaf'],
                               splitter = grid_result.best_params_['splitter'])

#model = LogisticRegression()
model_cart.fit(X_train, Y_train)

# estimate accuracy on validation set
predictions = model_cart.predict(X_test)
print('- Accuracy score:\t', accuracy_score(Y_test, predictions))
print('- Confusion matrix:\n', confusion_matrix(Y_test, predictions))
print('- Classification report:\n', classification_report(Y_test, predictions))

df_cm = pd.DataFrame(confusion_matrix(Y_test, predictions), columns=np.unique(Y_test), index = np.unique(Y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font sizes

########## Check results on test data ##############

test_data_ = test_data.copy()

test_data_['prediction'] = predictions
test_data_['strategy'] = test_data_['prediction'] * test_data_[ret_1]

fig, axes = plt.subplots(nrows=2, figsize=(12,8), gridspec_kw={'height_ratios': [5, 1]})

test_data_[[ret_1, 'strategy']].cumsum().apply(np.exp).plot(ax=axes[0])
test_data_['prediction'].plot(ax=axes[1])

fig.suptitle('Out of sample - CART')

plot_tree_(model_cart, feature_cols)

text_representation = tree.export_text(model_cart, feature_names=feature_cols)
print(text_representation)

plt.show()
