import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn: to suppress SettingWithCopyWarning
import numpy as np
import seaborn as sns
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree

from genetic_selection import GeneticSelectionCV
pd.set_option('display.max_columns', 500)

def download_price_data(pair, start_date, end_date=None, where='yahoo'):

    if where == 'yahoo':
        if end_date is not None:
            return pd.DataFrame(data=yf.download(pair, start=start_date, end=end_date))
        else:
            return 'Adj Close', pd.DataFrame(data=yf.download(pair, start=start_date))

def features_engineering(df, price_col, lags_p_drets=None, lags_rets=None, lags_smas=None, lags_std=None, lags_rsi=None):
    feature_cols = []

    # price daily return features
    if lags_p_drets is not None:
        for i in range(len(lags_p_drets)):
            n = lags_p_drets[i]
            col = 'feat_lag_dreturn_' + str(n)  # f'lag_{lag}'
            df[col] = df['return'].shift(n)
            feature_cols.append(col)

    # price return features
    if lags_rets is not None:
        for i in range(len(lags_rets)):
            n = lags_rets[i]
            col = 'feat_lag_return_'+str(n) #f'lag_{lag}'
            df[col] = np.log(df[price_col] / df[price_col].shift(n))
            df[col] = df[col].shift(1)
            feature_cols.append(col)

    # add SMA (simple moving average) features
    if lags_smas is not None:
        for i in range(len(lags_smas)):
            n = lags_smas[i]
            col = 'feat_sma_' + str(n)
            df[col] = df[price_col].rolling(n).mean().shift(1)
            feature_cols.append(col)

    # add rolling variance columns as features
    if lags_std is not None:
        for i in range(len(lags_std)):
            n = lags_std[i]
            col = 'feat_std_' + str(n)
            df[col] = df['return'].rolling(n).std().shift(1)
            feature_cols.append(col)

    if lags_rsi is not None:
        # add RSI (Relative Strenght index) columns as features
        df['change'] = data[price_col].diff()
        df['dUp'] = df['change'].copy()
        df['dDown'] = df['change'].copy()
        df['dUp'].loc[df['dUp'] < 0] = 0
        df['dDown'].loc[df['dDown'] > 0] = 0
        df['dDown'] = -df['dDown']

        for i in range(len(lags_rsi)):
            n = lags_rsi[i]
            col = 'feat_rsi_' + str(n)
            sm_up = df['dUp'].rolling(n).mean()
            #TODO fix porcata below
            sm_down = df['dDown'].rolling(n).mean() + 1e-6  # PORCATA per non avere inf!!!!!!!!!!!!!!!!!!!!!!!!!!
            rs = sm_up / sm_down
            rsi = 100 - 100 / (1 + rs)
            df[col] = rsi.shift(1)
            feature_cols.append(col)

    ll = feature_cols.copy()
    ll.append('return')
    df = df[ll]
    return feature_cols, df


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

    print('\n')

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


########## inputs ###############
coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'

# test options for classification
num_folds = 10
seed = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# features
lags_price_daily_rets = [1, 7, 10, 14, 20, 28, 60, 90] #[1, 7, 28]
lags_rets = [5, 7, 10, 14, 20, 28, 60, 90]
lags_sma = [5, 7, 10, 14, 20, 28, 60, 90] # [7, 28, 90]
lags_std = [5, 7, 10, 14, 20, 28, 60, 90]
lags_rsi = [5, 7, 10, 14, 20, 28, 60, 90] #[7, 20, 60]]

#######
###########
###############

price_col, data = download_price_data(coin, s_date)
data = data[[price_col]].ffill()

data['return'] = np.log(data[price_col] / data[price_col].shift(1))

feature_cols, data = features_engineering(data, price_col, lags_p_drets=lags_price_daily_rets, lags_rets=lags_rets, lags_smas=lags_sma, lags_std=lags_std, lags_rsi=lags_rsi)
data.dropna(inplace=True)
print(feature_cols)
print(data.tail())


## Split data into training and test set
training_data = data.loc[:date_split]
test_data = data.loc[date_split:]

X_train = training_data[feature_cols]
Y_train = np.sign(training_data['return']) #np.where(training_data['return'] >= 0, 1.0, 0.0)

X_test = test_data[feature_cols]
Y_test = np.sign(test_data['return']) # np.where(test_data['return'] >= 0, 1.0, 0.0) it's ok if it is not binary. The algo
#automatically switches to multinomial.

print(Y_train)

######################################################################
####################   Grid Search: GB    ##########################
#####################################################################

## Results on test data set
# prepare model

gb_clf = GradientBoostingClassifier()

n_estimators = 250
learning_rate = 0.01
max_depth = 5
min_samples_leaf = 10
subsample = 0.5

## Results on test data set
estimator = GradientBoostingClassifier(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  subsample=subsample)


selector = GeneticSelectionCV(estimator, cv=10, n_generations=40)

selector = selector.fit(X_train, Y_train)
#print(selector.support_) # doctest: +NORMALIZE_WHITESPACE
print('Features:', X_train.columns[selector.support_])


