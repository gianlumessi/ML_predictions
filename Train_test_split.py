import numpy as np
from sklearn.base import clone
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from Data_manager import Data_manager


# Function for non-anchored walk-forward optimization
class Walk_forward_nonanchored_nonoverlapping_ts_split:
    '''
    This is non-anchored, non-overlapping walk-froward split. Run code below to understand what this means.
    '''

    def __init__(self, n_splits, train_pct):
        self.n_splits = n_splits
        self.train_pct = train_pct

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y = None, groups = None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_pct * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


# Function for non-anchored walk-forward optimization
class Walk_forward_nonanchored_ts_split:
    '''
    This is non-anchored, non-overlapping walk-froward split. Run code below to understand what this means.
    '''

    def __init__(self, fold_size, train_pct, X, first_test_index=0):
        self.fold_size = fold_size
        self.train_pct = train_pct
        self.first_test_index = first_test_index
        self.train_fold_size = int(self.fold_size * self.train_pct)
        self.test_fold_size = self.fold_size - self.train_fold_size
        self.first_test_index = first_test_index
        if first_test_index == 0:
            self.n_samples = len(X)
        elif first_test_index < self.train_fold_size:
            print('!!!ERROR!!! The first_test_index parameter provided is smaller than the train_fold_size, which does not make '
                  'sense !!!')
        else:
            self.n_samples = len(X) - first_test_index + self.train_fold_size

        self.X = X
        self.n_splits = int((self.n_samples - self.train_fold_size) // self.test_fold_size)

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X=None, y=None, groups=None, leftover=0.25):
        margin = 0
        leftover_ = round(self.test_fold_size * leftover, 0)

        if self.first_test_index != 0:
            burn = self.first_test_index - self.train_fold_size
            indices = np.arange(self.n_samples) + burn
            last_ind = indices[-1] - burn
        else:
            indices = np.arange(self.n_samples)
            last_ind = indices[-1]

        start = 0

        i = 0
        stop = 0
        while stop < last_ind:
            if i != 0:
                start = start + self.test_fold_size
            mid = int(start + self.train_fold_size)
            stop = int(mid + self.test_fold_size)
            remaining = last_ind - stop
            if (remaining > 0) and (remaining <= leftover_):
                stop = last_ind
            if stop >= last_ind:
                stop = last_ind + 1
            i +=1
            yield indices[start: mid], indices[mid + margin: stop]

    def run_analysis(self, grid, Y, data, ret_1, leftover=0.25, verbose=True):
        print('\nWalk forward testing started ...')
        frames = []
        i = 0
        for train_index, test_index in self.split(self.X, leftover=leftover):
            clone_clf = clone(grid)
            X_train_folds = self.X.iloc[train_index]
            y_train_folds = Y.iloc[train_index]
            X_test_fold = self.X.iloc[test_index]
            y_test_fold = Y.iloc[test_index]
            clone_clf.fit(X_train_folds, y_train_folds)
            y_pred = clone_clf.predict(X_test_fold)

            if verbose:
                ## Results:
                print('Fold ' + str(i) + '. Test dates:', X_test_fold.index[0], ' - ', X_test_fold.index[-1])
                print('Best estimator:', clone_clf.best_estimator_, ', best score:', clone_clf.best_score_)
                print('Accuracy:\t', accuracy_score(y_test_fold, y_pred))
                print('Recall:\t', recall_score(y_test_fold, y_pred))
                print('Precision:\t', precision_score(y_test_fold, y_pred))
                print('f1:\t', f1_score(y_test_fold, y_pred), '\n')

            y_pred = pd.Series(y_pred, index=X_test_fold.index, name='prediction')
            returns = data.loc[y_test_fold.index, ret_1]
            tmp_df = pd.concat([returns, y_pred], axis=1)
            frames.append(tmp_df)
            i += 1

        print('\nEnd of walk forward testing.')
        result_df = pd.concat(frames)
        return result_df


    def plot_sets(self, train_index, test_index, i):
        plt.fill_between(train_index, -i + 0.5, -i - 0.5, color='blue')
        plt.fill_between(test_index, -i + 0.5, -i - 0.5, color='orange')

class Predefined_ts_split:
    def __init__(self, train_period_dict, test_period_dict):
        '''

        :param train_period_dict: Dictionary where keys are integer and values are lists of dates
        :param test_period_dict: Dictionary where keys are integer and values are lists of dates
        '''
        self.train_period_dict = train_period_dict
        self.test_period_dict = test_period_dict

    def split(self):
        for i in range(len(self.train_period_dict)):
            yield self.train_period_dict[i], self.test_period_dict[i]

    def plot_sets(self, train_index, test_index, i):
        plt.fill_between(train_index, -i + 0.5, -i - 0.5, color='blue')
        plt.fill_between(test_index, -i + 0.5, -i - 0.5, color='orange')

class Period_search_split():

    def __init__(self, X, train_period):
        '''

        :param X: data with a date index
        :param train_period: a list with two entries. Entry 0 is beginning of period, entry 1 is end of period
        '''
        self.X = X
        self.train_period = train_period

    def get_train_period(self):
        return self.X.loc[self.train_period[0]:self.train_period[-1]].index

    def get_test_periods(self):
        '''
        :return: the test periods
        '''
        n_samples = int(self.X.loc[self.train_period[0]:self.train_period[-1]].shape[0])

        init_prev_dt = self.train_period[0] #string, e.g. '2022-12-31'
        init_prev_dt = self.X.loc[init_prev_dt].name # date format, e.g. 2022-01-01 00:00:00
        #end_cur_idx = 999999
        init_cur_idx = 999999

        while init_cur_idx - n_samples > 0:
            end_cur_idx = list(self.X.index).index(init_prev_dt) - 1
            end_cur_dt = self.X.index[end_cur_idx]
            init_cur_idx = end_cur_idx - n_samples
            init_cur_dt = self.X.index[init_cur_idx]

            yield self.X.loc[init_cur_dt:end_cur_dt].index#init_cur_dt, end_cur_dt

            init_prev_dt = init_cur_dt#self.X.index[init_cur_idx-n_samples]#end_cur_dt

if __name__ == "__main__":
    n = 95
    vec = range(n)
    fold_size = 53
    test_pct = 0.75
    cl = Walk_forward_nonanchored_ts_split(fold_size, test_pct, vec)

    plt.figure(0)
    for i, (train_index, test_index) in enumerate(cl.split(vec, leftover=0.3)):
        print("\nFold:", i)
        print("Train: index=", train_index, ' -- Size:', len(train_index))
        print("Test:  index=", test_index, ' -- Size:', len(test_index))

        cl.plot_sets(train_index, test_index, i)

    print('\n\n')

    first_test_index = 60
    cl = Walk_forward_nonanchored_ts_split(fold_size, test_pct, vec, first_test_index=first_test_index)

    plt.figure(1)
    for i, (train_index, test_index) in enumerate(cl.split(vec, leftover=0.3)):
        print("\nFold:", i)
        print("Train: index=", train_index, ' -- Size:', len(train_index))
        print("Test:  index=", test_index, ' -- Size:', len(test_index))

        cl.plot_sets(train_index, test_index, i)

    train_period_dict = {0: ['2016-01-01', '2016-12-31'], 1: ['2017-01-01', '2017-12-31'], 2: ['2018-01-01', '2018-12-31']}
    test_period_dict = {0: ['2020-01-01', '2020-12-31'], 1: ['2021-01-01', '2021-12-31'], 2: ['2022-01-01', '2022-12-31']}
    pt = Predefined_ts_split(train_period_dict, test_period_dict)

    for i, (train_dates, test_dates) in enumerate(pt.split()):
        print(train_dates, test_dates)


    dm = Data_manager('BTC-USD', '2015-01-01', where='yahoo')
    dm.download_price_data()
    data = dm.get_data()
    train_period = ['2022-01-01', '2022-12-31']
    print('Period search:')
    hp = Period_search_split(data, train_period)
    for i, (train_dates, test_dates) in enumerate(hp.get_test_periods()):
        print(train_dates, test_dates)

    plt.show()
