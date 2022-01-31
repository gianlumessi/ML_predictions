import yfinance as yf
import pandas as pd
import numpy as np
import os

class Data_manager:

    def __init__(self, pair, start_date, end_date=None, where=None, search_term=None, path=None):
        self.df = None
        self.feature_cols = None
        self.price_col = None
        self.search_term = search_term
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        if path == 'local_file':
            self.path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, '/') + '/Data/'# 'C:/Users/Gianluca/Desktop/Coding projects/Crypto_prediction_with_ML/'
        elif path is not None:
            self.path = path
        self.where = where
        if where == None:
            self.where = 'yahoo'

    def download_price_data(self):
        p = self.path + 'btc-usd-coingecko_2015.csv'
        if p is not None:
            idx = 'snapped_at'
            self.df = pd.read_csv(p)
            self.df[idx] = pd.to_datetime(self.df[idx])
            self.df.set_index(idx, inplace=True)
            self.price_col = 'price'
            print('Price data from file:', p)

        elif self.where == 'yahoo':
            if self.end_date is not None:
                self.df = pd.DataFrame(data=yf.download(self.pair, start=self.start_date, end=self.end_date))
            else:
                self.df = pd.DataFrame(data=yf.download(self.pair, start=self.start_date))
            self.price_col = 'Adj Close'

        print('Data starts from:', self.df.index[0])

        self.df = self.df.ffill()


    def merge_search_with_price_data(self):

        if self.search_term == 'bitcoin':
            filen = 'bicoin_searches_from_2015.xlsx'
        else:
            'ERROR!! No other search file stored for currencies other than bitcoin'

        print('Merging price data with search data...')

        df_search = pd.read_excel(self.path + filen)
        idx = 'date'
        df_search[idx] = pd.to_datetime(df_search[idx])
        df_search.set_index(idx, inplace=True)
        df_search.index = df_search.index.tz_localize('UTC')
        print('Time zone of search set to UTC')

        #TODO: need to change line below for cases where more than one search_term is used
        self.df = self.df.merge(df_search[self.search_term], left_index=True, right_index=True)
        self.df = self.df.ffill().dropna()


    def features_engineering(self, lags_p_drets=None, lags_rets=None, lags_smas=None, lags_std=None,
                             lags_rsi=None, lags_search=None, lags_search_sma=None, lags_price=None):
        '''

        :param lags_p_drets:
        :param lags_rets:
        :param lags_smas:
        :param lags_std:
        :param lags_rsi:
        :param lags_search:
        :param lags_search_sma:
        :param lags_price:
        :return:

        Note: not all cols are added to the list of features. This is because features such as past returns or past
        price are useless if not compared to something else (e.g. latest monthly return vs latest daily return).
        '''

        feature_cols = []
        col_list = []
        self.df['return'] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))

        # lagged prices features
        if lags_price is not None:
            for i in range(len(lags_price)):
                n = lags_price[i]
                col = 'feat_price_' + str(n)  # f'lag_{lag}'
                self.df[col] = self.df[self.price_col].shift(n)
                col_list.append(col)

        # price daily return features
        if lags_p_drets is not None:
            for i in range(len(lags_p_drets)):
                n = lags_p_drets[i]
                col = 'feat_dreturn_' + str(n)  # f'lag_{lag}'
                self.df[col] = self.df['return'].shift(n)
                col_list.append(col)

        # price return features
        if lags_rets is not None:
            for i in range(len(lags_rets)):
                n = lags_rets[i]
                col = 'feat_return_' + str(n)  # f'lag_{lag}'
                self.df[col] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(n))
                self.df[col] = self.df[col].shift(1)
                col_list.append(col)

        # add SMA (simple moving average) features
        if lags_smas is not None:
            for i in range(len(lags_smas)):
                n = lags_smas[i]
                col = 'feat_sma_' + str(n)
                self.df[col] = self.df[self.price_col].rolling(n).mean().shift(1)
                col_list.append(col)

        # add rolling variance columns as features
        if lags_std is not None:
            for i in range(len(lags_std)):
                n = lags_std[i]
                col = 'feat_std_' + str(n)
                self.df[col] = self.df['return'].rolling(n).std().shift(1)
                col_list.append(col)

        if lags_rsi is not None:
            # add RSI (Relative Strenght index) columns as features
            self.df['change'] = self.df[self.price_col].diff()
            self.df['dUp'] = self.df['change'].copy()
            self.df['dDown'] = self.df['change'].copy()
            self.df['dUp'].loc[self.df['dUp'] < 0] = 0
            self.df['dDown'].loc[self.df['dDown'] > 0] = 0
            self.df['dDown'] = -self.df['dDown']

            for i in range(len(lags_rsi)):
                n = lags_rsi[i]
                col = 'feat_rsi_' + str(n)
                sm_up = self.df['dUp'].rolling(n).mean()
                # TODO fix porcata below
                sm_down = self.df['dDown'].rolling(n).mean() + 1e-6  # PORCATA per non avere inf!!!!!!!!!!!!!!!!!!!!!!!!!!
                rs = sm_up / sm_down
                rsi = 100 - 100 / (1 + rs)
                self.df[col] = rsi.shift(1)
                feature_cols.append(col)

        #features engineering for search terms
        if self.search_term is not None:
            if (lags_search is None or lags_search_sma is None):
                print('A search term was given, but attributes for features engineering of search terms are missing!!!')

            for i in range(len(lags_search)):
                n = lags_search[i]
                col = 'feat_search_lag_' + str(n)  # f'lag_{lag}'
                self.df[col] = self.df[self.search_term].shift(n)
                col_list.append(col)

            for i in range(len(lags_search_sma)):
                n = lags_search_sma[i]
                col = 'feat_search_sma_' + str(n)
                self.df[col] = self.df[self.search_term].rolling(n).mean().shift(1)
                col_list.append(col)

        ll = feature_cols.copy()
        ll.append('return')
        self.df = self.df[ll + col_list].dropna()
        self.feature_cols = feature_cols


    def combo_subroutine(self, df0, str, lst):
        cols = [col for col in self.df.columns if str in col]
        if cols is not None:
            df1 = pd.DataFrame()
            for i in range(len(cols)):
                for j in range(i+1, len(cols)-1):
                    col = cols[i] + '_' + cols[j]
                    #df0[col] = self.df[cols[i]] - self.df[cols[j]]
                    df1[col] = self.df[cols[i]] - self.df[cols[j]]
                    lst.append(col)

        df0 = pd.concat([df0, df1], axis=1)
        return df0, lst


    def combo_subroutine_different(self, df0, l_1, l_2, lst):

        df1 = pd.DataFrame()

        for l in l_1:
            for s in l_2:
                col = l + '_' + s
                df1[col] = (self.df[l] - self.df[s])/self.df[s]
                lst.append(col)

        df0 = pd.concat([df0, df1], axis=1)
        return df0, lst


    def combine_features(self):
        '''
        Create new features by combining (i.e. subtracting) existing features
        :return:
        '''

        combined_cols = []
        df_ = self.df.copy()

        # lag price - sma
        lag_price_cols = [col for col in self.df.columns if 'feat_price_' in col]
        sma_cols = [col for col in self.df.columns if 'feat_sma_' in col]
        if (lag_price_cols and sma_cols) is not None:
            df_, combined_cols = self.combo_subroutine_different(df_, lag_price_cols, sma_cols, combined_cols)

        # lag return
        df_, combined_cols = self.combo_subroutine(df_, 'feat_return_', combined_cols)

        # lag dreturn
        df_, combined_cols = self.combo_subroutine(df_, 'feat_dreturn_', combined_cols)

        # lag std
        df_, combined_cols = self.combo_subroutine(df_, 'feat_std_', combined_cols)

        if self.search_term is not None:
            lag_search = [col for col in self.df.columns if 'feat_search_lag_' in col]
            lag_search_sma = [col for col in self.df.columns if 'feat_search_sma_' in col]
            if (lag_search or lag_search_sma) is None:
                print('ERROR!! A search term is given but the lag_search columns or lag_search_sma are None!!')
            else:
                df_, combined_cols = self.combo_subroutine_different(df_, lag_search, lag_search_sma, combined_cols)

        self.feature_cols = self.feature_cols + combined_cols
        self.df = df_
        print('Combined feature columns:', combined_cols)


    def split_train_test(self, date_split):
        ## Split data into training and test set
        training_data = self.df.loc[:date_split]
        test_data = self.df.loc[date_split:]

        X_train = training_data[self.feature_cols]
        Y_train = np.sign(training_data['return'])

        X_test = test_data[self.feature_cols]
        Y_test = np.sign(test_data['return'])

        return X_train, Y_train, X_test, Y_test


    def get_result_data(self, df, preds):
        df_ = df.copy()

        df_['prediction'] = preds
        df_['strategy'] = df_['prediction'] * df_['return']
        df_[['cum_return', 'cum_strategy']] = df_[['return', 'strategy']].cumsum().apply(np.exp)

        return df_
