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
        self.ret_1d = 'ret_1d'
        self.transf_dict = {'transf_1': 'percentile', 'transf_2': 'min_max'}
        self.transf_features = {}
        self.price_data = None

    def get_price_data(self):
        return self.price_data

    def download_price_data(self):
        if self.path is not None:
            #get data from file
            if self.pair == 'BTC-USD':
                p = self.path +'btc-usd-coingecko_2015.csv'
            elif self.pair == 'ETH-USD':
                p = self.path + 'eth-usd-coingecko_2016.csv'
            elif self.pair == 'XRP-USD':
                p = self.path + 'xrp-usd-coingecko_2016.csv'
            idx = 'snapped_at'
            self.df = pd.read_csv(p)
            self.df[idx] = pd.to_datetime(self.df[idx])
            self.df.set_index(idx, inplace=True)
            self.price_col = 'price'
            print('Price data from file:', p)
            print('Data ' + idx, self.df.index[0])

        elif self.where == 'yahoo':
            if self.end_date is not None:
                self.df = pd.DataFrame(data=yf.download(self.pair, start=self.start_date, end=self.end_date))
            else:
                self.df = pd.DataFrame(data=yf.download(self.pair, start=self.start_date))
            self.price_col = 'Adj Close'

        print('Data starts from:', self.df.index[0])

        self.df = self.df.ffill()
        self.price_data = self.df[self.price_col]


    def merge_search_with_price_data(self):
        #TODO: if search term is given, this should be done automatically

        if self.search_term == 'bitcoin':
            filen = 'bicoin_searches_from_2015.xlsx'
        elif self.search_term == 'Ethereum':
            filen = 'ethereum_eth__searches_from_Feb_2016.xlsx'
        elif self.search_term == 'xrp':
            filen = 'ripple_xrp_searches_from_Feb_2017.xlsx'
        else:
            'ERROR!! No other search file stored for currencies other than bitcoin, eth and xrp'

        print('Reading search data from ' + filen)
        print('Merging price data with search data ...')

        df_search = pd.read_excel(self.path + filen)
        idx = 'date'
        df_search[idx] = pd.to_datetime(df_search[idx])
        df_search.set_index(idx, inplace=True)
        df_search.index = df_search.index.tz_localize('UTC')
        print('Time zone of search set to UTC')

        #TODO: need to change line below for cases where more than one search_term is used
        self.df = self.df.merge(df_search[self.search_term], left_index=True, right_index=True)
        self.df = self.df.ffill().dropna()

    def features_engineering_for_dec_tree(self, lags_p_smas, lags_smas, lags_rsi=None, lags_std=None):
        '''
        Note: not all cols are added to the list of features. This is because features such as past returns or past
        price are useless if not compared to something else (e.g. latest monthly return vs latest daily return).
        '''

        #TODO list:
        # 1. you should calculate the percentile of most of the features below.
        # 2. Given you are using smas and other quantities that depend on previous data, is this affecting the kfold
        # validation - the algo is using data it should not know. CORRECTION NEEDED

        ret_str = self.ret_1d
        ret_1d_lagged_str = 'feat_ret_1d'
        ret_1w_lagged_str = 'feat_ret_1w'
        feature_cols = []
        self.df[ret_str] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
        self.df[ret_1d_lagged_str] = self.df[ret_str].shift(1)
        self.df[ret_1w_lagged_str] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(7)).shift(1)
        feature_cols.append(ret_1d_lagged_str)
        feature_cols.append(ret_1w_lagged_str)

        transf1_ls = [ret_1d_lagged_str, ret_1w_lagged_str]
        #transf2_ls = []

        # 1-day lagged price
        self.df['price_lag_1'] = self.df[self.price_col].shift(1)

        # SMAs to be compared to 1-day lagged price
        sma_p_cols = []
        for i in range(len(lags_p_smas)):
            n = lags_p_smas[i]
            col = 'sma_p_' + str(n)
            self.df[col] = self.df[self.price_col].rolling(n).mean().shift(2) #shift 2 is correct because this is compared to 1-day lagged price (which is one day ahead, hence the 2)
            sma_p_cols.append(col)

        # Feature: SMAs vs last price
        for c in sma_p_cols:
            col = 'feat_price1_vs_'+c
            self.df[col] = np.log(self.df['price_lag_1'] / self.df[c])
            feature_cols.append(col)

        # SMAs to be compared against 1 year SMA
        #TODO careful, this is data intensive
        sma_cols = []
        for i in range(len(lags_smas)):
            n = lags_smas[i]
            col = 'sma_' + str(n)
            self.df[col] = self.df[self.price_col].rolling(n).mean().shift(1)
            sma_cols.append(col)

        # Features: short sma vs long sma
        self.df['sma_6m'] = self.df[self.price_col].rolling(180).mean().shift(1)
        self.df['sma_1y'] = self.df[self.price_col].rolling(365).mean().shift(1)
        for c in sma_cols:
            col1 = 'feat_' + c + '_sma_1y'
            col2 = 'feat_' + c + '_sma_6m'
            self.df[col1] = np.log(self.df[c] / self.df['sma_1y'])
            self.df[col2] = np.log(self.df[c] / self.df['sma_6m'])
            feature_cols.append(col1)
            feature_cols.append(col2)

        data_ = {'r1': self.df[ret_1d_lagged_str], 'r2': self.df[ret_1d_lagged_str].shift(1),
                 'r3': self.df[ret_1d_lagged_str].shift(2)}
        df_ = pd.DataFrame(data=data_)

        #Features: momentum indicator that looks at previous 2 days
        conditions = [(df_['r1'] > 0) & (df_['r2'] > 0), (df_['r1'] < 0) & (df_['r2'] < 0)]
        values = [1, -1]
        self.df['feat_mom_2'] = np.select(conditions, values, default=0)

        #Features: momentum indicator that looks at previous 3 days
        conditions = [(df_['r1'] > 0) & (df_['r2'] > 0) & (df_['r3'] > 0),
                      (df_['r1'] < 0) & (df_['r2'] < 0) & (df_['r3'] < 0)]
        values = [1, -1]
        self.df['feat_mom_3'] = np.select(conditions, values, default=0)

        feature_cols.append('feat_mom_2')
        feature_cols.append('feat_mom_3')

        # Features: rolling variance columns as features
        if lags_std is not None:
            for i in range(len(lags_std)):
                n = lags_std[i]
                col = 'feat_std_' + str(n)
                self.df[col] = self.df[ret_str].rolling(n).std().shift(1)
                feature_cols.append(col)

        # Features: RSI (Relative Strength index)
        if lags_rsi is not None:

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

        #Features: latest week average searches vs past 3 weeks average searches
        if self.search_term is not None:
            data_ = {'week1_avg_search': self.df[self.search_term].shift(1).rolling(7).mean(),
                     'week3_avg_search': self.df[self.search_term].shift(8).rolling(21).mean()} #shift by 8 since need to shift by 1 week + 1 day
            df_ = pd.DataFrame(data=data_)
            self.df['feat_search'] = np.log(df_['week1_avg_search'] / df_['week3_avg_search'])
            feature_cols.append('feat_search')
            print('Features using google trends have been created...')
        else:
            print('Search term is missing. Google trend data will not be used!!!')

        ll = feature_cols.copy()
        ll.append(self.ret_1d)
        self.df = self.df[ll].dropna()
        self.feature_cols = feature_cols


        transf_1 = self.transf_dict['transf_1']
        #transf_2 = self.transf_dict['transf_2']
        self.transf_features = {transf_1: transf1_ls}#, transf_2: transf2_ls} #1. Percentile, 2. MinMax


    def split_train_test(self, date_split):
        ## Split data into training and test set
        training_data = self.df.loc[:date_split]
        test_data = self.df.loc[date_split:]

        X_train = training_data[self.feature_cols]
        Y_train = np.sign(training_data[self.ret_1d])

        X_test = test_data[self.feature_cols]
        Y_test = np.sign(test_data[self.ret_1d])

        return X_train, Y_train, X_test, Y_test


    def get_result_data(self, df, preds):
        df_ = df.copy()

        df_['prediction'] = preds

        # np.exp(df_[self.ret_1d]) = x_i / x_{i-1} = r_i + 1
        df_['ret_'] = np.exp(df_[self.ret_1d]) - 1
        df_['strategy'] = df_['ret_'] * df_['prediction'] + 1
        df_['cum_return'] = np.cumprod(df_['ret_'] + 1) #df_[self.ret_1d].cumsum().apply(np.exp)
        df_['cum_strategy'] = df_['strategy'].cumprod()

        return df_


    def backtest_strategy_with_fees(self, df, preds, fee=None, short_funding=None):
        df_ = df.copy()

        df_['prediction'] = preds

        df_['shifted_prediction'] = df_['prediction'].shift(1)
        df_['buy_sell_fees'] = np.abs((df_['prediction'] - df_['shifted_prediction']))

        inv = 1
        if df_['prediction'].iloc[0] != 1:
            inv = 0
        df_['buy_sell_fees'].iloc[0] = inv
        df_['buy_sell_fees'] = np.where(df_['buy_sell_fees'] != 0, fee, 0)

        conditions = [df_['prediction'] < 0, df_['prediction'] >= 0]
        values = [short_funding, 0]
        df_['short_fees'] = np.select(conditions, values, default=0)

        df_['strategy'] = df_['prediction'] * df_[self.ret_1d]
        df_['cum_return'] = df_[self.ret_1d].cumsum().apply(np.exp)
        df_['cum_strategy_no_fees'] = df_['strategy'].cumsum().apply(np.exp)
        df_['cum_strategy'] = (df_['strategy'] - df_['buy_sell_fees'] - df_['short_fees']).cumsum().apply(np.exp)

        return df_