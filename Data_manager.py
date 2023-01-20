import yfinance as yf
import pandas as pd
import numpy as np
import os
from ta.momentum import rsi, stoch#, import volme as vol
from ta.volume import on_balance_volume

class Data_manager:

    def __init__(self, pair, start_date, end_date=None, where=None, search_term=None, path=None):
        self.data = None
        self.raw_data = None
        self.feature_names = []
        self.close_price_col = None
        self.search_term = search_term
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        if path == 'local_file':
            self.path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, '/') + '/Data/'# 'C:/Users/Gianluca/Desktop/Coding projects/Crypto_prediction_with_ML/'
        elif path is not None:
            self.path = path
        else:
            self.path = None
        self.where = where
        if where == None:
            self.where = 'yahoo'
        self.ret_1d = 'ret_1d'
        self.transf_dict = {'transf_1': 'percentile', 'transf_2': 'min_max'}
        self.transf_features = {}
        self.vol_col = None

    def get_price_data(self):
        return self.data[self.close_price_col]

    def reset_data(self):
        self.data = self.raw_data.copy()

    def get_data(self):
        return self.data

    def download_price_data(self, raw_file=None, is_save_raw_data=False):

        if (raw_file is not None) and (os.path.exists(raw_file)):
            print('Data was already downloaded and stored here:', raw_file, '. It will be downloaded again ...')

        if self.path is not None:
            #get data from file
            if self.pair == 'BTC-USD':
                p = self.path +'btc-usd-coingecko_2015.csv'
            elif self.pair == 'ETH-USD':
                p = self.path + 'eth-usd-coingecko_2016.csv'
            elif self.pair == 'XRP-USD':
                p = self.path + 'xrp-usd-coingecko_2016.csv'
            idx = 'snapped_at'
            self.data = pd.read_csv(p)
            self.data[idx] = pd.to_datetime(self.data[idx])
            self.data.set_index(idx, inplace=True)
            self.close_price_col = 'price'
            print('Price data from file:', p)
            print('Data ' + idx, self.data.index[0])

        elif self.where == 'yahoo':
            if self.end_date is not None:
                self.data = pd.DataFrame(data=yf.download(self.pair, start=self.start_date, end=self.end_date))
            else:
                self.data = pd.DataFrame(data=yf.download(self.pair, start=self.start_date))
            self.close_price_col = 'Adj Close'
            self.vol_col = 'Volume'

        print('Data start date:', self.data.index[0])
        print('Data end date:', self.data.index[-1])

        self.data = self.data.ffill()
        #self.close_price = self.data[self.close_price_col]

        if self.search_term is not None:
            #get google trend data for search_term
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

            # TODO: need to change line below for cases where more than one search_term is used
            self.data = self.data.merge(df_search[self.search_term], left_index=True, right_index=True)
            self.data = self.data.ffill().dropna()

        if (raw_file is not None) and (not os.path.exists(raw_file)) and is_save_raw_data:
            print('Saving raw data to:', raw_file)
            self.data.to_excel(raw_file)

        self.raw_data = self.data.copy()

    def add_returns_to_data(self):
        self.data[self.ret_1d] = np.log(self.data[self.close_price_col] / self.data[self.close_price_col].shift(1))

    def dropna_in_data(self):
        self.data = self.data.dropna()

    def features_engineering_ONBALVOL(self):
        feature_name = 'on_bal_vol'
        self.data[feature_name] = on_balance_volume(self.data[self.close_price_col], self.data[self.vol_col]).shift(1)
        self.feature_names.append(feature_name)

    def features_engineering_PCTCHANGE(self, list_n, id_str=None):
        if id_str is None:
            id_str = 'pctChange_'
        for n in list_n:
            feature_name = id_str + str(n)
            pctChange_ts = self.data[self.close_price_col].pct_change(periods=n)
            self.data[feature_name] = pctChange_ts.shift(1)
            self.feature_names.append(feature_name)

    def features_engineering_STD(self, list_n, id_str=None):

        if id_str is None:
            id_str = 'std_'
        for n in list_n:
            feature_name = id_str + str(n)
            std_ts = self.data[self.close_price_col].rolling(n).std()
            self.data[feature_name] = std_ts.shift(1)
            self.feature_names.append(feature_name)

    def features_engineering_RSI(self, list_n, id_str=None):

        if id_str is None:
            id_str = 'rsi_'
        for n in list_n:
            feature_name = id_str + str(n)
            rsi_ts = rsi(self.data[self.close_price_col], window=n)
            self.data[feature_name] = rsi_ts.shift(1)
            self.feature_names.append(feature_name)

    def features_engineering_STOCHOSC(self, list_n, id_str=None):

        if id_str is None:
            id_str = 'stochOsc_'
        for n in list_n:
            feature_name = id_str + str(n)
            #so_ts = stoch(high: pandas.core.series.Series, low: pandas.core.series.Series, close: pandas.core.series.Series, window: int = 14, smooth_window: int = 3)
            so_ts = stoch(self.data['High'], self.data['Low'], self.data[self.close_price_col], window=n)
            self.data[feature_name] = so_ts.shift(1)
            self.feature_names.append(feature_name)

    def features_engineering_SMA(self, list_n, id_str=None, is_calc_sma_diffs=True):
        '''
        Note: not all cols are added to the list of features. This is because features such as past returns or past
        price are useless if not compared to something else (e.g. latest monthly return vs latest daily return).
        '''

        if id_str is None:
            id_str = 'sma_'

        for n in list_n:
            feature_name = id_str + str(n)

            self.data['to_drop_ma_' + str(n)] = self.data[self.close_price_col].rolling(n).mean()
            self.data['to_drop_std' + str(n)] = self.data[self.close_price_col].rolling(n).std()

            self.data[feature_name] = ((self.data[self.close_price_col] - self.data['to_drop_ma_' + str(n)]) / self.data['to_drop_std' + str(n)]).shift(1)
            self.feature_names.append(feature_name)

        if is_calc_sma_diffs:
            for n1 in list_n:
                for n2 in list_n:
                    if n2 > n1:
                        feature_name = id_str + str(n1) + '_minus_' + id_str + str(n2)
                        self.data[feature_name] = self.data[id_str + str(n1)] - self.data[id_str + str(n2)]
                        self.feature_names.append(feature_name)

        column_list = list(self.data)
        list_to_drop = [s for s in column_list if "to_drop" in s]

        self.data = self.data.drop(list_to_drop, axis=1)

    def run_simple_backtest(self, df, preds=None, trading_costs=None, strategy_type='long_only'):
        '''
        Run a backtest using vectorization approach. !!! Note that the prediction at time t_i indicates that you should:
        buy/sell at t_{i+1} open price, not t_{i+1} closing price!!! For cryptos the t_{i+1} open price is equal to t_i
        closing price. Therefore the simple approach below for backtesting works.

        :param df:
        :param preds:
        :param trading_costs:
        :param strategy_type:
        :return:
        '''
        df_ = df.copy()

        df_['ret_'] = np.exp(df_[self.ret_1d]) - 1

        if preds is not None:
            df_['prediction'] = preds

        if strategy_type == 'long_only':
            # check only abs diffs of prediction column to see when trades occur
            # fillna(0) to fill initial nans of the strategy (if any)
            df_['trades'] = df_['prediction'].diff().fillna(0).abs()
            print('n of trades (ones):', df_['trades'].value_counts())
            df_['strategy'] = (df_['ret_'] * df_['prediction'] - df_['trades'] * trading_costs)
        elif strategy_type == 'long_short':
            pass
        else:
            # np.exp(df_[self.ret_1d]) = x_i / x_{i-1} = r_i + 1
            df_['strategy'] = df_['ret_'] * df_['prediction']

        df_['cum_strategy'] = (df_['strategy'] + 1).cumprod() #Strategy cumulative returns
        df_['cum_return'] = np.cumprod(df_['ret_'] + 1) #Coin cumulative returns

        return df_

    def features_engineering_for_SMA_OLD(self, n1, n2):
        '''
        # Note: not all cols are added to the list of features. This is because features such as past returns or past
        # price are useless if not compared to something else (e.g. latest monthly return vs latest daily return).
        '''

        ret_str = self.ret_1d
        feature_cols = []

        #TODO: this should be done in a separate method
        self.data[ret_str] = np.log(self.data[self.close_price_col] / self.data[self.close_price_col].shift(1))

        col_ma1 = 'ma1'
        col_ma2 = 'ma2'
        col_std1 = 'std1'
        col_std2 = 'std2'
        self.data[col_ma1] = self.data[self.close_price_col].rolling(n1).mean()
        self.data[col_ma2] = self.data[self.close_price_col].rolling(n2).mean()
        self.data[col_std1] = self.data[self.close_price_col].rolling(n1).std()
        self.data[col_std2] = self.data[self.close_price_col].rolling(n2).std()

        feat1 = 'feat1'
        feat2 = 'feat2'
        self.data[feat1] = ((self.data[self.close_price_col] - self.data[col_ma1]) / self.data[col_std1]).shift(1)
        self.data[feat2] = ((self.data[self.close_price_col] - self.data[col_ma2]) / self.data[col_std2]).shift(1)
        feature_cols.append(feat1)
        feature_cols.append(feat2)

        tmp = feature_cols.copy()

        #TODO: ret_1d should be added to self.df in separate method
        tmp.append(self.ret_1d)

        #TODO: this should be in separate method
        self.data = self.data[tmp].dropna()

        #TODO: Features should be appended to self.feature_cols
        self.feature_names = feature_cols

    def features_engineering_for_dec_tree(self, lags_p_smas, lags_smas, lags_rsi=None, lags_std=None):
        '''
        Note: not all cols are added to the list of features. This is because features such as past returns or past
        price are useless if not compared to something else (e.g. latest monthly return vs latest daily return).
        '''

        #TODO list:
        # this uses the OLD PARADIGM. Should align it with other features eng methods, like features_engineering_rsi
        # 1. you should calculate the percentile of most of the features below.
        # 2. Given you are using smas and other quantities that depend on previous data, is this affecting the kfold
        # validation - the algo is using data it should not know. CORRECTION NEEDED

        ret_str = self.ret_1d
        ret_1d_lagged_str = 'feat_ret_1d'
        ret_1w_lagged_str = 'feat_ret_1w'
        feature_cols = []
        self.data[ret_str] = np.log(self.data[self.close_price_col] / self.data[self.close_price_col].shift(1))
        self.data[ret_1d_lagged_str] = self.data[ret_str].shift(1)
        self.data[ret_1w_lagged_str] = np.log(self.data[self.close_price_col] / self.data[self.close_price_col].shift(7)).shift(1)
        feature_cols.append(ret_1d_lagged_str)
        feature_cols.append(ret_1w_lagged_str)

        transf1_ls = [ret_1d_lagged_str, ret_1w_lagged_str]
        #transf2_ls = []

        # 1-day lagged price
        self.data['price_lag_1'] = self.data[self.close_price_col].shift(1)

        # SMAs to be compared to 1-day lagged price
        sma_p_cols = []
        for i in range(len(lags_p_smas)):
            n = lags_p_smas[i]
            col = 'sma_p_' + str(n)
            self.data[col] = self.data[self.close_price_col].rolling(n).mean().shift(2) #shift 2 is correct because this is compared to 1-day lagged price (which is one day ahead, hence the 2)
            sma_p_cols.append(col)

        # Feature: SMAs vs last price
        for c in sma_p_cols:
            col = 'feat_price1_vs_'+c
            self.data[col] = np.log(self.data['price_lag_1'] / self.data[c])
            feature_cols.append(col)

        # SMAs to be compared against 1 year SMA
        #TODO careful, this is data intensive
        sma_cols = []
        for i in range(len(lags_smas)):
            n = lags_smas[i]
            col = 'sma_' + str(n)
            self.data[col] = self.data[self.close_price_col].rolling(n).mean().shift(1)
            sma_cols.append(col)

        # Features: short sma vs long sma
        self.data['sma_6m'] = self.data[self.close_price_col].rolling(180).mean().shift(1)
        self.data['sma_1y'] = self.data[self.close_price_col].rolling(365).mean().shift(1)
        for c in sma_cols:
            col1 = 'feat_' + c + '_sma_1y'
            col2 = 'feat_' + c + '_sma_6m'
            self.data[col1] = np.log(self.data[c] / self.data['sma_1y'])
            self.data[col2] = np.log(self.data[c] / self.data['sma_6m'])
            feature_cols.append(col1)
            feature_cols.append(col2)

        data_ = {'r1': self.data[ret_1d_lagged_str], 'r2': self.data[ret_1d_lagged_str].shift(1),
                 'r3': self.data[ret_1d_lagged_str].shift(2)}
        df_ = pd.DataFrame(data=data_)

        #Features: momentum indicator that looks at previous 2 days
        conditions = [(df_['r1'] > 0) & (df_['r2'] > 0), (df_['r1'] < 0) & (df_['r2'] < 0)]
        values = [1, -1]
        self.data['feat_mom_2'] = np.select(conditions, values, default=0)

        #Features: momentum indicator that looks at previous 3 days
        conditions = [(df_['r1'] > 0) & (df_['r2'] > 0) & (df_['r3'] > 0),
                      (df_['r1'] < 0) & (df_['r2'] < 0) & (df_['r3'] < 0)]
        values = [1, -1]
        self.data['feat_mom_3'] = np.select(conditions, values, default=0)

        feature_cols.append('feat_mom_2')
        feature_cols.append('feat_mom_3')

        # Features: rolling variance columns as features
        if lags_std is not None:
            for i in range(len(lags_std)):
                n = lags_std[i]
                col = 'feat_std_' + str(n)
                self.data[col] = self.data[ret_str].rolling(n).std().shift(1)
                feature_cols.append(col)

        # Features: RSI (Relative Strength index)
        if lags_rsi is not None:

            self.data['change'] = self.data[self.close_price_col].diff()
            self.data['dUp'] = self.data['change'].copy()
            self.data['dDown'] = self.data['change'].copy()
            self.data['dUp'].loc[self.data['dUp'] < 0] = 0
            self.data['dDown'].loc[self.data['dDown'] > 0] = 0
            self.data['dDown'] = -self.data['dDown']

            for i in range(len(lags_rsi)):
                n = lags_rsi[i]
                col = 'feat_rsi_' + str(n)
                sm_up = self.data['dUp'].rolling(n).mean()
                # TODO fix porcata below
                sm_down = self.data['dDown'].rolling(n).mean() + 1e-6  # PORCATA per non avere inf!!!!!!!!!!!!!!!!!!!!!!!!!!
                rs = sm_up / sm_down
                rsi = 100 - 100 / (1 + rs)
                self.data[col] = rsi.shift(1)
                feature_cols.append(col)

        #Features: latest week average searches vs past 3 weeks average searches
        if self.search_term is not None:
            data_ = {'week1_avg_search': self.data[self.search_term].shift(1).rolling(7).mean(),
                     'week3_avg_search': self.data[self.search_term].shift(8).rolling(21).mean()} #shift by 8 since need to shift by 1 week + 1 day
            df_ = pd.DataFrame(data=data_)
            self.data['feat_search'] = np.log(df_['week1_avg_search'] / df_['week3_avg_search'])
            feature_cols.append('feat_search')
            print('Features using google trends have been created...')
        else:
            print('Search term is missing. Google trend data will not be used!!!')

        ll = feature_cols.copy()
        ll.append(self.ret_1d)
        self.data = self.data[ll].dropna()
        self.feature_names = feature_cols

        transf_1 = self.transf_dict['transf_1']
        #transf_2 = self.transf_dict['transf_2']
        self.transf_features = {transf_1: transf1_ls}#, transf_2: transf2_ls} #1. Percentile, 2. MinMax

    def split_train_test(self, date_split):
        ## Split data into training and test set
        training_data = self.data.loc[:date_split]
        test_data = self.data.loc[date_split:]

        X_train = training_data[self.feature_names]
        Y_train = np.sign(training_data[self.ret_1d])

        X_test = test_data[self.feature_names]
        Y_test = np.sign(test_data[self.ret_1d])

        return X_train, Y_train, X_test, Y_test

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