import numpy as np
from pylab import mpl, plt
from decimal import Decimal, localcontext, ROUND_DOWN
import seaborn as sns
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

''' Gianluca's notes...

I want to track:
    - cash balance
    - amount of asset held

Need to consider:
    - minimum amount of asset that can be bought/sold 

'''


def truncate(number, places):
    if not isinstance(places, int):
        raise ValueError("Decimal places must be an integer.")
    if places < 1:
        raise ValueError("Decimal places must be at least 1.")
    # If you want to truncate to 0 decimal places, just do int(number).

    with localcontext() as context:
        context.rounding = ROUND_DOWN
        exponent = Decimal(str(10 ** - places))
        return float(Decimal(str(number)).quantize(exponent).to_eng_string())


class Backtester(object):

    #TODO:
    #   - short costs
    #   - do not use entire initial_cash_balance so that you can cover when short positions go bad

    def __init__(self, cash_balance, data, col_names_dict, min_order_size=0.0001, max_vol_precision_decimals=4,
                 ftc=0.0, ptc=0.002, spi_d=0.0001, max_vol_precision_decimals_official=8, rebalance_threshold=None,
                 start_date=None, end_date=None,
                 verbose=True):

        # see price and volume precision
        #https://support.kraken.com/hc/en-us/articles/4521313131540

        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash_balance = cash_balance #old name: initial_amount
        self.current_cash_balance = cash_balance #old name: amount
        self.ftc = ftc #fixed transaction costs per trade
        self.ptc = ptc #proportional transaction costs per trade
        self.spi_d = spi_d #daily borrowing interest for short position
        self.units_of_asset_held = 0 #old name: units
        self.n_of_trades_executed = 0
        self.verbose = verbose
        self.data = data
        self.rebalance_threshold = rebalance_threshold #Threshold to rebalance portfolio when position is short
        if rebalance_threshold is None:
            self.rebalance_threshold = cash_balance * 0.05


        ### variable names ###
        self.min_order_size = min_order_size
        self.max_vol_precision_decimals = max_vol_precision_decimals #Expresses the number of decimals.
        # You can't give a volume to buy / sell with higher precision than this
        self.max_vol_precision_decimals_official = max_vol_precision_decimals_official #official value on exchange website
        self.price_col = col_names_dict['price_col']
        self.prediction_col = col_names_dict['buy_sell_signal_col']
        self.ptf_value_col = col_names_dict['ptf_value_col']

        if max_vol_precision_decimals > max_vol_precision_decimals_official:
            raise ValueError('max_vol_precision > max_vol_precision_official')

    def plot_data(self, cols=None):
        ''' Plots the closing prices for symbol.'''
        if cols is None:
            cols = self.price_col
        self.data[cols].plot(figsize=(10, 6), title=self.price_col)

    def plot_strategy_vs_asset(self):
        ''' Plots the strategy returns vs the asset returns.'''
        fig, axes = plt.subplots(nrows=2, figsize=(10, 6), gridspec_kw={'height_ratios': [5, 1]})

        self.data[self.price_col] = self.data[self.price_col] / self.data[self.price_col].iloc[0]
        self.data[self.ptf_value_col] = self.data[self.ptf_value_col] / self.data[self.ptf_value_col].iloc[0]
        self.data[[self.price_col, self.ptf_value_col]].plot(ax=axes[0])

        self.data[self.prediction_col].plot(ax=axes[1])

        fig.suptitle(self.price_col)
        plt.tight_layout()

    def get_date_price(self, bar):
        ''' Return date and price for bar.
        bar: integer
        '''
        date = str(self.data.index[bar])[:10]
        price = self.data[self.price_col].iloc[bar]
        return date, price

#    def print_current_cash_balance_and_asset_units(self, bar):
#        ''' Print out current cash balance info.'''

#        date, price = self.get_date_price(bar)
#        print(str(date), '| Cash balance = ', str(np.round(self.current_cash_balance, 2)),
#              '| Units of asset held = ', self.units_of_asset_held)
        #print(f'{date} | current balance {self.current_cash_balance:.2f}')


    def calc_ptf_value(self, bar):
        ''' Print out current cash balance info.    '''

        date, price = self.get_date_price(bar)
        ptf_value = self.units_of_asset_held * price + self.current_cash_balance
        self.data[self.ptf_value_col].iloc[bar] = ptf_value

        if self.verbose:
            print(str(date), '| Cash balance = ', str(np.round(self.current_cash_balance, 2)),
                  '| Units of asset held = ', self.units_of_asset_held)
            print(f'{date} | Ptf value {ptf_value:.2f}')

    def place_buy_market_order(self, bar, units=None, cash_amount=None):
        ''' Place a buy order.    '''

        date, price = self.get_date_price(bar)
        if units is None:
            # TODO: need to take into account transaction costs when calculating units
            units = cash_amount / price
        self.current_cash_balance -= (units * price) * (1 + self.ptc) + self.ftc
        self.units_of_asset_held += units
        self.n_of_trades_executed += 1
        if self.verbose:
            print(f'{date} | buying {units} units at {price:.2f}')

#        else:
#            print('Insufficient cash to purchase the asset. Cash balance = ', self.current_cash_balance)


    def OLD__place_buy_market_order(self, bar, units=None, cash_amount=None):
        ''' Place a buy order.    '''

        date, price = self.get_date_price(bar)
        if units is None:
            # TODO: need to take into account transaction costs when calculating units
            units = int(cash_amount / price)
        self.current_cash_balance -= (units * price) * (1 + self.ptc) + self.ftc
        self.units_of_asset_held += units
        self.n_of_trades_executed += 1
        if self.verbose:
            print(f'{date} | buying {units} units at {price:.2f}')
#        self.print_current_cash_balance_and_asset_units(bar)
#        self.calc_net_wealth(bar)

    def place_sell_market_order(self, bar, units=None, cash_amount=None):
        ''' Place a sell order.    '''

        date, price = self.get_date_price(bar)
        if units is None:
            units = cash_amount / price
        self.current_cash_balance += (units * price) * (1 - self.ptc) - self.ftc
        self.units_of_asset_held -= units
        self.n_of_trades_executed += 1
        if self.verbose:
            if self.verbose:
                print(f'{date} | selling {units} units at {price:.2f}')


    def OLD__place_sell_market_order(self, bar, units=None, cash_amount=None):
        ''' Place a sell order.    '''

        date, price = self.get_date_price(bar)
        if units is None:
            units = int(cash_amount / price)
        self.current_cash_balance += (units * price) * (1 - self.ptc) - self.ftc
        self.units_of_asset_held -= units
        self.n_of_trades_executed += 1
        if self.verbose:
            print(f'{date} | selling {units} units at {price:.2f}')


    def close_out(self, bar):
        ''' Closing out a long or short position.    '''

        date, price = self.get_date_price(bar)
        self.current_cash_balance += self.units_of_asset_held * price

        self.units_of_asset_held = 0
        self.n_of_trades_executed += 1
        if self.verbose:
            print(f'{date} | inventory {self.units_of_asset_held} units at {price:.2f}')
            print('=' * 55)


    def get_strategy_stats(self, strategy_str):
        digits = 4
        plt.figure()

        v1 = self.data[self.price_col].pct_change()
        v2 = self.data[self.ptf_value_col].pct_change()

        sig_p = v1.std()
        sig_s = v2.std()

        avg_ret_p = v1.mean()
        avg_ret_s = v2.mean()

        sh_rat_p = avg_ret_p / sig_p
        sh_rat_s = avg_ret_s / sig_s

        print('--- Performances ' + strategy_str + ' ---')
        print('ASSET stats | Avg ret = ' + str(np.round(avg_ret_p, digits)) +
              ' | Std of ret = ' + str(np.round(sig_p, digits)) + ' | Sharpe ratio = ' + str(np.round(sh_rat_p, digits)))
        print('STRATEGY stats | Avg ret = ' + str(np.round(avg_ret_s, digits)) +
              ' | Std = ' + str(np.round(sig_s, digits)) + ' | Sharpe ratio = ' + str(np.round(sh_rat_s, digits)))

        print('Final balance [$] {:.2f}'.format(self.current_cash_balance))
        perf_s = ((self.current_cash_balance - self.initial_cash_balance) /
                self.initial_cash_balance * 100)

        perf_a = ((self.data[self.price_col].iloc[-1] - self.data[self.price_col].iloc[0]) /
                self.data[self.price_col].iloc[-1] * 100)

        print('Strategy Net Performance [%] {:.2f}'.format(perf_s))
        print('Asset Net Performance [%] {:.2f}'.format(perf_a))

        print('Trades Executed {:.2f}'.format(self.n_of_trades_executed))
        print('=' * 55)

        sns.set(style="darkgrid")
        sns.histplot(data=v1, color="skyblue", label="Asset returns. sigma = " + str(np.round(sig_p, digits)))#, x="sepal_length", color="skyblue", label="Sepal Length", kde=True)
        sns.histplot(data=v2, color="red", label="Strategy returns. sigma = " + str(np.round(sig_s, digits)))#, x="sepal_width", color="red", label="Sepal Width", kde=True)
        plt.legend()


class BacktestLongShort(Backtester):

    def go_long(self, bar, units=None, cash_amount=None):
        #first close short positions (if any are open)
        #Note you need to buy the same number of units of your short position
        if self.units_of_asset_held < 0: #self.position == 'short':
            self.place_buy_market_order(bar, units=-self.units_of_asset_held)
        # if units is given
        if units:
            self.place_buy_market_order(bar, units=units)
        #if amoount is given
        elif cash_amount:
            if cash_amount == 'all':
                cash_amount = self.current_cash_balance
            self.place_buy_market_order(bar, cash_amount=cash_amount)
            #TODO: not checked you have cash for this

    def go_short(self, bar, units=None, cash_amount=None):
        # first close long positions (if any are open)
        if self.units_of_asset_held > 0: #self.position == 'long':
            self.place_sell_market_order(bar, units=self.units_of_asset_held)
        if units:
            self.place_sell_market_order(bar, units=units)
        elif cash_amount:
            if cash_amount == 'all':
                cash_amount = self.current_cash_balance
            self.place_sell_market_order(bar, cash_amount=cash_amount)


    def bt_long_short_signal(self, is_debug=False):

        self.data[self.ptf_value_col] = np.nan

        if is_debug:
            self.data['price_change'] = np.nan
            self.data['units'] = np.nan
            self.data['cash_balance'] = np.nan

        for bar in range(self.data.shape[0]):

            asset_units = self.units_of_asset_held

            date, price = self.get_date_price(bar)
            price_change = 0

            if bar > 0:
                prev_date, prev_price = self.get_date_price(bar-1)
                price_change = price - prev_price

            #check if short positions are open from previous time step and charge fees
            #TODO: this is not how short fees are charged.
            if asset_units < 0:
                self.current_cash_balance -= np.abs(asset_units) * price * self.spi_d

            pnl = price_change * asset_units

            prediction = self.data[self.prediction_col].iloc[bar]
            if prediction == 1 and asset_units <= 0: #if 'neutral' or 'short':
                #in this case you want to go long only if current position is neutral or short
                    self.go_long(bar, cash_amount='all')
            elif prediction == -1:
                if asset_units >= 0: #if 'neutral' or 'long':
                    self.go_short(bar, cash_amount='all')
                elif price_change > self.rebalance_threshold: #Rebalance position
                    # In this case you were short the asset and the asset increased in value,
                    # so you have to buy asset to close part of the short position
                    self.place_buy_market_order(bar, cash_amount=-2*pnl)
                elif price_change < -self.rebalance_threshold:
                    # Sell asset to increase the short position
                    self.place_sell_market_order(bar, cash_amount=2*pnl)
            elif prediction == 0:
                raise ValueError("!!!ERROR: Prediction value = 0!!!")
                break

            #self.print_current_cash_balance_and_asset_units(bar)
            self.calc_ptf_value(bar)

            if is_debug:
                self.data['price_change'].iloc[bar] = price_change
                self.data['units'].iloc[bar] = self.units_of_asset_held
                self.data['cash_balance'].iloc[bar] = self.current_cash_balance

        if is_debug:
            _df = self.data.copy()
            _df.reset_index(drop=True, inplace=True)
            _df.to_excel('C:/Users/Gianluca/Desktop/BT_debug.xlsx')

        self.close_out(bar)

        self.get_strategy_stats('long/short')


    def bt_long_only_signal(self):

        self.data[self.ptf_value_col] = np.nan
        for bar in range(self.data.shape[0]):

            asset_units = self.units_of_asset_held

            prediction = self.data[self.prediction_col].iloc[bar]
            if prediction == 1 and asset_units <= 0: #if 'neutral' or 'short':
                self.go_long(bar, cash_amount='all')
            elif prediction == -1:
                if asset_units > 0:  # self.position == 'long':
                    self.place_sell_market_order(bar, units=asset_units)
            elif prediction == 0:
                raise ValueError("!!!ERROR: Prediction value = 0!!!")
                break

            #self.print_current_cash_balance_and_asset_units(bar)
            self.calc_ptf_value(bar)

        self.close_out(bar)
        self.get_strategy_stats('long only')

#if __name__ == '__main__':
#    bb = Backtester('AAPL.O', '2010-1-1', '2019-12-31', 10000)
#    print(bb.data.info())
#    print(bb.data.tail())
#    bar = 0
#    print(bb.get_date_price(bar))
#    bb.plot_data()


#    plt.show()


