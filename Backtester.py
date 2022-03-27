import numpy as np
from pylab import mpl, plt
from decimal import Decimal, localcontext, ROUND_DOWN

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
                 ftc=0.0, ptc=0.002, max_vol_precision_decimals_official=8, start_date=None, end_date=None, verbose=True):

        # see price and volume precision
        #https://support.kraken.com/hc/en-us/articles/4521313131540

        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash_balance = cash_balance #old name: initial_amount
        self.current_cash_balance = cash_balance #old name: amount
        self.ftc = ftc #fixed transaction costs per trade
        self.ptc = ptc #proportional transaction costs per trade
        self.units_of_asset_held = 0 #old name: units
        #self.position = 'neutral' #+1 if long, -1 if short (useless as you have units_of_asset_held)
        self.n_of_trades_executed = 0
        self.verbose = verbose
        self.data = data

        ## variable names
        self.min_order_size = min_order_size
        self.max_vol_precision_decimals = max_vol_precision_decimals #Expresses the number of decimals.
        # You can't give a volume to buy / sell with higher precision than this
        self.max_vol_precision_decimals_official = max_vol_precision_decimals_official #official value on exchange website
        self.price_col = col_names_dict['price_col']
        self.prediction_col = col_names_dict['buy_sell_signal_col']
        self.net_wealth_col = col_names_dict['net_wealth_col']

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
        self.data[self.net_wealth_col] = self.data[self.net_wealth_col] / self.data[self.net_wealth_col].iloc[0]
        self.data[[self.price_col, self.net_wealth_col]].plot(ax=axes[0])

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

    def print_current_cash_balance_and_asset_units(self, bar):
        ''' Print out current cash balance info.'''

        date, price = self.get_date_price(bar)
        print(str(date), '| Cash balance = ', str(np.round(self.current_cash_balance, 2)),
              '| Units of asset held = ', self.units_of_asset_held)
        #print(f'{date} | current balance {self.current_cash_balance:.2f}')


    def calc_net_wealth(self, bar):
        ''' Print out current cash balance info.    '''

        date, price = self.get_date_price(bar)
        net_wealth = self.units_of_asset_held * price + self.current_cash_balance
        self.data[self.net_wealth_col].iloc[bar] = net_wealth
        print(f'{date} | Net wealth {net_wealth:.2f}')

    def place_buy_market_order(self, bar, units=None, cash_amount=None):
        ''' Place a buy order.    '''

        date, price = self.get_date_price(bar)
        if units is None:
            units = truncate(cash_amount / price, self.max_vol_precision_decimals)

        cash_balance_after_trade = self.current_cash_balance - (units*price * (1 + self.ptc) + self.ftc)

        while cash_balance_after_trade <= 0:
            units = truncate(units - self.min_order_size, self.max_vol_precision_decimals)
            cash_balance_after_trade = self.current_cash_balance - (units*price * (1 + self.ptc) + self.ftc)

        if units >= self.min_order_size:
            #TODO: careful, there is the risk you won't be able to close long positions
            self.current_cash_balance = cash_balance_after_trade
            self.units_of_asset_held += units
            self.n_of_trades_executed += 1
            if self.verbose:
                print(f'{date} | buying {units} units at {price:.2f}')


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
            units = truncate(cash_amount / price, self.max_vol_precision_decimals)

        cash_balance_after_trade = self.current_cash_balance + (units*price * (1 - self.ptc) - self.ftc)

        if np.abs(units) >= self.min_order_size:
            self.current_cash_balance = cash_balance_after_trade
            self.units_of_asset_held -= units
            self.n_of_trades_executed += 1
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
        print('Final balance [$] {:.2f}'.format(self.current_cash_balance))
        perf = ((self.current_cash_balance - self.initial_cash_balance) /
                self.initial_cash_balance * 100)
        print('Net Performance [%] {:.2f}'.format(perf))
        print('Trades Executed [#] {:.2f}'.format(self.n_of_trades_executed))
        print('=' * 55)


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


    def bt_long_short_signal(self):

        self.data[self.net_wealth_col] = np.nan

        for bar in range(self.data.shape[0]):
            prediction = self.data[self.prediction_col].iloc[bar]
            if prediction == 1:
                #in this case you want to go long only if current position is neutral or short
                if self.units_of_asset_held <= 0: #self.position in ['neutral', 'short']:
                    self.go_long(bar, cash_amount='all')
                    #self.units_of_asset_held <= 0: #self.position = 'long'
            elif prediction == -1:
                if self.units_of_asset_held >= 0: #self.position in ['neutral', 'long']:
                    self.go_short(bar, cash_amount='all')
                    #self.position = 'short'
            elif prediction == 0:
                print('!!! Error, unclear indication!!!!')
                break

            self.print_current_cash_balance_and_asset_units(bar)
            self.calc_net_wealth(bar)

        self.close_out(bar)

#if __name__ == '__main__':
#    bb = Backtester('AAPL.O', '2010-1-1', '2019-12-31', 10000)
#    print(bb.data.info())
#    print(bb.data.tail())
#    bar = 0
#    print(bb.get_date_price(bar))
#    bb.plot_data()


#    plt.show()


