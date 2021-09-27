# Jupyter Strategy
# docker-compose -f docker-compose.yml run --rm freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces roi stoploss --strategy Jupyter --config user_data/config.json --config user_data/config-dev.json --epochs 100

from freqtrade.strategy.hyper import IntParameter, DecimalParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta

class Jupyter(IStrategy):

    # ROI table:
    minimal_roi = {
        "0": 0.457,
        "101": 0.092,
        "223": 0.072,
        "816": 0
    }

    # Stoploss:
    stoploss = -0.192

    # Buy hypers
    timeframe = '30m'

    # buy params
    buy_mojo_ma_timeframe = IntParameter(2, 100, default=7, space='buy')
    buy_fast_ma_timeframe = IntParameter(2, 100, default=14, space='buy')
    buy_slow_ma_timeframe = IntParameter(2, 100, default=28, space='buy')
    buy_div_max = DecimalParameter(
        0, 2, decimals=4, default=2.25446, space='buy')
    buy_div_min = DecimalParameter(
        0, 2, decimals=4, default=0.29497, space='buy')

    sell_mojo_ma_timeframe = IntParameter(2, 100, default=7, space='sell')
    sell_fast_ma_timeframe = IntParameter(2, 100, default=14, space='sell')
    sell_slow_ma_timeframe = IntParameter(2, 100, default=28, space='sell')
    sell_div_max = DecimalParameter(
        0, 2, decimals=4, default=1.54593, space='sell')
    sell_div_min = DecimalParameter(
        0, 2, decimals=4, default=2.81436, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMA - ex Moving Average
        dataframe['buy-mojoMA'] = ta.SMA(dataframe,
                                         timeperiod=self.buy_mojo_ma_timeframe.value)
        dataframe['buy-fastMA'] = ta.SMA(dataframe,
                                         timeperiod=self.buy_fast_ma_timeframe.value)
        dataframe['buy-slowMA'] = ta.SMA(dataframe,
                                         timeperiod=self.buy_slow_ma_timeframe.value)
        dataframe['sell-mojoMA'] = ta.SMA(dataframe,
                                          timeperiod=self.sell_mojo_ma_timeframe.value)
        dataframe['sell-fastMA'] = ta.SMA(dataframe,
                                          timeperiod=self.sell_fast_ma_timeframe.value)
        dataframe['sell-slowMA'] = ta.SMA(dataframe,
                                          timeperiod=self.sell_slow_ma_timeframe.value)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['buy-mojoMA'].div(dataframe['buy-fastMA'])
                    > self.buy_div_min.value) &
                (dataframe['buy-mojoMA'].div(dataframe['buy-fastMA'])
                    < self.buy_div_max.value) &
                (dataframe['buy-fastMA'].div(dataframe['buy-slowMA'])
                    > self.buy_div_min.value) &
                (dataframe['buy-fastMA'].div(dataframe['buy-slowMA'])
                    < self.buy_div_max.value)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['sell-fastMA'].div(dataframe['sell-mojoMA'])
                    > self.sell_div_min.value) &
                (dataframe['sell-fastMA'].div(dataframe['sell-mojoMA'])
                    < self.sell_div_max.value) &
                (dataframe['sell-slowMA'].div(dataframe['sell-fastMA'])
                    > self.sell_div_min.value) &
                (dataframe['sell-slowMA'].div(dataframe['sell-fastMA'])
                    < self.sell_div_max.value)
            ),
            'sell'] = 1
        return dataframe
