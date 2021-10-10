import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from technical.indicators import ichimoku

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from functools import reduce

import logging
logger = logging.getLogger(__name__)

class Ichi(IStrategy):
    INTERFACE_VERSION = 2

    # ROI
    minimal_roi = {
        "0": 0.10,
    }
    
    # Timeframe
    timeframe = '5m'

    # Disabled stoploss
    stoploss = -0.99

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Protections
    protection_params = {
        "low_profit_lookback": 48,
        "low_profit_min_req": 0.04,
        "low_profit_stop_duration": 14,

        "cooldown_lookback": 2,
        "stoploss_lookback": 72,
        "stoploss_stop_duration": 20,
    }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)
    
    low_profit_optimize = False
    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2, optimize=low_profit_optimize)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })

        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.low_profit_lookback.value,
            "trade_limit": 1,
            "stop_duration": int(self.low_profit_stop_duration.value),
            "required_profit": self.low_profit_min_req.value
        })

        return prot

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
            return 0.01
        return 0.99

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Ichimoku
        ichi=ichimoku(dataframe)
        dataframe['tenkan']=ichi['tenkan_sen']
        dataframe['kijun']=ichi['kijun_sen']
        dataframe['senkou_a']=ichi['senkou_span_a']
        dataframe['senkou_b']=ichi['senkou_span_b']
        dataframe['cloud_green']=ichi['cloud_green']
        dataframe['cloud_red']=ichi['cloud_red']

        # EMA
        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
             (
                (dataframe['close'] > dataframe['senkou_a']) &
                (dataframe['close'] > dataframe['senkou_b']) &
                (dataframe['cloud_green'] == True) &
                (dataframe['tenkan'] > dataframe['kijun']) &
                (dataframe['tenkan'].shift(1) < dataframe['kijun'].shift(1)) &
                (dataframe['close'].shift(-24) > dataframe['close'].shift(24))
             ) &
             (
                (dataframe['ema21'] > dataframe['ema55']) &
                (dataframe['ema13'] > dataframe['ema21']) &
                (dataframe['ema8'] > dataframe['ema13'])
             )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'].shift(-24) <= dataframe['close'].shift(24))
            )
            ,
            'sell'
        ] = 1
        return dataframe