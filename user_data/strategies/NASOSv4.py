# for live trailing_stop = False and use_custom_stoploss = True
# for backtest trailing_stop = True and use_custom_stoploss = False
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt

# Buy params
buy_params = {
    "base_candles_buy": 8,
    "lookback_candles": 8,
    "volume_shift": 60,
    "ewo_high": 1.800,
    "low_offset": 0.984,
    "profit_threshold": 0.900,
    "rsi_buy": 70,
    "rsi_fast_buy": 25
}

# Sell params
sell_params = {
    "base_candles_sell": 4,
    "high_offset": 1.084,
    "high_offset_2": 1.401,
    "rsi_sell": 50,
}

# Elliot Waves Oscilator
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()

    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100

    return emadif

class NASOSv4(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table
    minimal_roi = {
        "0": 10
    }

    # Stoploss
    stoploss = -0.25
    use_custom_stoploss = False

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force (only works on Binance)
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Candle params
    process_only_new_candles = False
    startup_candle_count = 200

    # Plot config
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    # Slippage
    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    # Protections
    protection_params = {
      "cooldown_lookback": 12,
      "low_profit_lookback": 6,
      "low_profit_min_req": -0.04,
      "low_profit_stop_duration": 64
    }

    # Hyperopt params
    base_candles_buy = IntParameter(2, 20, default=buy_params['base_candles_buy'], space='buy', optimize=True)
    base_candles_sell = IntParameter(2, 25, default=sell_params['base_candles_sell'], space='sell', optimize=True)

    low_offset = DecimalParameter(0.1, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)

    high_offset = DecimalParameter(0.1, 2.0, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.1, 2.0, default=sell_params['high_offset_2'], space='sell', optimize=True)

    lookback_candles = IntParameter(1, 24, default=buy_params['lookback_candles'], space='buy', optimize=True)
    profit_threshold = DecimalParameter(1.0, 1.03, default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_high = DecimalParameter(1.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    rsi_buy = IntParameter(1, 100, default=buy_params['rsi_buy'], space='buy', optimize=True)
    rsi_fast_buy = IntParameter(1, 100, default=buy_params['rsi_fast_buy'], space='buy', optimize=True)
    rsi_sell = IntParameter(1, 100, default=sell_params['rsi_sell'], space='sell', optimize=True)

    volume_shift = IntParameter(0, 100, default=buy_params['volume_shift'], space="buy", optimize=True)

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)
    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=True)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2, optimize=True)

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

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        dataframe['EWO'] = EWO(dataframe, 50, 200)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)

        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        check_volume = (
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(self.volume_shift.value) * 0.4) &                          # Try to exclude pumping
            (dataframe['close_1h'].rolling(self.lookback_candles.value).max() > (dataframe['close'] * self.profit_threshold.value)) &       # Try to exclude pumping
            (dataframe['volume'] < (dataframe['volume'].shift() * self.volume_shift.value)) &                                               # Don't buy if someone drop the market
            (dataframe['volume'] > 0)                                                                                                       # Make sure Volume is not 0
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                check_volume
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   
                (dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] > self.rsi_sell.value) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
