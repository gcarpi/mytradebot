import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, stoploss_from_open
from pandas import DataFrame, Series
from datetime import datetime
from typing import Dict, List
from skopt.space import Dimension, Integer, Real

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class ClucHAnix(IStrategy):

    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params[
                'roi_p5'] + params['roi_p6']
            roi_table[params['roi_t6']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + \
                                          params['roi_p5']
            roi_table[params['roi_t6'] + params['roi_t5']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + \
                                                             params['roi_p4']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4']] = params['roi_p1'] + params['roi_p2'] + \
                                                                                params['roi_p3']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3']] = params['roi_p1'] + \
                                                                                                   params['roi_p2']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2']] = \
            params['roi_p1']
            roi_table[
                params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params[
                    'roi_t1']] = 0

            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            return [
                Integer(1, 15, name='roi_t6'),
                Integer(1, 45, name='roi_t5'),
                Integer(1, 90, name='roi_t4'),
                Integer(45, 120, name='roi_t3'),
                Integer(45, 180, name='roi_t2'),
                Integer(90, 300, name='roi_t1'),

                Real(0.005, 0.10, name='roi_p6'),
                Real(0.005, 0.07, name='roi_p5'),
                Real(0.005, 0.05, name='roi_p4'),
                Real(0.005, 0.025, name='roi_p3'),
                Real(0.005, 0.01, name='roi_p2'),
                Real(0.003, 0.007, name='roi_p1'),
            ]

    # Buy params
    buy_params = {
        "bbdelta_close": 0.0523,
        "bbdelta_tail": 0.17595,
        "close_bblower": 0.00446,
        "closedelta_close": 0.00728,
        "rocr_1h": 0.45662,
    }

    # Sell params
    sell_params = {
        "pHSL": -0.194,
        "pPF_1": 0.012,
        "pPF_2": 0.058,
        "pSL_1": 0.02,
        "pSL_2": 0.064,
        "sell_bbmiddle_close": 2.18407,
        "sell_fisher": 0.50127,
    }

    # ROI table
    minimal_roi = {
        "0": 0.14642,
        "2": 0.10886,
        "19": 0.05173,
        "58": 0.02162,
        "112": 0.01432,
        "187": 0.00502,
        "349": 0
    }

    # Stoploss
    stoploss = -0.99

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.193
    trailing_stop_positive_offset = 0.198
    trailing_only_offset_is_reached = True
    
    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    # Candles params
    process_only_new_candles = True
    startup_candle_count = 168

    # Hyperopt buy params
    bbdelta_close = DecimalParameter(0.01000, 0.09999, default=0.01965, decimals=5, space='buy', load=True)
    bbdelta_tail = DecimalParameter(0.10000, 0.99999, default=0.95089, decimals=5, space='buy', load=True)
    close_bblower = DecimalParameter(0.00100, 0.00999, default=0.00799, decimals=5, space='buy', load=True)
    closedelta_close = DecimalParameter(0.00100, 0.00999, default=0.00556, decimals=5, space='buy', load=True)
    rocr_1h = DecimalParameter(0.10000, 0.99999, default=0.54904, decimals=5, space='buy', load=True)

    # Hyperopt sell params
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)
    sell_fisher = DecimalParameter(0.10000, 0.90000, default=0.38414, decimals=5, space='sell', load=True)
    sell_bbmiddle_close = DecimalParameter(1.00000, 9.00000, default=1.07634, decimals=5, space='sell', load=True)

    # Protections
    protection_params = {
        "low_profit_lookback": 48,
        "low_profit_min_req": 0.04,
        "low_profit_stop_duration": 14,
        "cooldown_lookback": 2,
        "stoploss_lookback": 72,
        "stoploss_stop_duration": 20,
    }

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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid
        
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        
        inf_tf = '1h'
        
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
     
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr_1h'])
            ) &
            ((      
                    (dataframe['lower'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta_close'])) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta_close'])) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta_tail'])) &
                    (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
            ) |
            (       
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < params['close_bblower'] * dataframe['bb_lowerband']) 
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['fisher'] > params['sell_fisher']) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * params['sell_bbmiddle_close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe