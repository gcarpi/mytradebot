# DefaultStra Strategy
# AVG/MID profit in USDT
# docker-compose -f docker-compose.yml run --rm freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell roi stoploss --strategy DefaultStra --config user_data/config.json --config user_data/config.dev.json --epochs 100

import logging
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter

from numpy.lib import math
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import pandas as pd
import ta
from ta.utils import dropna
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np

class DefaultStra(IStrategy):
    # Buy params:
    buy_params = {
        "buy_cat": "<R",
        "buy_real": 0.6344,
    }

    # Sell params:
    sell_params = {
        "sell_cat": "=R",
        "sell_real": 0.2861,
    }

    # ROI params:
    minimal_roi = {
      "0": 0.28900000000000003,
      "129": 0.062,
      "450": 0.036,
      "570": 0
    }

    # Stoploss:
    stoploss = -0.256

    # Buy hypers
    buy_real = DecimalParameter(
        0.001, 0.999, decimals=4, default=0.11908, space='buy')
    buy_cat = CategoricalParameter(
        [">R", "=R", "<R"], default='<R', space='buy')
    sell_real = DecimalParameter(
        0.001, 0.999, decimals=4, default=0.59608, space='sell')
    sell_cat = CategoricalParameter(
        [">R", "=R", "<R"], default='>R', space='sell')

    timeframe = '30m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['trend_ichimoku_base'] = ta.trend.ichimoku_base_line(
            dataframe['high'],
            dataframe['low'],
            window1=9,
            window2=26,
            visual=False,
            fillna=False
        )
        KST = ta.trend.KSTIndicator(
            close=dataframe['close'],
            roc1=10,
            roc2=15,
            roc3=20,
            roc4=30,
            window1=10,
            window2=10,
            window3=10,
            window4=15,
            nsig=9,
            fillna=False
        )

        dataframe['trend_kst_diff'] = KST.kst_diff()

        # Normalization
        tib = dataframe['trend_ichimoku_base']
        dataframe['trend_ichimoku_base'] = (
            tib-tib.min())/(tib.max()-tib.min())
        tkd = dataframe['trend_kst_diff']
        dataframe['trend_kst_diff'] = (tkd-tkd.min())/(tkd.max()-tkd.min())
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        IND = 'trend_ichimoku_base'
        REAL = self.buy_real.value
        OPR = self.buy_cat.value
        DFIND = dataframe[IND]

        if OPR == ">R":
            conditions.append(DFIND > REAL)
        elif OPR == "=R":
            conditions.append(np.isclose(DFIND, REAL))
        elif OPR == "<R":
            conditions.append(DFIND < REAL)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        IND = 'trend_kst_diff'
        REAL = self.sell_real.value
        OPR = self.sell_cat.value
        DFIND = dataframe[IND]

        if OPR == ">R":
            conditions.append(DFIND > REAL)
        elif OPR == "=R":
            conditions.append(np.isclose(DFIND, REAL))
        elif OPR == "<R":
            conditions.append(DFIND < REAL)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
