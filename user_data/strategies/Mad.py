import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
from functools import reduce

# SSL Channels
def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class Mad(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
      "0": 0.034,
      "19": 0.022,
      "45": 0.012,
      "139": 0
    }

    # Disabled stoploss
    stoploss = -0.99

    # Timeframe
    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.173
    trailing_stop_positive_offset = 0.239

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

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

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
            return 0.01
        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 10)
        informative_1h['ssl_down'] = ssl_down_1h
        informative_1h['ssl_up'] = ssl_up_1h

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=20).mean()

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Conditions
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        check_volume = (
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * 0.4) &   # Try to exclude pumping
            (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &                         # Don't buy if someone drop the market
            (dataframe['volume'] > 0)                                                           # Make sure Volume is not 0
        )
        
        buy_ema_200 = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h']) &
            (dataframe['close'] < dataframe['ema_slow']) &
            (dataframe['close'] < 0.99 * dataframe['bb_lowerband']) &
            check_volume
        )

        buy_ema_slow = (
            (dataframe['close'] < dataframe['ema_slow']) &
            (dataframe['close'] < 0.975 * dataframe['bb_lowerband']) &
            (dataframe['rsi_1h'] < 36) &
            check_volume
        )

        buy_ema_200_ema_26 = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h']) &
            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
            (dataframe['close'] < (dataframe['bb_lowerband'])) &
            check_volume
        )

        buy_ema_26_ema_12 = (
            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
            (dataframe['close'] < (dataframe['bb_lowerband'])) &
            check_volume
        )

        buy_ema_50 = (
            (dataframe['close'] < dataframe['sma_5']) &
            (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
            (dataframe['ema_slow'] > dataframe['ema_200']) &
            (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
            (dataframe['rsi'] < dataframe['rsi_1h'] - 50) &
            check_volume
        )

        # Append conditions
        conditions.append(buy_ema_200)
        dataframe.loc[buy_ema_200, 'buy_tag'] += 'buy_ema_200 '

        conditions.append(buy_ema_slow)
        dataframe.loc[buy_ema_slow, 'buy_tag'] += 'buy_ema_slow '

        conditions.append(buy_ema_200_ema_26)
        dataframe.loc[buy_ema_200_ema_26, 'buy_tag'] += 'buy_ema_200_ema_26 '

        conditions.append(buy_ema_26_ema_12)
        dataframe.loc[buy_ema_26_ema_12, 'buy_tag'] += 'buy_ema_26_ema_12 '

        conditions.append(buy_ema_50)
        dataframe.loc[buy_ema_50, 'buy_tag'] += 'buy_ema_50 '

        # Set conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy' ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband'] * 1) &                 # Don't be gready, sell fast
                (dataframe['volume'] > 0)                                              # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe
