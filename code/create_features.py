import numpy as np
import pandas as pd


# TREND INDICATORS
def add_MA_indicators(group, simple_ndays=200, exp_ndays=26):
    """
    Compute exponential moving average.
    """
    group = group.copy()
    group['EMA'] = group['Close'].ewm(span=exp_ndays, min_periods=exp_ndays-1).mean()
    group = group.dropna()

    return group


# VOLUME INDICATORS
def add_OBV_indicator(group):
    """
    Compute On-Balance Volume.
    """
    group = group.copy()
    group["OBV"] = (np.sign(group["Close"].diff()) * group["Volume"]).fillna(0).cumsum()
    group = group.tail(-1)
    return group


def add_AD_indicator(group):
    """
    Compute Accumulation/Distribution.
    """
    group = group.copy()
    denominator = group['High'] - group['Low']
    group["MFM"] = ((group['Close'] - group['Low'])
                    - (group['High'] - group['Close'])) * group['Volume'] / denominator
    group["AD"] = group["MFM"].cumsum()

    group = group.replace([np.inf, -np.inf], 0)  # Might be better if dropped
    group = group.dropna()
    return group


def add_previous_day_volume_indicator(group):
    """
    Compute Previous Day Volume.
    """
    group = group.copy()
    group['Previous_Volume'] = group['Volume'].shift(1)
    group = group.dropna()
    return group


# VOLATILITY INDICATORS
def add_ATR_indicator(group, ndays=14):
    """
    Compute Average True Range.
    """
    group = group.copy()
    high_low = group['High'] - group['Low']
    high_close = np.abs(group['High'] - group['Close'].shift())
    low_close = np.abs(group['Low'] - group['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    group['ATR'] =  true_range.rolling(ndays).mean()
    group = group.dropna()

    return group

# MOMENTUM INDICATORS
def gain(x):
    return ((x > 0) * x).sum()

def loss(x):
    return ((x < 0) * x).sum()

def add_MFI_indicator(group, ndays=14):
    """
    Compute Money Flow Index.
    """
    group = group.copy()
    typical_price = (group['High'] + group['Low'] + group['Close']) / 3
    money_flow = typical_price * group['Volume']
    money_flow_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    money_flow_signed = money_flow * money_flow_sign
    pos_money_flow = money_flow_signed.rolling(ndays).apply(gain, raw=True)
    neg_money_flow = money_flow_signed.rolling(ndays).apply(loss, raw=True)
    group["MFI"] = (100 - (100 / (1 + (pos_money_flow / abs(neg_money_flow))))).to_numpy()
    group = group.dropna()

    return group


# https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
def add_RSI_indicator(group, ndays=14):
    """
    Compute Relative Strength Index.
    """
    group = group.copy()
    close_diff = group['Close'].diff()
    gains = close_diff.clip(lower=0)
    losses = -1 * close_diff.clip(upper=0)
    avg_gain = gains.ewm(com=ndays-1, min_periods=ndays).mean()
    avg_loss = losses.ewm(com=ndays-1, min_periods=ndays).mean()
    rs = avg_gain / avg_loss
    group['RSI'] = 100 - (100/(1 + rs))
    group = group.dropna()

    return group


# https://tcoil.info/compute-macd-indicator-for-stocks-with-python/
def add_MACD_indicators(group):
    """
    Compute Moving Averge Convergence Divergence.
    """
    group = group.copy()
    for short_horizon, long_horizon in [(12, 26)]:
        # Short EMA of the closing price
        fast_ema = group['Close'].ewm(span=short_horizon,
                                   min_periods=short_horizon,
                                   adjust=False).mean()
        # Long EMA of the closing price
        slow_ema = group['Close'].ewm(span=long_horizon,
                                   min_periods=long_horizon,
                                   adjust=False).mean()
        macd_line = f'MACD_{short_horizon}_{long_horizon}'
        group[macd_line] = fast_ema - slow_ema
    group = group.dropna()
    return group


# https://www.alpharithms.com/stochastic-oscillator-in-python-483214/
def add_stochastic_oscillator_indicator(group, k_period=14, d_period=3):
    """
    Compute stochastic oscilattor indicator.
    """
    group = group.copy()
    max_vals = group['High'].rolling(k_period).max()
    min_vals = group['Low'].rolling(k_period).min()
    group['K'] = (group['Close'] - min_vals) / (max_vals - min_vals)
    group['K'] = group['K'] * 100 
    group['D'] = group['K'].rolling(d_period).mean()
    group = group.dropna()
    return group


def add_target_column(group):
    """
    Create target column for training.
    """
    group = group.copy()
    group['Target_Move'] = (group['Close'] - group['Close'].shift(-1)) < 0
    group = group.dropna()
    return group


def add_log_returns(group):
    """
    Compute log returns.
    """
    group = group.copy()
    group['Returns'] = group['Close'].shift(-1) / group['Close'] - 1
    group = group.dropna()
    return group
