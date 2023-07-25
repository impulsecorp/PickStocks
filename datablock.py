import numpy as np
import pandas as pd
import tsfel as ts
from tqdm.notebook import tqdm
import pandas_ta as ta
import uuid

def uuname(): return str(uuid.uuid4()).replace('-','')[0:5]

def compute_custom_features(data, open_, high, low, close, uchar):
    # Some datetime features for good measure
    data['X' + uchar + 'day'] = data.index.dayofweek
    if not data.daily:
        data['X' + uchar + 'hour'] = data.index.hour
    # Additional custom features
    # Convert the index to datetime and create a temporary date variable
    dix = pd.to_datetime(data.index)
    dates = dix.date

    # Calculate the "overnight move" indicator
    overnight_move = []
    last_open = None
    overnight = 0
    for i, (xopen_, date) in enumerate(zip(open_.values, dates)):
        if (i > 0) and (date != dates[i - 1]):
            overnight = xopen_ - last_open
        overnight_move.append(overnight)
        last_open = xopen_
        # Add the "overnight move" column to the DataFrame
    data['X' + uchar + 'overnight_move'] = overnight_move

    b = open_.values[1:] - open_.values[0:-1]
    b = np.hstack([np.zeros(1), b])
    data['X' + uchar + 'open_move'] = b
    b = high.shift(1).values[1:] - high.shift(1).values[0:-1]
    b = np.hstack([np.zeros(1), b])
    data['X' + uchar + 'high_move'] = b
    b = low.shift(1).values[1:] - low.shift(1).values[0:-1]
    b = np.hstack([np.zeros(1), b])
    data['X' + uchar + 'low_move'] = b
    b = close.shift(1).values[1:] - close.shift(1).values[0:-1]
    b = np.hstack([np.zeros(1), b])
    data['X' + uchar + 'close_move'] = b
    b = close.shift(1).values - open_.shift(1).values
    data['X' + uchar + 'last_move'] = b
    b = high.shift(1).values - low.shift(1).values
    data['X' + uchar + 'last_span'] = b

    # times in row
    # Calculate the "X times in row" indicator
    x_in_row = []
    count = 0
    last_move = 0
    last_date = None
    for i, move in enumerate(data['X' + uchar + 'last_move'].values):
        if move * last_move > 0 and move != 0:
            count += 1
        else:
            count = 0
        date = dates[i]
        if not data.daily:
            if date != last_date:
                count = 0
        x_in_row.append(count)
        last_date = date
        last_move = move
    # Add the "X times in row" column to the DataFrame
    data['X' + uchar + 'times_in_row'] = x_in_row


    def mlag(n=1):
        b = open_.values[n:] - open_.values[0:-n]
        b = np.hstack([np.zeros(n), b])
        return b

    data['X' + uchar + 'pmove_2'] = mlag(2)
    data['X' + uchar + 'pmove_3'] = mlag(3)
    data['X' + uchar + 'pmove_4'] = mlag(4)
    data['X' + uchar + 'pmove_5'] = mlag(5)
    data['X' + uchar + 'pmove_10'] = mlag(10)

    # Compute the overnight move direction
    data['X' + uchar + 'overnight_direction'] = np.where(data['X' + uchar + 'overnight_move'] > 0, 1, -1)

    # Compute yesterday's close-to-open move
    yesterday_close = close.shift(1)
    yesterday_open = open_.shift(1)

    data['X' + uchar + 'yesterday_move'] = yesterday_open - yesterday_close

    # Indicator 1: Overnight move in the same direction as yesterday's close-to-open move
    data['X' + uchar + 'f1'] = np.where(
        data['X' + uchar + 'overnight_direction'].values == np.sign(data['X' + uchar + 'overnight_move'].values), 1, 0)

    # Indicator 2: Today's open is above yesterday's high
    data['X' + uchar + 'f2'] = np.where(open_ > high.shift(1), 1, 0)
    # Indicator 3: Today's open is below yesterday's low
    data['X' + uchar + 'f3'] = np.where(open_ < low.shift(1), 1, 0)
    # Indicator 4: Today's open is above yesterday's open but below yesterday's high
    data['X' + uchar + 'f4'] = np.where((open_ > yesterday_open) & (open_ < high.shift(1)), 1, 0)
    # Indicator 5: Today's open is below yesterday's close, but above yesterday's low
    data['X' + uchar + 'f5'] = np.where((open_ < yesterday_close) & (open_ > low.shift(1)), 1, 0)
    # Indicator 6: Today's open is between yesterday's open and close
    data['X' + uchar + 'f6'] = np.where((open_ > np.minimum(yesterday_open, yesterday_close)) &
                                        (open_ < np.maximum(yesterday_open, yesterday_close)), 1, 0)


def compute_custom_features_llm(data, open_, high, low, close, uchar):
    # Some datetime features for good measure
    data['Day of week'] = data.index.dayofweek
    if not data.daily:
        data['Hour of day'] = data.index.hour
    # Additional custom features
    # Convert the index to datetime and create a temporary date variable
    dix = pd.to_datetime(data.index)
    dates = dix.date

    # Calculate the "overnight move" indicator
    overnight_move = []
    last_open = None
    overnight = 0
    for i, (xopen_, date) in enumerate(zip(open_.values, dates)):
        if (i > 0) and (date != dates[i - 1]):
            overnight = xopen_ - last_open
        overnight_move.append(overnight)
        last_open = xopen_
        # Add the "overnight move" column to the DataFrame
    data['Overnight price move'] = overnight_move

    # b = open_.values[1:] - open_.values[0:-1]
    # b = np.hstack([np.zeros(1), b])
    # data['X' + uchar + 'open_move'] = b
    # b = high.shift(1).values[1:] - high.shift(1).values[0:-1]
    # b = np.hstack([np.zeros(1), b])
    # data['X' + uchar + 'high_move'] = b
    # b = low.shift(1).values[1:] - low.shift(1).values[0:-1]
    # b = np.hstack([np.zeros(1), b])
    # data['X' + uchar + 'low_move'] = b
    # b = close.shift(1).values[1:] - close.shift(1).values[0:-1]
    # b = np.hstack([np.zeros(1), b])
    # data['X' + uchar + 'close_move'] = b
    #
    b = close.shift(1).values - open_.shift(1).values
    data['Last price move'] = b
    b = high.shift(1).values - low.shift(1).values
    data['Last High-Low span'] = b

    # times in row
    # Calculate the "X times in row" indicator
    x_in_row = []
    count = 0
    last_move = 0
    last_date = None
    for i, move in enumerate(data['Last price move'].values):
        if move * last_move > 0 and move != 0:
            count += 1
        else:
            count = 0
        date = dates[i]
        if not data.daily:
            if date != last_date:
                count = 0
        x_in_row.append(count)
        last_date = date
        last_move = move
    # Add the "X times in row" column to the DataFrame
    data['Times in a row Up'] = x_in_row

    x_in_row = []
    count = 0
    last_move = 0
    last_date = None
    for i, move in enumerate(data['Last price move'].values):
        if move * last_move < 0 and move != 0:
            count += 1
        else:
            count = 0
        date = dates[i]
        if not data.daily:
            if date != last_date:
                count = 0
        x_in_row.append(count)
        last_date = date
        last_move = move
    # Add the "X times in row" column to the DataFrame
    data['Times in a row Down'] = x_in_row

    # Compute the overnight move direction
    data['Direction of overnight move'] = np.where(data['Overnight price move'] > 0, 1, -1)

    # Compute yesterday's close-to-open move
    yesterday_close = close.shift(1)
    yesterday_open = open_.shift(1)

    data["Yesterday's Close-to-Open move"] = yesterday_open - yesterday_close



didx = 0
data = None
dindex = None

def procdata(ddd, 
             use_tsfel=False, dwinlen=60,
             use_forex=False, double_underscore=True,
             cut_first_N=-1, with_lagged=1):
    global data, dindex

    daily = ddd.daily

    if not daily:
        ddd = ddd.between_time('09:30', '16:00')

    data = ddd
    dindex = ddd.index

    print('Computing features..', end=' ')
    
    uchar = '__' if double_underscore else '_'

    def addx(x):
        global data, didx, dindex
        if len(x.shape) > 1:
            dx = x.rename(lambda k: 'X' + uchar + k.lower() + '_' + uuname(), axis=1)
            data = pd.concat([data, dx], axis=1)
            data.index = dindex
        else:
            didx += 1
            data['X' + uchar + x.name.lower() + '_' + uuname()] = x
            data.index = dindex
        data.daily = daily

    # Retrieves a pre-defined feature configuration file to extract all available features
    if use_tsfel:
        cfg = ts.get_features_by_domain()
        nf = ts.time_series_features_extractor(cfg, data[0:dwinlen], verbose=0).shape[1]
        dw = [np.zeros(nf)] * dwinlen
        for i in tqdm(range(len(data) - dwinlen)):
            # Extract features
            X = ts.time_series_features_extractor(cfg, data[i:i + dwinlen], verbose=0)
            dw.append(X.values)
        dw = np.vstack(dw)
        cs = ['X'+uchar+'mmm' + str(i) for i in range(dw.shape[1])]
        d = pd.DataFrame(dw, columns=cs, index=data.index)
        data = pd.concat([data, d], axis=1)

    open_ = data.open.shift(1)
    high = data.high.shift(1)
    low = data.low.shift(1)
    close = data.close.shift(1)
    if not use_forex: volume = data.volume.shift(1)

    if not use_forex:
        data = data.rename({'open': 'X'+uchar+'Open',
                        'high': 'X'+uchar+'High',
                        'low': 'X'+uchar+'Low',
                        'close': 'X'+uchar+'Close',
                        'volume': 'X'+uchar+'Volume',
                        }, axis=1)
    else:
        data = data.rename({'open': 'X'+uchar+'Open',
                        'high': 'X'+uchar+'High',
                        'low': 'X'+uchar+'Low',
                        'close': 'X'+uchar+'Close',
                        }, axis=1)

    if 1:
        addx(ta.aberration(high, low, close, length=14, atr_length=14))
        addx(ta.aberration(high, low, close, length=20, atr_length=20))
        addx(ta.accbands(high, low, close, length=20, c=2, drift=1, mamode='sma'))
        addx(ta.accbands(high, low, close, length=14, c=2.5, drift=1, mamode='ema'))
        if not use_forex: addx(ta.ad(high, low, close, volume, open_=open_))
        if not use_forex: addx(ta.adosc(high, low, close, volume, open_=open_, fast=3, slow=10))
        if not use_forex: addx(ta.adosc(high, low, close, volume, open_=open_, fast=5, slow=21))
        addx(ta.adx(high, low, close, length=14, scalar=100, drift=1))
        addx(ta.adx(high, low, close, length=20, scalar=100, drift=1))
        addx(ta.alma(close, length=9, sigma=6, distribution_offset=0.85))
        addx(ta.alma(close, length=20, sigma=6, distribution_offset=0.85))
        addx(ta.ao(high, low, fast=5, slow=34))
        addx(ta.ao(high, low, fast=7, slow=20))
        if not use_forex: addx(ta.aobv(close, volume, fast=12, slow=26, mamode='ema', max_lookback=None, min_lookback=None, offset=None))
        addx(ta.apo(close, fast=12, slow=26))
        addx(ta.apo(close, fast=5, slow=20))
        addx(ta.aroon(high, low, length=25, scalar=100))
        addx(ta.aroon(high, low, length=14, scalar=100))
        addx(ta.atr(high, low, close, length=14, mamode='ema', drift=1))
        addx(ta.atr(high, low, close, length=20, mamode='ema', drift=1))
        addx(ta.bbands(close, length=20, std=2, mamode='sma'))
        addx(ta.bbands(close, length=14, std=2, mamode='ema'))
        addx(ta.bias(close, length=14, mamode='ema'))
        addx(ta.bias(close, length=20, mamode='sma'))
        addx(ta.bop(open_, high, low, close, scalar=100))
        addx(ta.brar(open_, high, low, close, length=26, scalar=100, drift=1))
        addx(ta.brar(open_, high, low, close, length=14, scalar=100, drift=1))
        addx(ta.cci(high, low, close, length=20, c=0.015))
        addx(ta.cci(high, low, close, length=14, c=0.015))
        addx(ta.cfo(close, length=9, scalar=100, drift=1))
        addx(ta.cfo(close, length=14, scalar=100, drift=1))
        addx(ta.cg(close, length=100))
        addx(ta.cg(close, length=200))
        addx(ta.chop(high, low, close, length=14, atr_length=14, scalar=100, drift=1))
        addx(ta.chop(high, low, close, length=20, atr_length=20, scalar=100, drift=1))
        addx(ta.cksp(high, low, close, p=1, x=10, q=9))
        addx(ta.cksp(high, low, close, p=1, x=14, q=9))
        if not use_forex: addx(ta.cmf(high, low, close, volume, open_=open_, length=20))
        if not use_forex: addx(ta.cmf(high, low, close, volume, open_=open_, length=14))
        addx(ta.cmo(close, length=9, scalar=100, drift=1))
        addx(ta.cmo(close, length=14, scalar=100, drift=1))
        addx(ta.coppock(close, length=11, fast=14, slow=11))
        addx(ta.coppock(close, length=11, fast=10, slow=15))
        addx(ta.decay(close, kind='linear', length=14, mode=None))
        addx(ta.decay(close, kind='exponential', length=14, mode=None))
        addx(ta.decreasing(close, length=5, strict=False, asint=True))
        addx(ta.decreasing(close, length=10, strict=False, asint=True))
        addx(ta.dema(close, length=20))
        addx(ta.dema(close, length=14))
        addx(ta.donchian(high, low, lower_length=20, upper_length=20))
        addx(ta.donchian(high, low, lower_length=14, upper_length=14))
        addx(ta.ebsw(close, length=20, bars=5))
        addx(ta.ebsw(close, length=14, bars=7))
        if not use_forex: addx(ta.efi(close, volume, length=13, drift=1, mamode='ema'))
        if not use_forex: addx(ta.efi(close, volume, length=20, drift=1, mamode='sma'))
        addx(ta.ema(close, length=20))
        addx(ta.ema(close, length=14))
        addx(ta.entropy(close, length=20, base=2))
        addx(ta.entropy(close, length=14, base=2))
        if not use_forex: addx(ta.eom(high, low, close, volume, length=14, divisor=10000, drift=1))
        if not use_forex: addx(ta.eom(high, low, close, volume, length=20, divisor=10000, drift=1))
        addx(ta.er(close, length=10, drift=1))
        addx(ta.er(close, length=14, drift=1))
        addx(ta.eri(high, low, close, length=14))
        addx(ta.eri(high, low, close, length=20))
        addx(ta.fisher(high, low, length=9, signal=1))
        addx(ta.fisher(high, low, length=14, signal=1))
        addx(ta.fwma(close, length=20, asc=True))
        addx(ta.fwma(close, length=14, asc=True))
        addx(ta.ha(open_, high, low, close))
        addx(ta.hilo(high, low, close, length=20, atr_length=20, scalar=100))
        addx(ta.hilo(high, low, close, length=14, atr_length=14, scalar=100))
        addx(ta.increasing(close, length=5, strict=False, asint=True))
        addx(ta.increasing(close, length=10, strict=False, asint=True))
        addx(ta.kama(close, length=10))
        addx(ta.kama(close, length=14))
        addx(ta.kc(high, low, close, length=20, kc_mult=2, mamode='sma', scalar=None, drift=1))
        addx(ta.kc(high, low, close, length=14, kc_mult=2, mamode='ema', scalar=None, drift=1))
        addx(ta.kdj(high, low, close, k_period=9, d_period=3, j_period=3))
        addx(ta.kdj(high, low, close, k_period=14, d_period=3, j_period=3))
        addx(ta.kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, s1=3, s2=3, s3=3, s4=9))
        addx(ta.kst(close, r1=14, r2=20, r3=30, r4=40, n1=14, n2=14, n3=14, n4=21, s1=3, s2=3, s3=3, s4=9))
        addx(ta.linreg(close, length=20, as_slope=True, r_squared=False))
        addx(ta.linreg(close, length=14, as_slope=True, r_squared=False))
        addx(ta.log_return(close, length=1))
        addx(ta.log_return(close, length=2))
        addx(ta.macd(close, fast=12, slow=26, signal=9))
        addx(ta.macd(close, fast=5, slow=35, signal=5))
        addx(ta.mad(close, length=20, mamode='ema', as_log=True))
        addx(ta.mad(close, length=14, mamode='sma', as_log=True))
        addx(ta.massi(high, low, length=25))
        addx(ta.massi(high, low, length=9))
        addx(ta.mfi(high, low, close, volume, length=14, scalar=100, drift=1))
        addx(ta.mfi(high, low, close, volume, length=20, scalar=100, drift=1))
        addx(ta.midpoint(close, length=20))
        addx(ta.midpoint(close, length=14))
        addx(ta.midprice(high, low, length=20))
        addx(ta.midprice(high, low, length=14))
        addx(ta.mom(close, length=14, scalar=100, drift=1))
        addx(ta.mom(close, length=20, scalar=100, drift=1))
        addx(ta.natr(high, low, close, length=20, mamode='ema', drift=1))
        addx(ta.natr(high, low, close, length=14, mamode='ema', drift=1))
        addx(ta.obv(close, volume, drift=1))
        addx(ta.pdist(open_, high, low, close, fast=4, slow=9, scalar=100, drift=1))
        addx(ta.pdist(open_, high, low, close, fast=2, slow=5, scalar=100, drift=1))
        addx(ta.percent_return(close, length=1))
        addx(ta.percent_return(close, length=2))
        addx(ta.ppo(close, fast=12, slow=26, scalar=100))
        addx(ta.ppo(close, fast=5, slow=35, scalar=100))
        addx(ta.pvol(close, volume))
        addx(ta.pwma(close, length=20, asc=True))
        addx(ta.pwma(close, length=14, asc=True))
        addx(ta.qstick(open_, close, length=20, mamode='ema'))
        addx(ta.qstick(open_, close, length=14, mamode='sma'))
        addx(ta.rsi(close, length=14, scalar=100, drift=1))
        addx(ta.rsi(close, length=20, scalar=100, drift=1))
        addx(ta.rvgi(open_, high, low, close, length=14))
        addx(ta.rvgi(open_, high, low, close, length=20))
        addx(ta.rvi(close, high, low, length=14))
        addx(ta.rvi(close, high, low, length=20))
        addx(ta.slope(close, length=20))
        addx(ta.slope(close, length=14))
        addx(ta.sma(close, length=20))
        addx(ta.sma(close, length=14))
        addx(ta.stdev(close, length=20))
        addx(ta.stdev(close, length=14))
        addx(ta.stoch(high, low, close, fast_k=14, slow_k=3, slow_d=3, scalar=100))
        addx(ta.stoch(high, low, close, fast_k=20, slow_k=5, slow_d=5, scalar=100))
        addx(ta.supertrend(high, low, close, length=10, multiplier=3))
        addx(ta.supertrend(high, low, close, length=14, multiplier=3))
        addx(ta.t3(close, length=20, a=0.7))
        addx(ta.t3(close, length=14, a=0.7))
        addx(ta.tema(close, length=20))
        addx(ta.tema(close, length=14))
        addx(ta.trima(close, length=20))
        addx(ta.trima(close, length=14))
        addx(ta.trix(close, length=18))
        addx(ta.trix(close, length=14))
        addx(ta.tsi(close, fast=25, slow=13, signal=13))
        addx(ta.tsi(close, fast=20, slow=10, signal=10))
        addx(ta.uo(high, low, close, fast=7, medium=14, slow=28, scalar=100, drift=1))
        addx(ta.uo(high, low, close, fast=5, medium=10, slow=20, scalar=100, drift=1))
        addx(ta.vhf(close, length=20, mamode='ema', drift=1))
        addx(ta.vhf(close, length=14, mamode='sma', drift=1))
        addx(ta.vortex(high, low, close, length=14))
        addx(ta.vortex(high, low, close, length=20))
        addx(ta.wcp(high, low, close, length=20, mamode='ema'))
        addx(ta.wcp(high, low, close, length=14, mamode='sma'))
        addx(ta.willr(high, low, close, length=14, scalar=100, drift=1))
        addx(ta.willr(high, low, close, length=20, scalar=100, drift=1))
        addx(ta.wma(close, length=20))
        addx(ta.wma(close, length=14))
        addx(ta.zlma(close, length=20))
        addx(ta.zlma(close, length=14))

    data = data.rename({'X__Open': 'Open',
                        'X__High': 'High',
                        'X__Low': 'Low',
                        'X__Close': 'Close',
                        'X__Volume': 'Volume',
                        }, axis=1)

    if with_lagged:
        # lag all features except the raw prices
        # Filter columns with the 'X' prefix
        features = data.filter(like='X')
        # Lag the features by one period
        lagged_features = features.shift(1)
        difference = features - lagged_features
        for column in difference.columns:
            data[f'{column}_lagged'] = difference[column]

    data.daily = daily
    compute_custom_features(data, open_, high, low, close, uchar)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0).astype(float)

    # cut off the first N rows, because they are likely nans
    if cut_first_N > 0: data = data[cut_first_N:]

    data.daily = daily
    print('Done.')
    return data






def procdata_lite(ddd, use_forex=False, double_underscore=True, cut_first_N=-1,
                  with_lagged=1):
    global data, dindex

    daily = ddd.daily

    if not daily:
        ddd = ddd.between_time('09:30', '16:00')

    data = ddd
    dindex = ddd.index

    print('Computing features..', end=' ')
    
    uchar = '__' if double_underscore else '_'

    def addx(x):
        global data, didx, dindex
        if len(x.shape) > 1:
            dx = x.rename(lambda k: 'X' + uchar + k.lower() + '_' + uuname(), axis=1)
            data = pd.concat([data, dx], axis=1)
            data.index = dindex
        else:
            didx += 1
            data['X' + uchar + x.name.lower() + '_' + uuname()] = x
            data.index = dindex
        data.daily = daily

    open_ = data.open.shift(1)
    high = data.high.shift(1)
    low = data.low.shift(1)
    close = data.close.shift(1)
    if not use_forex: volume = data.volume.shift(1)

    if not use_forex:
        data = data.rename({'open': 'X'+uchar+'Open',
                        'high': 'X'+uchar+'High',
                        'low': 'X'+uchar+'Low',
                        'close': 'X'+uchar+'Close',
                        'volume': 'X'+uchar+'Volume',
                        }, axis=1)
    else:
        data = data.rename({'open': 'X'+uchar+'Open',
                        'high': 'X'+uchar+'High',
                        'low': 'X'+uchar+'Low',
                        'close': 'X'+uchar+'Close',
                        }, axis=1)

    if 1:
        addx(ta.adx(high, low, close, length=14))
        addx(ta.adx(high, low, close, length=20))
        addx(ta.atr(high, low, close, length=14))
        addx(ta.atr(high, low, close, length=20))
        addx(ta.bbands(close, length=20, std=2))
        addx(ta.bbands(close, length=14, std=2))
        addx(ta.cci(high, low, close, length=20, c=0.015))
        addx(ta.cci(high, low, close, length=14, c=0.015))
        addx(ta.cmo(close, length=14))
        addx(ta.cmo(close, length=20))
        addx(ta.decay(close, kind='linear', length=20))
        addx(ta.decay(close, kind='linear', length=14))
        addx(ta.ema(close, length=20))
        addx(ta.ema(close, length=14))
        addx(ta.entropy(close, length=20))
        addx(ta.entropy(close, length=14))
        addx(ta.macd(close, fast=12, slow=26, signal=9))
        addx(ta.macd(close, fast=8, slow=16, signal=6))
        addx(ta.mom(close, length=14))
        addx(ta.mom(close, length=20))
        addx(ta.natr(high, low, close, length=14))
        addx(ta.natr(high, low, close, length=20))
        addx(ta.rma(close, length=14))
        addx(ta.rma(close, length=20))
        addx(ta.roc(close, length=14))
        addx(ta.roc(close, length=20))
        addx(ta.rsi(close, length=14))
        addx(ta.rsi(close, length=20))
        addx(ta.rsx(close, length=14))
        addx(ta.rsx(close, length=20))
        addx(ta.slope(close, length=14))
        addx(ta.slope(close, length=20))
        addx(ta.sma(close, length=20))
        addx(ta.sma(close, length=14))
        addx(ta.stoch(high, low, close, k=14, d=3, smooth_k=3))
        addx(ta.stoch(high, low, close, k=20, d=4, smooth_k=4))
        addx(ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3))
        addx(ta.stochrsi(close, length=20, rsi_length=20, k=4, d=4))
        addx(ta.supertrend(high, low, close, length=7, multiplier=3))
        addx(ta.supertrend(high, low, close, length=10, multiplier=3))
        addx(ta.willr(high, low, close, length=14))
        addx(ta.willr(high, low, close, length=20))


    data = data.rename({'X__Open': 'Open',
                        'X__High': 'High',
                        'X__Low': 'Low',
                        'X__Close': 'Close',
                        'X__Volume': 'Volume',
                        }, axis=1)

    if with_lagged:
        # lag all features except the raw prices
        # Filter columns with the 'X' prefix
        features = data.filter(like='X')
        # Lag the features by one period
        lagged_features = features.shift(1)
        difference = features - lagged_features
        for column in difference.columns:
            data[f'{column}_lagged'] = difference[column]

    data.daily = daily
    compute_custom_features(data, open_, high, low, close, uchar)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0).astype(float)

    # cut off the first N rows, because they are likely nans
    if cut_first_N > 0: data = data[cut_first_N:]

    data.daily = daily
    print('Done.')
    return data


def procdata_llm(ddd, use_forex=False, double_underscore=True, cut_first_N=-1,
                  with_lagged=0):
    global data, dindex

    daily = ddd.daily

    if not daily:
        ddd = ddd.between_time('09:30', '16:00')

    data = ddd
    dindex = ddd.index

    print('Computing features..', end=' ')

    uchar = '__' if double_underscore else '_'

    def addx(x):
        global data, didx, dindex
        if len(x.shape) > 1:
            dx = x.rename(lambda k: k.upper(), axis=1)
            data = pd.concat([data, dx], axis=1)
            data.index = dindex
        else:
            didx += 1
            data[x.name.upper()] = x
            data.index = dindex
        data.daily = daily

    open_ = data.open.shift(1)
    high = data.high.shift(1)
    low = data.low.shift(1)
    close = data.close.shift(1)
    if not use_forex: volume = data.volume.shift(1)

    if not use_forex:
        data = data.rename({'open': 'X' + uchar + 'Open',
                            'high': 'X' + uchar + 'High',
                            'low': 'X' + uchar + 'Low',
                            'close': 'X' + uchar + 'Close',
                            'volume': 'X' + uchar + 'Volume',
                            }, axis=1)
    else:
        data = data.rename({'open': 'X' + uchar + 'Open',
                            'high': 'X' + uchar + 'High',
                            'low': 'X' + uchar + 'Low',
                            'close': 'X' + uchar + 'Close',
                            }, axis=1)

    if 1:
        addx(ta.adx(high, low, close, length=14))
        addx(ta.atr(high, low, close, length=14))
        # addx(ta.bbands(close, length=14, std=2))
        addx(ta.cci(high, low, close, length=14, c=0.015))
        # addx(ta.cmo(close, length=14))
        # addx(ta.decay(close, kind='linear', length=14))
        # addx(ta.ema(close, length=14))
        # addx(ta.entropy(close, length=14))
        # addx(ta.macd(close, fast=8, slow=16, signal=6))
        # addx(ta.mom(close, length=14))
        # addx(ta.natr(high, low, close, length=14))
        # addx(ta.rma(close, length=14))
        # addx(ta.roc(close, length=14))
        addx(ta.rsi(close, length=14))
        # addx(ta.rsx(close, length=14))
        addx(ta.sma(close, length=14))
        # addx(ta.stoch(high, low, close, k=14, d=3, smooth_k=3))
        # addx(ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3))
        # addx(ta.supertrend(high, low, close, length=7, multiplier=3))
        addx(ta.willr(high, low, close, length=14))

    data = data.rename({'X__Open': 'Open',
                        'X__High': 'High',
                        'X__Low': 'Low',
                        'X__Close': 'Close',
                        'X__Volume': 'Volume',
                        }, axis=1)

    data.daily = daily
    compute_custom_features_llm(data, open_, high, low, close, uchar)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0).astype(float)

    # cut off the first N rows, because they are likely nans
    if cut_first_N > 0: data = data[cut_first_N:]

    data.daily = daily
    print('Done.')
    return data
