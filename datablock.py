import numpy as np
import pandas as pd
import tsfel as ts
from tqdm.notebook import tqdm
import pandas_ta as ta
from system import make_dataset, get_data, shuffle_split, make_synth_data

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
    # yesterday_close = close.shift(1)
    # yesterday_open = open_.shift(1)
    #
    # data['X' + uchar + 'yesterday_move'] = yesterday_open - yesterday_close

    # Indicator 1: Overnight move in the same direction as yesterday's close-to-open move
    data['X' + uchar + 'f1'] = np.where(
        data['X' + uchar + 'overnight_direction'].values == np.sign(data['X' + uchar + 'yesterday_move'].values), 1, 0)
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

didx = 0
data = None
dindex = None

def procdata(ddd, 
             use_tsfel=True, dwinlen=60,
             use_forex=False, double_underscore=True,
             cut_first_N=-1):
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
            dx = x.rename(lambda k: 'X' + uchar + k.lower(), axis=1)
            data = pd.concat([data, dx], axis=1)
            data.index = dindex
        else:
            didx += 1
            data['X' + uchar + 'feat_' + str(didx).lower()] = x
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
        addx(ta.aberration(high, low, close, length=None, atr_length=None, offset=None))
        addx(ta.accbands(high, low, close, length=None, c=None, drift=None, mamode=None, offset=None))
        if not use_forex: addx(ta.ad(high, low, close, volume, open_=None, offset=None))
        if not use_forex: addx(ta.adosc(high, low, close, volume, open_=None, fast=None, slow=None, offset=None))
        addx(ta.adx(high, low, close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.alma(close, length=None, sigma=None, distribution_offset=None, offset=None))
        addx(ta.ao(high, low, fast=None, slow=None, offset=None))
        if not use_forex: addx(ta.aobv(close, volume, fast=None, slow=None, mamode=None, max_lookback=None, min_lookback=None,
                     offset=None))
        addx(ta.apo(close, fast=None, slow=None, offset=None))
        addx(ta.aroon(high, low, length=None, scalar=None, offset=None))
        addx(ta.atr(high, low, close, length=None, mamode=None, drift=None, offset=None))
        addx(ta.bbands(close, length=None, std=None, mamode=None, offset=None))
        addx(ta.bias(close, length=None, mamode=None, offset=None))
        addx(ta.bop(open_, high, low, close, scalar=None, offset=None))
        addx(ta.brar(open_, high, low, close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.cci(high, low, close, length=None, c=None, offset=None))
        addx(ta.cfo(close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.cg(close, length=None, offset=None))
        addx(ta.chop(high, low, close, length=None, atr_length=None, scalar=None, drift=None, offset=None))
        addx(ta.cksp(high, low, close, p=None, x=None, q=None, offset=None))
        if not use_forex: addx(ta.cmf(high, low, close, volume, open_=None, length=None, offset=None))
        addx(ta.cmo(close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.coppock(close, length=None, fast=None, slow=None, offset=None))
        addx(ta.decay(close, kind=None, length=None, mode=None, offset=None))
        addx(ta.decreasing(close, length=None, strict=None, asint=None, offset=None))
        addx(ta.dema(close, length=None, offset=None))
        addx(ta.donchian(high, low, lower_length=None, upper_length=None, offset=None))
        #addx(ta.dpo(close, length=None, centered=True, offset=None)) # LEAK 
        addx(ta.ebsw(close, length=None, bars=None, offset=None))
        if not use_forex: addx(ta.efi(close, volume, length=None, drift=None, mamode=None, offset=None))
        addx(ta.ema(close, length=None, offset=None))
        addx(ta.entropy(close, length=None, base=None, offset=None))
        if not use_forex: addx(ta.eom(high, low, close, volume, length=None, divisor=None, drift=None, offset=None))
        addx(ta.er(close, length=None, drift=None, offset=None))
        addx(ta.eri(high, low, close, length=None, offset=None))
        addx(ta.fisher(high, low, length=None, signal=None, offset=None))
        addx(ta.fwma(close, length=None, asc=None, offset=None))
        addx(ta.ha(open_, high, low, close, offset=None))
        addx(ta.hilo(high, low, close, high_length=None, low_length=None, mamode=None, offset=None))
        addx(ta.hl2(high, low, offset=None))
        addx(ta.hlc3(high, low, close, offset=None))
        addx(ta.hma(close, length=None, offset=None))
        addx(ta.hwc(close, na=None, nb=None, nc=None, nd=None, scalar=None, channel_eval=None, offset=None))
        addx(ta.hwma(close, na=None, nb=None, nc=None, offset=None))
        addx(ta.increasing(close, length=None, strict=None, asint=None, offset=None))
        addx(ta.kama(close, length=None, fast=None, slow=None, drift=None, offset=None))
        addx(ta.kc(high, low, close, length=None, scalar=None, mamode=None, offset=None))
        addx(ta.kst(close, roc1=None, roc2=None, roc3=None, roc4=None, sma1=None, sma2=None, sma3=None, sma4=None,
                    signal=None, drift=None, offset=None))
        addx(ta.kurtosis(close, length=None, offset=None))
        addx(ta.linreg(close, length=None, offset=None))
        addx(ta.log_return(close, length=None, cumulative=False, offset=None))
        addx(ta.macd(close, fast=None, slow=None, signal=None, offset=None))
        addx(ta.mad(close, length=None, offset=None))
        addx(ta.massi(high, low, fast=None, slow=None, offset=None))
        addx(ta.mcgd(close, length=None, offset=None, c=None))
        addx(ta.median(close, length=None, offset=None))
        if not use_forex: addx(ta.mfi(high, low, close, volume, length=None, drift=None, offset=None))
        addx(ta.midpoint(close, length=None, offset=None))
        addx(ta.midprice(high, low, length=None, offset=None))
        addx(ta.mom(close, length=None, offset=None))
        addx(ta.natr(high, low, close, length=None, mamode=None, scalar=None, drift=None, offset=None))
        if not use_forex: addx(ta.nvi(close, volume, length=None, initial=None, offset=None))
        if not use_forex: addx(ta.obv(close, volume, offset=None))
        addx(ta.ohlc4(open_, high, low, close, offset=None))
        addx(ta.pdist(open_, high, low, close, drift=None, offset=None))
        addx(ta.percent_return(close, length=None, cumulative=False, offset=None))
        addx(ta.pgo(high, low, close, length=None, offset=None))
        addx(ta.ppo(close, fast=None, slow=None, signal=None, scalar=None, offset=None))
        addx(ta.psar(high, low, close=None, af=None, max_af=None, offset=None))
        addx(ta.psl(close, open_=None, length=None, scalar=None, drift=None, offset=None))
        if not use_forex: addx(ta.pvi(close, volume, length=None, initial=None, offset=None))
        if not use_forex: addx(ta.pvo(volume, fast=None, slow=None, signal=None, scalar=None, offset=None))
        if not use_forex: addx(ta.pvol(close, volume, offset=None))
        if not use_forex: addx(ta.pvr(close, volume))
        if not use_forex: addx(ta.pvt(close, volume, drift=None, offset=None))
        addx(ta.pwma(close, length=None, asc=None, offset=None))
        addx(ta.qqe(close, length=None, smooth=None, factor=None, mamode=None, drift=None, offset=None))
        addx(ta.qstick(open_, close, length=None, offset=None))
        addx(ta.quantile(close, length=None, q=None, offset=None))
        addx(ta.rma(close, length=None, offset=None))
        addx(ta.roc(close, length=None, offset=None))
        addx(ta.rsi(close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.rsx(close, length=None, drift=None, offset=None))
        addx(ta.rvgi(open_, high, low, close, length=None, swma_length=None, offset=None))
        addx(ta.rvi(close, high=None, low=None, length=None, scalar=None, refined=None, thirds=None, mamode=None,
                    drift=None, offset=None))
        addx(ta.sinwma(close, length=None, offset=None))
        addx(ta.skew(close, length=None, offset=None))
        addx(ta.slope(close, length=None, as_angle=None, to_degrees=None, vertical=None, offset=None))
        addx(ta.sma(close, length=None, offset=None))
        addx(ta.smi(close, fast=None, slow=None, signal=None, scalar=None, offset=None))
        addx(ta.squeeze(high, low, close, bb_length=None, bb_std=None, kc_length=None, kc_scalar=None, mom_length=None,
                        mom_smooth=None, use_tr=None, offset=None))
        addx(ta.ssf(close, length=None, poles=None, offset=None))
        addx(ta.stdev(close, length=None, ddof=1, offset=None))
        addx(ta.stoch(high, low, close, k=None, d=None, smooth_k=None, offset=None))
        addx(ta.stochrsi(close, length=None, rsi_length=None, k=None, d=None, offset=None))
        addx(ta.supertrend(high, low, close, length=None, multiplier=None, offset=None))
        addx(ta.swma(close, length=None, asc=None, offset=None))
        addx(ta.t3(close, length=None, a=None, offset=None))
        addx(ta.tema(close, length=None, offset=None))
        addx(ta.thermo(high, low, length=None, long=None, short=None, mamode=None, drift=None, offset=None))
        addx(ta.trima(close, length=None, offset=None))
        addx(ta.trix(close, length=None, signal=None, scalar=None, drift=None, offset=None))
        addx(ta.true_range(high, low, close, drift=None, offset=None))
        addx(ta.tsi(close, fast=None, slow=None, scalar=None, drift=None, offset=None))
        addx(ta.ttm_trend(high, low, close, length=None, offset=None))
        addx(ta.ui(close, length=None, scalar=None, offset=None))
        addx(ta.uo(high, low, close, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None,
                   drift=None, offset=None))
        addx(ta.variance(close, length=None, ddof=None, offset=None))
        addx(ta.vidya(close, length=None, drift=None, offset=None))
        addx(ta.vortex(high, low, close, length=None, drift=None, offset=None))
        if not use_forex: addx(ta.vwma(close, volume, length=None, offset=None))
        addx(ta.wcp(high, low, close, offset=None))
        addx(ta.willr(high, low, close, length=None, offset=None))
        addx(ta.wma(close, length=None, asc=None, offset=None))
        addx(ta.zlma(close, length=None, mamode=None, offset=None))
        addx(ta.zscore(close, length=None, std=None, offset=None))

    data.daily = daily
    compute_custom_features(data, open_, high, low, close, uchar)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0).astype(float)

    # cut off the first N rows, because they are likely nans
    if cut_first_N > 0: data = data[cut_first_N:]

    data = data.rename({'X__Open': 'Open',
                        'X__High': 'High',
                        'X__Low': 'Low',
                        'X__Close': 'Close',
                        'X__Volume': 'Volume',
                        }, axis=1)
    data.daily = daily
    print('Done.')
    return data






def procdata_lite(ddd, use_forex=False, double_underscore=True, cut_first_N=-1):
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
            dx = x.rename(lambda k: 'X' + uchar + k.lower(), axis=1)
            data = pd.concat([data, dx], axis=1)
            data.index = dindex
        else:
            didx += 1
            data['X' + uchar + 'feat_' + str(didx).lower()] = x
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
        addx(ta.adx(high, low, close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.atr(high, low, close, length=None, mamode=None, drift=None, offset=None))
        addx(ta.bbands(close, length=None, std=None, mamode=None, offset=None))
        addx(ta.cci(high, low, close, length=None, c=None, offset=None))
        addx(ta.cmo(close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.decay(close, kind=None, length=None, mode=None, offset=None))
        addx(ta.ema(close, length=None, offset=None))
        addx(ta.entropy(close, length=None, base=None, offset=None))
        addx(ta.macd(close, fast=None, slow=None, signal=None, offset=None))
        addx(ta.mom(close, length=None, offset=None))
        addx(ta.natr(high, low, close, length=None, mamode=None, scalar=None, drift=None, offset=None))
        addx(ta.rma(close, length=None, offset=None))
        addx(ta.roc(close, length=None, offset=None))
        addx(ta.rsi(close, length=None, scalar=None, drift=None, offset=None))
        addx(ta.rsx(close, length=None, drift=None, offset=None))
        addx(ta.slope(close, length=None, as_angle=None, to_degrees=None, vertical=None, offset=None))
        addx(ta.sma(close, length=None, offset=None))
        addx(ta.stoch(high, low, close, k=None, d=None, smooth_k=None, offset=None))
        addx(ta.stochrsi(close, length=None, rsi_length=None, k=None, d=None, offset=None))
        addx(ta.supertrend(high, low, close, length=None, multiplier=None, offset=None))   
        addx(ta.willr(high, low, close, length=None, offset=None))

    data.daily = daily
    compute_custom_features(data, open_, high, low, close, uchar)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0).astype(float)

    # cut off the first N rows, because they are likely nans
    if cut_first_N > 0: data = data[cut_first_N:]

    data = data.rename({'X__Open': 'Open',
                        'X__High': 'High',
                        'X__Low': 'Low',
                        'X__Close': 'Close',
                        'X__Volume': 'Volume',
                        }, axis=1)
    data.daily = daily
    print('Done.')
    return data


