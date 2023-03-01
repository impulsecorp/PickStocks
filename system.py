import os
import pickle as pkl
import random as rnd
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import plot
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from tqdm.notebook import tqdm

def seed_everything(seed=0):
    rnd.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed = 0
while seed == 0:
    seed = int(time.time() * 100000) % 1000000
seed_everything(seed)

def fix_data(data1, data2):
    data1 = data1[::-1]
    data2 = data2[::-1]
    s1, s2 = data1.index[0], data2.index[0]
    if s1 < s2:
        # find the index of s2 in data1
        try:
            idx = data1.index.to_list().index(s2)
            data1 = data1[idx:]
        except:
            idx = data2.index.to_list().index(s2)
            data2 = data2[idx:]
    elif s1 > s2:
        # find the index of s1 in data2
        try:
            idx = data2.index.to_list().index(s1)
            data2 = data2[idx:]
        except:
            idx = data1.index.to_list().index(s1)
            data1 = data1[idx:]
    data1 = data1[::-1]
    data2 = data2[::-1]
    s1, s2 = data1.index[0], data2.index[0]
    if s1 < s2:
        # find the index of s1 in data2
        try:
            idx = data2.index.to_list().index(s1)
            data2 = data2[idx:]
        except:
            idx = data1.index.to_list().index(s1)
            data1 = data1[idx:]
    elif s1 > s2:
        # find the index of s2 in data1
        try:
            idx = data1.index.to_list().index(s2)
            data1 = data1[idx:]
        except:
            idx = data2.index.to_list().index(s2)
            data2 = data2[idx:]
    return data1, data2


datadir = 'data'
try:
    os.mkdir('data')
except:
    pass


def get_data(symbol, period='D', nrows=None):
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    return data

def get_data_proc(symbol, period='D', nrows=None):
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '_proc.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    return data

def get_data_forex(from_symbol, to_symbol, period='D', nrows=None, parse_dates=['time'], index_col=0):
    if period == 'd': period = 'D'
    sfn = (from_symbol+to_symbol) + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    return data

def get_data_forex_proc(from_symbol, to_symbol, period='D', nrows=None, parse_dates=['time'], index_col=0):
    if period == 'd': period = 'D'
    sfn = (from_symbol+to_symbol) + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '_proc.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    return data

def get_data_pair(symbol1, symbol2, period='D'):
    s1, s1_f = get_data(symbol1, period=period)
    s2, s2_f = get_data(symbol2, period=period)

    s1, s2 = fix_data(s1, s2)
    s1_f, s2_f = fix_data(s1_f, s2_f)

    return (s1 / s2).dropna(), (s1_f / s2_f).dropna()


def read_tradestation_data(filename, test_size=0.2, max_rows=None, nosplit=0):
    d = pd.read_csv(filename)
    d = d[::-1]
    dts = [x + ' ' + y for x, y in zip(d['<Date>'].values.tolist(), d[' <Time>'].values.tolist())]
    idx = pd.DatetimeIndex(data=dts)
    v = np.vstack([d[' <Open>'].values,
                   d[' <High>'].values,
                   d[' <Low>'].values,
                   d[' <Close>'].values,
                   d[' <Volume>'].values, ]).T
 # type: ignore
    data = pd.DataFrame(data=v, columns=['1. open', '2. high', '3. low', '4. close', '5. volume', ], index=idx)

    if max_rows is not None:
        data = data[0:max_rows]

    if not nosplit:
        cp = int(test_size * len(data))
        future_data = data[0:cp]
        data = data[cp:]

        return data, future_data
    else:
        return data


def make_synth_data(data_timeperiod='D', start='1990-1-1', end='2020-1-1', freq='1min', nosplit=0):
    idx = pd.date_range(start=start, end=end, freq=freq)
    start_price = 10000
    plist = []
    vlist = []
    p = start_price
    for i in range(len(idx)):
        plist.append(p)
        p += rnd.uniform(-0.1, 0.1)
    plist = np.array(plist)
    plot(plist);
    df = pd.DataFrame(data=plist, index=idx)
    data_timeperiod = 'D'
    rdf = df.resample(data_timeperiod).ohlc()
    ddf = pd.DataFrame(data=rdf.values, columns=['Open', 'High', 'Low', 'Close'], index=rdf.index)
    vlist = [rnd.randint(10000, 50000) for _ in range(len(ddf))]
    ddf['Volume'] = vlist
    data = ddf
    if not nosplit:
        test_size = 0.2
        cp = int(test_size * len(data))
        future_data = data[cp:]
        data = data[0:cp]
        return data, future_data
    else:
        return data


def make_dataset(input_source, to_predict,
                 winlen=1, sliding_window_jump=1, predict_time_ahead=1,
                 remove_outliers=0, outlier_bounds=None, scaling=0, predict_method="default"):
    # create training set
    c0 = []
    c1 = []

    for i in range(0, input_source.shape[1] - (winlen + predict_time_ahead + 1), sliding_window_jump):
        # form the input
        xs = input_source[:, i:i + winlen].T
        if scaling: xs = scale(xs, axis=0)
        xs = xs.reshape(-1)

        # form the output
        if predict_method == "default":
            before_idx = 0
            after_idx = 3
            if before_idx == after_idx:
                q = 1
            else:
                q = 0
            sp = to_predict[before_idx, i + winlen - q]
            st = to_predict[after_idx, i + winlen]
        elif predict_method == "alternate":
            q = 1
            sp = to_predict[i + winlen - q]
            st = to_predict[i + winlen]
        else:
            raise ValueError("predict_method must be either 'default' or 'alternate'")

        if remove_outliers and (isinstance(outlier_bounds, tuple) and len(outlier_bounds) == 2):
            if ((st - sp) < outlier_bounds[0]) or ((st - sp) > outlier_bounds[1]):
                # outlier - too big move, something isn't right, so skip it
                continue

        if st >= sp:  # up move
            c0.append((xs, np.array([0])))
        else:  # down move
            c1.append((xs, np.array([1])))

    return c0, c1


def shuffle_split(c0, c1, balance_data=1):
    # shuffle and shape data 
    if balance_data:
        samplesize = min(len(c0), len(c1))
        s1 = rnd.sample(c0, samplesize)
        s2 = rnd.sample(c1, samplesize)
        a = s1 + s2
    else:
        a = c0 + c1
    rnd.shuffle(a)
    x = [x[0] for x in a]
    y = [x[1] for x in a]

    x = np.vstack(x)
    y = np.vstack(y)

    # use 80% as training set
    cutpoint = int(0.8 * x.shape[0])
    x_train = x[0:cutpoint]
    x_test = x[cutpoint:]
    y_train = y[0:cutpoint]
    y_test = y[cutpoint:]

    return x_train.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(
        np.float32), x.astype(np.float32), y.astype(np.float32)


def get_X(data):
    """Return matrix X"""
    return data.filter(like='X').values


def get_y(data):
    """ Return dependent variable y """
    y = data.Open.pct_change(1)  # Returns after roughly two days
    y[y >= 0] = 1
    y[y < 0] = 0
    return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y


def do_RL_backtest(env, clfs, scaling=0, scalers=None,
                envs=None, callback=None,
                do_plot=1, keras=0, proba=0, force_action=None, remove_outliers=0, outlier_bounds=None):
    binary = env.binary
    observation = env.reset()
    if envs is not None: _ = [x.reset() for x in envs]
    done = False
    obs = [observation]
    acts = []
    preds = []
    rewards = [0]
    try:
        gener = tqdm(range(env.bars_per_episode)) if do_plot else range(env.bars_per_episode)
        for i in gener:
            if not keras:
                if (callback is not None) and callable(callback):
                    if scaling and (scalers is not None):
                        aa = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                    else:
                        aa = callback(observation.reshape(1, -1), env, envs)
                else:
                    if proba:
                        if scaling and (scalers is not None):
                            aa = [clf.predict_proba(sc.transform(observation.reshape(1, -1)))[0][1]
                                  for sc, clf in zip(scalers, clfs)]
                        else:
                            aa = [clf.predict_proba(observation.reshape(1, -1))[0][1] for clf in clfs]
                    else:
                        if scaling and (scalers is not None):
                            aa = [clf.predict(sc.transform(observation.reshape(1, -1)))[0]
                                  for sc, clf in zip(scalers, clfs)]
                        else:
                            aa = [clf.predict(observation.reshape(1, -1))[0] for clf in clfs]
            else:
                aa = []
                for clf in clfs:
                    o = observation
                    if scaling: o = scale(o, axis=0)
                    p = clf.predict(o.reshape(1, env.winlen, -1))[0]
                    if len(p) > 1:
                        aa += [p[1]]
                    else:
                        aa += [p]

            # get the average
            if np.mean(aa) > 0.5:
                a = 1
            else:
                a = 0

            if envs is None:
                if not binary:
                    if a == 0:  # up
                        action = 0  # buy
                    elif a == 1:  # mid
                        action = 3  # do nothing
                    elif a == 2:  # down
                        action = 1  # sell
                else:
                    if a == 0:  # up
                        action = 0  # buy
                    elif a == 1:  # down
                        action = 1  # sell

                if force_action is not None:
                    if isinstance(force_action, str) and (force_action == 'random'): action = rnd.choice([0, 1])
                    if isinstance(force_action, int): action = force_action

                observation, reward, done, info = env.step(action)
            else:
                if len(envs) == 2:  # pair trading

                    if not binary:
                        if (callback is not None) and callable(callback):
                            if scaling and (scalers is not None):
                                actions = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                            else:
                                actions = callback(observation.reshape(1, -1), env, envs)
                        else:

                            if a == 0:  # up
                                actions = (1, 0)  # sell/buy
                            elif a == 1:  # mid
                                actions = (3, 3)  # do nothing
                            elif a == 2:  # down
                                actions = (0, 1)  # buy/sell
                    else:
                        if a == 0:  # up
                            actions = (1, 0)  # sell/buy
                        elif a == 1:  # down
                            actions = (0, 1)  # buy/sell

                    if force_action is not None:
                        if isinstance(force_action, str) and (force_action == 'random'): actions = (rnd.choice([0, 1])
                                                                                                    for x in envs)
                        if isinstance(force_action, int): actions = (force_action for x in envs)
                        if isinstance(force_action, tuple): actions = force_action

                    observation, reward, done, info = env.step(0)
                    rs = [x.step(y) for x, y in zip(envs, actions)]
                    reward = np.sum([x[1] for x in rs])

                    action = actions

                else:

                    if not binary:
                        if (callback is not None) and callable(callback):
                            if scaling and (scalers is not None):
                                actions = callback(scalers[0].transform(observation.reshape(1, -1)), env, envs)
                            else:
                                actions = callback(observation.reshape(1, -1), env, envs)

                    observation, reward, done, info = env.step(0)
                    rs = [x.step(y) for x, y in zip(envs, actions)]
                    reward = np.sum([x[1] for x in rs])

                    action = actions

                # raise ValueError('Not implemented.')

            obs.append(observation)
            acts.append(action)
            rewards.append(reward)
            preds.append(np.mean(aa))

            if done: break
    except KeyboardInterrupt:
        pass

    obs = np.vstack([x.reshape(-1) for x in obs])
    acts = np.array(acts)
    rewards = np.array(rewards)
    preds = np.array(preds)

    if envs is None:
        navs = np.array(env.returns)
    else:
        allt = []
        for x in envs:
            allt += x.trades
        navs = sorted(allt, key=lambda k: k[-2])
        d = {}
        for n in navs:
            if n[-2] in d:
                d[n[-2]] += n[3]
            else:
                d[n[-2]] = n[3]
        kv = list([(k, v) for k, v in d.items()])
        kv = sorted(kv, key=lambda k: k[0])
        navs = [x[1] for x in kv]

    if remove_outliers and (isinstance(outlier_bounds, tuple) and len(outlier_bounds) == 2):
        idx = np.where((outlier_bounds[0] < rewards) & (rewards < outlier_bounds[1]))[0]

        obs = obs[idx]
        acts = acts[idx]
        rewards = rewards[idx]
        preds = preds[idx]
        navs = navs[idx]

    if np.sum(rewards) == 0:
        rewards = navs

    if do_plot:
        kl = []
        t = 0
        for n in navs:
            t = t + n
            kl.append(t)
        plt.plot(kl)
        plt.plot([0, len(navs)], [0.0, 0.0], color='g', alpha=0.5)  # the zero line
        plt.show()

    return obs, acts, rewards, preds


def kdplot(preds, rewards, *args, **kwargs):
    x = np.linspace(np.min(preds), np.max(preds), 100)
    y = np.linspace(np.min(rewards), np.max(rewards), 100)
    X, Y = np.meshgrid(x, y)

    n = np.vstack([np.array(preds),
                   np.array(rewards)]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(n)

    Z = np.zeros((Y.shape[0], X.shape[0]))
    Z.shape

    samples = []
    for ay in y:
        for ax in x:
            samples.append([ax, ay])

    samples = np.array(samples)
    mz = kde.score_samples(samples)
    nk = 0
    for ay in range(Z.shape[0]):
        for ax in range(Z.shape[1]):
            Z[ay, ax] = mz[nk]
            nk += 1

    plt.contourf(X, Y, Z, levels=80);
    plt.scatter(preds, rewards, color='r', alpha=0.15);
    plot([np.min(preds), np.max(preds)], [np.mean(rewards), np.mean(rewards)], color='g', alpha=0.5);
    plot([np.mean(preds), np.mean(preds)], [np.min(rewards), np.max(rewards)], color='g', alpha=0.5);
