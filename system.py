import datetime
import os

# os.environ['BOKEH_RESOURCES'] = 'inline'
# import bokeh.util.warnings
import warnings

warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=bokeh.util.warnings.BokehDeprecationWarning, module='bokeh')
# warnings.filterwarnings('ignore', category=bokeh.util.warnings.BokehUserWarning, module='bokeh')
import random as rnd
import time as stime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from matplotlib.pyplot import gca
from matplotlib.pyplot import plot
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp
import warnings
warnings.filterwarnings('ignore')

def reseed():
    def seed_everything(s=0):
        rnd.seed(s)
        np.random.seed(s)
        os.environ['PYTHONHASHSEED'] = str(s)

    seed = 0
    while seed == 0:
        seed = int(stime.time() * 100000) % 1000000
    seed_everything(seed)
    return seed
seed = reseed()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from datetime import datetime, timedelta, time

# global parameters

train_set_end = 0.4  # percentage point specifying the training set end point (1.0 means all data is training set)
val_set_end = 0.7  # percentage point specifying the validation set end point (1.0 means no test set and all data after the previous point is validation )
# basically this is the data with the values above, which are like sliders determining the layout
# [|0.0| ......... train .......... |0.4| ............ val ............ |0.7| .............. test ............... |1.0|]

max_tries = 0.2  # for optimization, percentage of the grid space to cover (1.0 = exchaustive search)


# the objective function to maximize during optimization
def objective(s):
    return (0.05 * s['SQN'] +
            0.0 * s['Profit Factor'] +
            0.25 * s['Win Rate [%]'] / 100.0 +
            0.2 * s['Exposure Time [%]'] / 100.0 +
            1.0 * s['Return [%]']
            )


# keyword parameters nearly always the same
btkw = dict(commission=.000, margin=1.0, trade_on_close=False, exclusive_orders=True)
optkw = dict(method='grid', max_tries=max_tries, maximize=objective, return_heatmap=True)


def get_optdata(results, consts):
    return results[1][tuple([consts[y][0] for y in [x for x in consts.keys()]
                             if consts[y][0] in [x[0] for x in results[1].index.levels]])]


def plot_result(bt, results):
    try:
        bt.plot(plot_width=1200, plot_volume=False, plot_pl=1, resample=False);
    except Exception as ex:
        print(str(ex))
        plot(np.cumsum(results[0]['_trades']['PnL'].values));


def plot_optresult(rdata, feature_name):
    if rdata.index.to_numpy().shape[0] > 2:
        rdata.plot(kind='line', use_index=False);
        gca().set_xlabel(feature_name)
        gca().set_ylabel('objective')
    else:
        xs = rdata.index.values
        goodidx = np.where(~np.isnan(rdata.values))[0]
        xs = xs[goodidx]
        rda = rdata.values[goodidx]

        if not isinstance(xs[0], time):
            plt.plot(xs, rda)
            gca().set_xlabel(feature_name)
            gca().set_ylabel('objective')
            if xs.dtype.kind == 'f':
                try:
                    gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10), rotation=45)
                except:
                    gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10))
            else:
                gca().set_xticks(np.linspace(np.min(xs), np.max(xs), 10))
        else:
            # convert xs to a list of datetime.datetime objects with a fixed date
            fixed_date = datetime(2022, 1, 1)  # or any other date you prefer
            ixs = xs[:]
            xs = [datetime.combine(fixed_date, x) for x in xs]
            # convert xs to a list of floats using date2num
            xs = date2num(xs)

            # plot the data
            ax = gca()
            ax.plot(xs, rda)
            ax.set_xticks(xs)
            ax.set_xticklabels([x.strftime('%H:%M') for x in ixs], rotation=45)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('objective')


def featformat(s):
    return 'X__' + '_'.join(s.lower().split(' '))


def featdeformat(s):
    return s[len('X__'):].replace('_', ' ').replace('-', ' ')


def filter_trades_by_feature(trades, data, feature, min_value=None, max_value=None, exact_value=None, use_abs=False):
    # Create a copy of the trades DataFrame
    filtered_trades = trades.copy()

    # Get the relevant portion of the predictions indicator that corresponds to the trades
    relevant_predictions = data[feature].iloc[filtered_trades['entry_bar']]

    # Add the rescaled predictions as a new column to the trades DataFrame
    if use_abs:
        ft = abs(relevant_predictions.values)
    else:
        ft = relevant_predictions.values

    # Filter the trades by the prediction value
    if exact_value is not None:
        filtered_trades = filtered_trades.loc[ft == exact_value]
    else:
        # closed interval
        if (min_value is not None) and (max_value is not None):
            filtered_trades = filtered_trades.loc[(min_value <= ft) & (ft <= max_value)]
        else:
            # open intervals
            if (min_value is not None) and (max_value is None):
                filtered_trades = filtered_trades.loc[min_value <= ft]
            else:
                filtered_trades = filtered_trades.loc[ft <= max_value]

    return filtered_trades


#####################
# STRATEGIES
#####################

def train_classifier(clf_class, data):
    print('Training..')
    try:
        clf = clf_class(random_state=reseed())
    except:
        clf = clf_class()
    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X, y = get_clean_Xy(df)
    try:
        clf.fit(X, y)
    except:
        clf = LogisticRegression()
        clf.fit(X, y)
    print('Classifier trained.')
    return clf


def optimize_model(model, model_name, space, X_train, y_train, max_evals=120):
    def objective(params):
        model.set_params(**params)
        return -np.mean(cross_val_score(model, X_train, y_train, cv=8, scoring="accuracy"))

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    return best



def train_hpo_ensemble(data):
    print('Training..')

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X_train, y_train = get_clean_Xy(df)

    # Define classifiers and hyperparameter search spaces
    classifiers = [
        ("lr", LogisticRegression(), {"C": hp.loguniform("C", -5, 2)},
         50),
        ("knn", KNeighborsClassifier(), {"n_neighbors": hp.choice("n_neighbors", range(2, 101))},
         50),
        ("dt", DecisionTreeClassifier(), {"max_depth": hp.choice("max_depth", range(2, 21))},
         50),
        ("rf", RandomForestClassifier(), {"n_estimators": hp.choice("n_estimators", range(50, 201)),
                                          "max_depth": hp.choice("max_depth", range(2, 21))},
         10),
        ("gb", GradientBoostingClassifier(), {"n_estimators": hp.choice("n_estimators", range(50, 201)),
                                              "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                              "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
         {"n_estimators": hp.choice("n_estimators", range(50, 201)),
          "learning_rate": hp.loguniform("learning_rate", -5, 0), "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("lgbm", LGBMClassifier(), {"n_estimators": hp.choice("n_estimators", range(50, 201)),
                                    "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                    "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("catboost", CatBoostClassifier(verbose=False), {"n_estimators": hp.choice("n_estimators", range(50, 201)),
                                                         "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                                         "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
    ]

    optimized_classifiers = []

    for name, model, space, max_evals in classifiers:
        print(f"Optimizing {name}...")
        default_score = np.mean(cross_val_score(model, X_train, y_train, cv=8, scoring="accuracy"))
        best_hyperparams = optimize_model(model, name, space, X_train, y_train, max_evals=max_evals)
        mp = model.get_params()
        try:
            model.set_params(**best_hyperparams)
            optimized_score = np.mean(cross_val_score(model, X_train, y_train, cv=8, scoring="accuracy"))
        except:
            model.set_params(**mp)
            optimized_score = np.mean(cross_val_score(model, X_train, y_train, cv=8, scoring="accuracy"))
            best_hyperparams = mp

        print(
            f"{name}: Default score = {default_score:.4f}, Optimized score = {optimized_score:.4f}, Best hyperparameters = {best_hyperparams}")
        optimized_classifiers.append((name, model))

    # Create ensemble classifier
    ensemble = VotingClassifier(optimized_classifiers, voting="soft")

    # Train ensemble on training data
    ensemble.fit(X_train, y_train)
    print('Ensemble trained.')

    return ensemble



class MLClassifierStrategy:
    def __init__(self, cllf, feature_columns: list):
        # the sklearn classifier is already fitted to the data, we just store it here
        self.clf = cllf
        self.feature_columns = feature_columns

    def next(self, idx, data):
        # the current row is data[idx]

        # extract features for the current row
        features = data[self.feature_columns].iloc[idx-1].values.reshape(1, -1)

        # get the classifier prediction
        prediction = self.clf.predict(features)[0]

        # predicts buy
        if prediction >= 0.5:
            return 'buy'
        else:
            return 'sell'


def compute_stats(data, trades):
    if not isinstance(trades, pd.DataFrame):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades
    gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
    gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    return profit_factor, trades_df


def backtest_ml_strategy(strategy, data, skip_train=True, skip_val=False, skip_test=True,
                         commission=0.0, slippage=0.0, position_value=100000):
    equity_curve = np.zeros(len(data))
    trades = []
    current_profit = 0

    for idx in tqdm(range(1, len(data))):
        current_time = data.index[idx].time()
        if current_time < pd.Timestamp("09:30:00").time() or current_time > pd.Timestamp("16:00:00").time():
            # Skip trading in pre/aftermarket hours
            equity_curve[idx] = current_profit
            continue
        if (idx < int(train_set_end * len(data))) and skip_train:
            continue
        if (idx < int(val_set_end * len(data))) and skip_val:
            continue
        if (idx > int(val_set_end * len(data))) and skip_test:
            continue

        action = strategy.next(idx, data)

        entry_price = data.iloc[idx]['Open']
        exit_price = data.iloc[idx]['Close']

        shares = int(position_value / entry_price)

        if action == 'buy':
            profit = (exit_price - entry_price - slippage) * shares - commission
        elif action == 'sell':
            profit = (entry_price - exit_price - slippage) * shares - commission
        else:
            raise ValueError(f"Invalid action '{action}' at index {idx}")

        current_profit += profit
        equity_curve[idx] = current_profit
        trades.append({
            'pos': action,
            'shares': shares,
            'entry_datetime': data.index[idx],
            'exit_datetime': data.index[idx],
            'entry_bar': idx,
            'exit_bar': idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit
        })

    return equity_curve, *compute_stats(data, trades)

#
# class MLEnsembleStrategy(MLClassifierStrategy):
#     num_clfs = 100
#     dropout = 0.05
#
#     def init(self):
#         self.make_inds()
#
#         # Init the ensemble of classifiers
#         try:
#             self.clfs = [self.clf_class() for _ in range(self.num_clfs)]
#         except:
#             self.clfs = [LogisticRegression() for _ in range(self.num_clfs)]
#
#         # Train the classifiers in advance on the first N_TRAIN examples
#         df = self.data.df.iloc[:self.N_TRAIN]
#         X, y = get_clean_Xy(df)
#         for i,clf in enumerate(self.clfs):
#             pn = rnd.sample(list(range(len(X))), int(self.dropout * len(X)))
#             Xt = X.copy()[pn]
#             yt = y.copy()[pn]
#             try:
#                 clf.fit(Xt, yt)
#             except:
#                 self.clfs[i] = LogisticRegression()
#                 self.clfs[i].fit(Xt, yt)
#
#     def get_prediction(self):
#         # Forecast the next movement
#         X = get_X(self.data.df.iloc[-1:])
#         return np.mean([clf.predict(X)[0] for clf in self.clfs])
#
#
# class MLSingleParamStrategy(MLClassifierStrategy):
#     feature_name = None
#     # True means it will trade only when abs(move) > threshold
#     # False means it will trade only when move > threshold
#     take_abs = False
#
#
# class MLEnsembleParamStrategy(MLEnsembleStrategy):
#     feature_name = None
#     # True means it will trade only when abs(move) > threshold
#     # False means it will trade only when move > threshold
#     take_abs = False
#
#
# # useful for the parametric strategies
# def getv(self):
#     v = self.data.df[self.feature_name].iloc[-1:].values[0]
#     if self.take_abs: v = abs(v)
#     return v
#
#
# MLSingleParamStrategy.getv = getv
# MLEnsembleParamStrategy.getv = getv
#
#
# class MLSingleParamEqStrategy(MLSingleParamStrategy):
#     target = None
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.getv()
#             if v == self.target:
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLSingleParamTimeEqStrategy(MLSingleParamStrategy):
#     target = None
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.data.index[-1].time()
#             if v == self.target:
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLSingleParamOverUnderStrategy(MLSingleParamStrategy):
#     threshold = None
#     direction = 'above'  # or below
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.getv()
#             if ((self.direction == 'above') or (self.direction == 1)) and (v > self.threshold):
#                 self.act(self.get_prediction())
#             elif ((self.direction == 'below') or (self.direction == -1)) and (v < self.threshold):
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLEnsembleParamEqStrategy(MLEnsembleParamStrategy):
#     target = None
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.getv()
#             if v == self.target:
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLEnsembleParamTimeEqStrategy(MLEnsembleParamStrategy):
#     target = None
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.data.index[-1].time()
#             if v == self.target:
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLEnsembleParamOverUnderStrategy(MLEnsembleParamStrategy):
#     threshold = None
#     direction = 'above'  # or below
#
#     def next(self):
#         if not self.outofbounds():
#             v = self.getv()
#             if ((self.direction == 'above') or (self.direction == 1)) and (v > self.threshold):
#                 self.act(self.get_prediction())
#             elif ((self.direction == 'below') or (self.direction == -1)) and (v < self.threshold):
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()
#
#
# class MLSingleMultiParamStrategy(MLClassifierStrategy):
#     feature_names = None
#     take_abs = None
#
#
# def getmv(self):
#     # get all feature names
#     fnames = sorted([x for x in dir(self) if (x[0:3]=='X__')])
#     vs = []
#     for i, feature_name in enumerate(fnames):
#         v = self.data.df[feature_name.replace('_target_134','')].iloc[-1:].values[0]
#         # if self.take_abs[i]: v = abs(v)
#         vs.append(v)
#     return vs
#
# MLSingleMultiParamStrategy.getmv = getmv
#
# class MLSingleMultiParamEqStrategy(MLSingleMultiParamStrategy):
#     def next(self):
#         targets = sorted([x for x in dir(self) if (x[0:3] == 'X__')])
#
#         if not self.outofbounds():
#             vs = self.getmv()
#             if any([(x == getattr(self, y)) for x,y in zip(vs, targets)]):
#                 self.act(self.get_prediction())
#             else:
#                 self.position.close()
#         else:
#             self.position.close()




#####################
# DATA PROCEDURES
#####################

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
    print('Loading..', end=' ')
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    print('Done.')
    return data


def get_data_proc(symbol, period='D', nrows=None):
    print('Loading..', end=' ')
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    data = pd.read_csv(datadir + '/' + sfn + '_proc.csv', nrows=nrows, parse_dates=['time'], index_col=0)
    print('Done.')
    return data


def get_data_pair(symbol1, symbol2, period='D'):
    s1, s1_f = get_data(symbol1, period=period)
    s2, s2_f = get_data(symbol2, period=period)

    s1, s2 = fix_data(s1, s2)
    s1_f, s2_f = fix_data(s1_f, s2_f)

    return (s1 / s2).dropna(), (s1_f / s2_f).dropna()


def get_data_features(data):
    return list(data.columns[0:5]) + [featdeformat(x) for x in data.columns[5:]]


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
    y = data.Open.pct_change(1)
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
