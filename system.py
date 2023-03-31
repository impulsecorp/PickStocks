import datetime
import os

import warnings

warnings.filterwarnings('ignore')
import random as rnd
import time as stime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num
from matplotlib.pyplot import gca
from matplotlib.pyplot import plot
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from tqdm.notebook import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from hyperopt import fmin, hp, rand
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
from imblearn.over_sampling import SMOTE


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


def newseed():
    seed = 0
    while seed == 0:
        seed = int(stime.time() * 100000) % 1000000
    return seed


seed = reseed()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from datetime import datetime, time

# global parameters

train_set_end = 0.4  # percentage point specifying the training set end point (1.0 means all data is training set)
val_set_end = 0.7  # percentage point specifying the validation set end point (1.0 means no test set and all data after the previous point is validation )
max_tries = 0.2  # for optimization, percentage of the grid space to cover (1.0 = exchaustive search)
cv_folds = 5
balance_data = 1
multiclass = 0
multiclass_move_threshold = 0.2
regression = 0


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
        bt.plot(plot_width=1200, plot_volume=False, plot_pl=1, resample=False)
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


def filter_trades_by_feature(the_trades, data, feature, min_value=None, max_value=None, exact_value=None,
                             use_abs=False):
    # Create a copy of the trades DataFrame
    filtered_trades = the_trades.copy()

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
            if min_value == max_value:
                filtered_trades = filtered_trades.loc[ft == min_value]
            else:
                min_value, max_value = np.min([min_value, max_value]), np.max([min_value, max_value])
                filtered_trades = filtered_trades.loc[(min_value <= ft) & (ft <= max_value)]
        else:
            # open intervals
            if (min_value is not None) and (max_value is None):
                filtered_trades = filtered_trades.loc[min_value <= ft]
            else:
                filtered_trades = filtered_trades.loc[ft <= max_value]

    return filtered_trades


def filter_trades_by_confidence(the_trades, min_conf=None, max_conf=None):
    trs = the_trades.copy()
    if (min_conf is None) and (max_conf is None):
        return trs
    elif (min_conf is not None) and (max_conf is None):
        return trs.loc[(np.abs(0.5 - trs['pred'].values) * 2.0) >= min_conf]
    elif (min_conf is None) and (max_conf is not None):
        return trs.loc[(np.abs(0.5 - trs['pred'].values) * 2.0) <= max_conf]
    else:
        return trs.loc[((np.abs(0.5 - trs['pred'].values) * 2.0) >= min_conf) & (
                (np.abs(0.5 - trs['pred'].values) * 2.0) <= max_conf)]


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMBinaryClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # only take the last output of the sequence
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


class PyTorchLSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, batch_size=32, learning_rate=1e-3, n_epochs=50, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.scaler = StandardScaler()
        self.model = LSTMBinaryClassifier(input_dim, hidden_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X = X.reshape(-1, 1, self.input_dim)  # reshape input data to (batch_size, sequence_length, input_dim)
        X_train_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            if epoch % 10 == 0: print(f'Epochs: {epoch}/{self.n_epochs}')
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X = X.reshape(-1, 1, self.input_dim)  # reshape input data to (batch_size, sequence_length, input_dim)
        X_test_tensor = torch.tensor(X, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor).cpu().numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# Define the 4-layer feed-forward neural network

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            SelfAttention(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Define the PyTorch wrapper to behave like an sklearn classifier
class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, batch_size=32, learning_rate=1e-3, n_epochs=50, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.scaler = StandardScaler()
        self.model = BinaryClassifier(input_dim, hidden_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_train_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                # Compute accuracy
                predicted = torch.round(outputs)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            epoch_acc = correct / total
            epoch_loss /= len(train_loader)

            print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        return self

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X_test_tensor = torch.tensor(X, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor).cpu().numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class SymbolicRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            # population_size=1000,
            # generations=20,
            # tournament_size=20,
            # stopping_criteria=0.0,
            # const_range=(-1.0, 1.0),
            # init_depth=(2, 6),
            # init_method="half and half",
            # function_set=("add", "sub", "mul", "div"),
            # metric="mean absolute error",
            # parsimony_coefficient=0.001,
            # p_crossover=0.9,
            # p_subtree_mutation=0.01,
            # p_hoist_mutation=0.01,
            # p_point_mutation=0.01,
            # p_point_replace=0.05,
            # max_samples=1.0,
            # feature_names=None,
            # warm_start=False,
            # low_memory=False,
            # n_jobs=1,
            # verbose=0,
            # random_state=None
    ):
        self.scaler = StandardScaler()
        self.model = SymbolicRegressor(
            # population_size=population_size,
            # generations=generations,
            # tournament_size=tournament_size,
            # stopping_criteria=stopping_criteria,
            # const_range=const_range,
            # init_depth=init_depth,
            # init_method=init_method,
            # function_set=function_set,
            # metric=metric,
            # parsimony_coefficient=parsimony_coefficient,
            # p_crossover=p_crossover,
            # p_subtree_mutation=p_subtree_mutation,
            # p_hoist_mutation=p_hoist_mutation,
            # p_point_mutation=p_point_mutation,
            # p_point_replace=p_point_replace,
            # max_samples=max_samples,
            # feature_names=feature_names,
            # warm_start=warm_start,
            # low_memory=low_memory,
            # n_jobs=n_jobs,
            # verbose=verbose,
            # random_state=random_state
        )

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype(int)


#####################
# STRATEGIES
#####################


def optimize_model(model, model_name, space, X_train, y_train, max_evals=120):
    defaults = model.get_params()

    def objective(params):
        try:
            model.set_params(random_state=newseed(), **params)
            score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy"))
            return score
        except:
            return 9999999.0

    best = fmin(fn=objective, space=space, algo=rand.suggest, max_evals=max_evals)

    # if we can't instantiate and train the model, use the defaults
    try:
        try:
            model.set_params(random_state=newseed(), **best)
        except:
            model.set_params(**best)
        model.fit(X_train[0:15], y_train[0:15])
    except:
        print('No better parameters than the defaults were found.')
        model.set_params(**defaults)
        best = defaults

    return best


def train_hpo_ensemble(data):
    print('Training..')

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X_train, y_train = get_clean_Xy(df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if balance_data:
        # Apply SMOTE oversampling to balance the training data
        sm = SMOTE(random_state=newseed())
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Define classifiers and hyperparameter search spaces
    classifiers = [
        ("lr", LogisticRegression(), {"C": hp.loguniform("C", -5, 2),
                                      "max_iter": hp.choice("max_iter", range(5, 501)),
                                      "dual": hp.choice("dual", [True, False]),
                                      "fit_intercept": hp.choice("fit_intercept", [True, False])},
         150),
        ("knn", KNeighborsClassifier(), {"n_neighbors": hp.choice("n_neighbors", range(2, 101))},
         50),
        ("dt", DecisionTreeClassifier(), {"max_depth": hp.choice("max_depth", range(2, 21))},
         50),
        ("rf", RandomForestClassifier(), {"n_estimators": hp.choice("n_estimators", range(5, 201)),
                                          "max_depth": hp.choice("max_depth", range(2, 21))},
         10),
        ("gb", GradientBoostingClassifier(), {"n_estimators": hp.choice("n_estimators", range(5, 201)),
                                              "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                              "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
         {"n_estimators": hp.choice("n_estimators", range(5, 201)),
          "learning_rate": hp.loguniform("learning_rate", -5, 0), "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("lgbm", LGBMClassifier(), {"n_estimators": hp.choice("n_estimators", range(5, 201)),
                                    "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                    "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
        ("catboost", CatBoostClassifier(verbose=False), {"n_estimators": hp.choice("n_estimators", range(5, 201)),
                                                         "learning_rate": hp.loguniform("learning_rate", -5, 0),
                                                         "max_depth": hp.choice("max_depth", range(2, 11))},
         3),
    ]

    optimized_classifiers = []

    for name, model, space, max_evals in classifiers:
        print(f"Optimizing {name}...")
        default_score = np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy"))
        def_params = model.get_params()
        best_hyperparams = optimize_model(model, name, space, X_train, y_train, max_evals=max_evals)
        try:
            model.set_params(**best_hyperparams)
            model.fit(X_train, y_train)
            optimized_score = np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy"))
        except:
            print('Problematic config found, reverting to default parameters.')
            try:
                model.set_params(**def_params)
                optimized_score = np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy"))
            except:
                pass
            best_hyperparams = def_params
        print(
            f"{name}: Default score = {default_score:.4f}, Optimized score = {optimized_score:.4f}, Best hyperparameters = {best_hyperparams}")
        optimized_classifiers.append((name, model))

    ensemble = VotingClassifier(optimized_classifiers, voting="soft")
    # Train ensemble on training data
    ensemble.fit(X_train, y_train)
    print(
        f'Ensemble trained. Mean CV score: {np.mean(cross_val_score(ensemble, X_train, y_train, cv=cv_folds, scoring="accuracy")):.5f}')

    return ensemble, scaler


def train_ensemble(clf_class, data, ensemble_size=100, max_samples=0.8, max_features=0.8, **kwargs):
    clfs = []
    print(f'Training ensemble: {ensemble_size} classifiers of type {clf_class.__name__.split(".")[-1]}... ', end=' ')
    for i in range(ensemble_size):
        try:
            clf = clf_class(random_state=newseed(), **kwargs)
        except:
            clf = clf_class(**kwargs)
        clfs.append((f'clf_{i}', clf))
    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X_train, y_train = get_clean_Xy(df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if balance_data:
        # Apply SMOTE oversampling to balance the training data
        sm = SMOTE(random_state=newseed())
        X_train, y_train = sm.fit_resample(X_train, y_train)
    # Create ensemble classifier
    ensemble = BaggingClassifier(estimator=clf, n_estimators=ensemble_size,
                                 max_samples=max_samples, max_features=max_features,
                                 oob_score=True, random_state=newseed(), n_jobs=-1)
    # Train ensemble on training data
    ensemble.fit(X_train, y_train)
    print(
        f'Done. Mean CV score: {np.mean(cross_val_score(ensemble, X_train, y_train, cv=cv_folds, scoring="accuracy")):.5f}')
    return ensemble, scaler


def train_classifier(clf_class, data, **kwargs):
    print('Training', clf_class.__name__.split('.')[-1], '...', end=' ')

    clf = clf_class(**kwargs)

    N_TRAIN = int(data.shape[0] * train_set_end)
    df = data.iloc[0:N_TRAIN]
    X, y = get_clean_Xy(df)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    if balance_data:
        # Apply SMOTE oversampling to balance the training data
        sm = SMOTE(random_state=newseed())
        Xt, y = sm.fit_resample(Xt, y)

    print('Data collected.')
    print('Class 0 (up):', len(y[y == 0]))
    print('Class 1 (down):', len(y[y == 1]))
    print('Class 2 (none):', len(y[y == 2]))

    clf.fit(Xt, y)
    # print(f'Done. Mean CV score: {np.mean(cross_val_score(clf, Xt, y, cv=cv_folds, scoring="accuracy")):.5f}')
    return clf, scaler


# MultiClass variant
class MLClassifierStrategy:
    def __init__(self, clf, feature_columns, scaler, min_confidence=0.0, reverse=False):
        # the sklearn classifier is already fitted to the data, we just store it here
        self.clf = clf
        self.feature_columns = feature_columns
        self.min_confidence = min_confidence
        self.scaler = scaler
        self.reverse = reverse

    def next(self, idx, data):
        if not hasattr(self, 'datafeats'):
            self.datafeats = data[self.feature_columns].values

        # the current row is data[idx]
        # extract features for the previous row
        features = self.scaler.transform(self.datafeats[idx].reshape(1, -1))

        # get the classifier prediction
        try:
            try:
                prediction_proba = self.clf.predict_proba(features)
            except:
                # AutoGluon prediction fix
                prediction_proba = self.clf.predict_proba(pd.DataFrame(features)).values

        except AttributeError:
            try:
                prediction = self.clf.predict(data[self.feature_columns].iloc[idx])[0]
            except:
                prediction = self.clf.predict(data[self.feature_columns].iloc[idx].values.reshape(1, -1))[0]

            prediction_proba = None

        if prediction_proba is not None:
            class_label = np.argmax(prediction_proba)
            conf = np.max(prediction_proba)
        else:
            class_label = prediction
            conf = np.abs(0.5 - prediction) * 2.0

        if conf > self.min_confidence:
            if not self.reverse:
                if class_label == 0:
                    return 'buy', conf
                elif class_label == 1:
                    return 'sell', conf
                elif class_label == 2:
                    return 'none', conf
            else:
                if class_label == 0:
                    return 'sell', conf
                elif class_label == 1:
                    return 'buy', conf
                elif class_label == 2:
                    return 'none', conf
        else:
            return 'none', conf


market_start_time = pd.Timestamp("09:30:00").time()
market_end_time = pd.Timestamp("16:00:00").time()


def backtest_ml_strategy(strategy, data, skip_train=1, skip_val=0, skip_test=1,
                         commission=0.0, slippage=0.0, position_value=100000):
    equity_curve = np.zeros(len(data))
    trades = []
    current_profit = 0

    for idx in tqdm(range(1, len(data))):
        current_time = data.index[idx].time()
        if not data.daily:
            if (current_time < market_start_time) or (current_time > market_end_time):
                # Skip trading in pre/aftermarket hours
                equity_curve[idx] = current_profit
                continue
        if (idx < int(train_set_end * len(data))) and skip_train:
            continue
        if (idx < int(val_set_end * len(data))) and skip_val:
            continue
        if (idx > int(val_set_end * len(data))) and skip_test:
            continue

        action, prediction = strategy.next(idx, data)

        entry_price = data.iloc[idx]['Open']
        exit_price = data.iloc[idx]['Close']

        shares = int(position_value / entry_price)

        if action == 'buy':
            profit = (exit_price - entry_price - slippage) * shares - commission
        elif action == 'sell':
            profit = (entry_price - exit_price - slippage) * shares - commission
        elif action == 'none':
            profit = 0.0
        else:
            raise ValueError(f"Invalid action '{action}' at index {idx}")

        current_profit += profit
        equity_curve[idx] = current_profit
        if action != 'none':
            trades.append({
                'pos': action,
                'pred': prediction,
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


def get_winner_pct(trades):
    atrades = trades.copy()
    if len(atrades) > 0:
        winners = (len(atrades.loc[atrades['profit'].values >= 0.0]) / len(atrades)) * 100.0
    else:
        winners = -1.0
    return winners


def get_profit_factor(trades):
    atrades = trades.copy()
    gross_profit = atrades[atrades['profit'] >= 0]['profit'].sum()
    gross_loss = np.abs(atrades[atrades['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else -1
    return profit_factor


def compute_stats(data, trades):
    if not isinstance(trades, pd.DataFrame):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades.copy()
    try:
        return get_profit_factor(trades_df), trades_df
    except:
        return -1, pd.DataFrame(columns=['pos', 'shares', 'entry_datetime', 'exit_datetime', 'entry_bar',
                                         'exit_bar', 'entry_price', 'exit_price', 'profit'])


def qbacktest(clf, scaler, data, quiet=0, reverse=False, **kwargs):
    s = MLClassifierStrategy(clf, list(data.filter(like='X')), scaler, reverse=reverse)
    equity, pf, trades = backtest_ml_strategy(s, data, **kwargs)
    if not quiet:
        plt.plot(equity)
        plt.xlabel('Bar #')
        plt.ylabel('Profit')
        print(f'Profit factor: {pf:.5f}, Winners: {get_winner_pct(trades):.2f}%, Trades: {len(trades)}')
    return equity, pf, trades


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
    if period != 'D':
        data = pd.read_csv(datadir + '/' + sfn + '.csv', nrows=nrows, parse_dates=['time'], index_col=0)
        data.daily = 0
    else:
        data = pd.read_csv(datadir + '/' + sfn + '.csv', nrows=nrows, parse_dates=['date'], index_col=0)
        data.daily = 1
    print('Done.')
    return data


def get_data_proc(symbol, period='D', nrows=None):
    print('Loading..', end=' ')
    if period == 'd': period = 'D'
    sfn = symbol + '_' + period
    if period != 'D':
        data = pd.read_csv(datadir + '/' + sfn + '_proc.csv', nrows=nrows, parse_dates=['time'], index_col=0)
        data.daily = 0
    else:
        data = pd.read_csv(datadir + '/' + sfn + '_proc.csv', nrows=nrows, parse_dates=['date'], index_col=0)
        data.daily = 1
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


def get_X(data):
    """Return matrix X"""
    return data.filter(like='X').values


def get_y(data):
    """ Return dependent variable y """
    if regression:
        y = (data.Close.shift(-1) - data.Open.shift(-1)).astype(np.float32)
        return y
    else:
        if not multiclass:
            y = ((data.Close.shift(-1) - data.Open.shift(-1)) >= 0).astype(np.float32)
            return y
        else:
            move = (data.Close.shift(-1) - data.Open.shift(-1)).astype(np.float32)

            y = np.zeros_like(move, dtype=np.int32)

            y[move >= multiclass_move_threshold] = 0  # Class 0: 'buy'
            y[move <= -multiclass_move_threshold] = 1  # Class 1: 'sell'
            y[np.abs(move) < multiclass_move_threshold] = 2  # Class 2: 'do nothing'

            return y


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    try:
        y = get_y(df).values
    except:
        y = get_y(df)
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]

    return X, y


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
