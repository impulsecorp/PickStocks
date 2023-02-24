import random as rnd

import gym
import numpy as np
from empyrical import sortino_ratio, omega_ratio, sharpe_ratio
from gym import spaces
from gym.utils import seeding


class TradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(  # np.min(input_source, axis=0),
            # np.max(input_source, axis=0)
            np.ones((winlen * input_source.shape[1] + 2,)) * -15,
            np.ones((winlen * input_source.shape[1] + 2,)) * 15,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        # try:
        #    if len(action)>1: action = int(np.argmax(action))
        # except:
        #    pass

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((action == 2) and (self.position != 0)) or (
                (self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.datetimes[self.idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        elif action == 3:
            pass
        else:
            pass

        self.prev_balance = self.balance

        info = {}

        sret = 0
        if ret > 0:
            sret = 1
        elif ret < 0:
            sret = -1
        observation = np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                                 self.position, sret,
                                 ])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class MultiTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source_list, to_predict_list, datetimes_list, names=None,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(3 * len(input_source_list))
        self.observation_space = spaces.Box(
            np.ones((winlen * input_source_list[0].shape[1] * len(input_source_list) + 2 * len(
                input_source_list),)) * -1500000,
            np.ones((winlen * input_source_list[0].shape[1] * len(input_source_list) + 2 * len(
                input_source_list),)) * 1500000,
        )
        self.input_source_list = input_source_list
        self.to_predict_list = to_predict_list
        self.datetimes_list = datetimes_list
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        if names is not None: self.names = names
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, act):

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        comm_paid = 2 * self.commission

        rets = [0] * len(self.input_source_list)
        qtys = [0] * len(self.input_source_list)

        for i in range(len(self.input_source_list)):

            if self.positions[i] == -1:  # long
                qtys[i] = self.traded_amt / self.to_predict_list[i][self.idx]
                slip_paid = 2 * self.slippage * qtys[i]
                rets[i] = (self.to_predict_list[i][self.idx] - self.to_predict_list[i][self.open_idx[i]]) * qtys[
                    i] - comm_paid - slip_paid
            elif self.positions[i] == 1:  # short
                qtys[i] = self.traded_amt / self.to_predict_list[i][self.idx]
                slip_paid = 2 * self.slippage * qtys[i]
                rets[i] = (self.to_predict_list[i][self.open_idx[i]] - self.to_predict_list[i][self.idx]) * qtys[
                    i] - comm_paid - slip_paid

            if (act // 3) == i:
                action = act % 3
            else:
                action = 3

            # execute the action and get the reward
            if (action == 0) and (self.positions[i] == 0):  # buy

                self.positions[i] = -1
                self.open_idx[i] = self.idx

            elif (action == 1) and (self.positions[i] == 0):  # sell

                self.positions[i] = 1
                self.open_idx[i] = self.idx

            elif ((action == 2) and (self.positions[i] != 0)) or (
                    (self.positions[i] != 0) and ((self.idx - self.open_idx[i]) >= self.max_position_time)):  # close

                if self.positions[i] == -1:  # long

                    self.balance += rets[i]
                    self.returns.append(rets[i])

                elif self.positions[i] == 1:  # short

                    self.balance += rets[i]
                    self.returns.append(rets[i])

                # log trade
                self.trades += [(self.to_predict_list[i][self.open_idx[i]], self.to_predict_list[i][self.idx],
                                 self.positions[i], rets[i],
                                 self.datetimes_list[i][self.open_idx[i]],
                                 self.datetimes_list[i][self.idx],
                                 self.names[i] if hasattr(self, 'names') else None
                                 )]

                self.positions[i] = 0  # reset position to out of market

            elif action == 3:
                pass
            else:
                pass

        self.prev_balance = self.balance

        info = {}

        observation = np.hstack(
            [x[self.idx - self.winlen: self.idx, :].reshape(-1) for x in self.input_source_list] + [x for x in
                                                                                                    self.positions] + [x
                                                                                                                       for
                                                                                                                       x
                                                                                                                       in
                                                                                                                       rets])
        observation[np.isnan(observation)] = 0
        observation[np.isinf(observation)] = 0

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = sum(rets)
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source_list[0].shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.positions = [0] * len(self.input_source_list)
        self.open_idx = [0] * len(self.input_source_list)
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack(
            [x[self.idx - self.winlen: self.idx, :].reshape(-1) for x in self.input_source_list] + [0, 0] * len(
                self.input_source_list))

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.positions = [0] * len(self.input_source_list)
        self.open_idx = [0] * len(self.input_source_list)
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack(
            [x[self.idx - self.winlen: self.idx, :].reshape(-1) for x in self.input_source_list] + [0, 0] * len(
                self.input_source_list))

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class SwitchTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source_list, to_predict_list, datetimes_list,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(  # np.min(input_source, axis=0),
            # np.max(input_source, axis=0)
            np.ones((winlen * input_source_list[0].shape[1] + 2,)) * -15,
            np.ones((winlen * input_source_list[0].shape[1] + 2,)) * 15,
        )
        self.input_source_list = input_source_list
        self.to_predict_list = to_predict_list
        self.datetimes_list = datetimes_list

        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        # try:
        #    if len(action)>1: action = int(np.argmax(action))
        # except:
        #    pass

        comm_paid = 2 * self.commission
        slip_paid = 2 * self.slippage * self.traded_amt

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((action == 2) and (self.position != 0)) or (
                (self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.datetimes[self.idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        elif action == 3:
            pass
        else:
            pass

        self.prev_balance = self.balance

        info = {}

        observation = np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                                 self.position, ret,
                                 ])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        idx = rnd.choice(list(range(len(self.input_source_list))))
        self.input_source = self.input_source_list[idx]
        self.to_predict = self.to_predict_list[idx]
        self.datetimes = self.datetimes_list[idx]

        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def reset2(self):
        # reset and return first observation
        idx = rnd.choice(list(range(len(self.input_source_list))))
        self.input_source = self.input_source_list[idx]
        self.to_predict = self.to_predict_list[idx]
        self.datetimes = self.datetimes_list[idx]

        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class ConstrainedTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(  # np.min(input_source, axis=0),
            # np.max(input_source, axis=0)
            np.ones((winlen * input_source.shape[1] + 2,)) * -15,
            np.ones((winlen * input_source.shape[1] + 2,)) * 15,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        # try:
        #    if len(action)>1: action = int(np.argmax(action))
        # except:
        #    pass

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                # qty = self.traded_amt / self.to_predict[self.idx]
                # ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.datetimes[self.idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        self.prev_balance = self.balance

        info = {}

        observation = np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                                 self.position, ret,
                                 ])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class SellConstrainedTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(  # np.min(input_source, axis=0),
            # np.max(input_source, axis=0)
            np.ones((winlen * input_source.shape[1] + 2,)) * -15,
            np.ones((winlen * input_source.shape[1] + 2,)) * 15,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    def step(self, action):

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((action == 1) and (self.position != 0)) or (
                (self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.datetimes[self.idx],  # .strftime("%Y/%m/%d, %H:%M:%S"),
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        self.prev_balance = self.balance

        info = {}

        observation = np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                                 self.position, ret,
                                 ])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []
        return np.hstack([self.input_source[self.idx - self.winlen: self.idx, :].reshape(-1),
                          self.position, 0,
                          ])

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class MLTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes, clfs,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            np.ones((len(clfs) + 2,)) * 0.0,
            np.ones((len(clfs) + 2,)) * 1.0,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.clfs = clfs
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((action == 2) and (self.position != 0)) or (
                (self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],
                             self.datetimes[self.idx],
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        elif action == 3:
            pass
        else:
            pass

        self.prev_balance = self.balance

        info = {}

        sret = 0
        if ret > 0:
            sret = 1
        elif ret < 0:
            sret = -1

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, sret])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class FakeMLTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes, clfs,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            np.ones((len(clfs) + 2,)) * 0.0,
            np.ones((len(clfs) + 2,)) * 1.0,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.clfs = clfs
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((action == 2) and (self.position != 0)) or (
                (self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],
                             self.datetimes[self.idx],
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        elif action == 3:
            pass
        else:
            pass

        self.prev_balance = self.balance

        info = {}

        sret = 0
        if ret > 0:
            sret = 1
        elif ret < 0:
            sret = -1

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [rnd.choice([0.0, 1.0]) for clf in self.clfs]
        observation = np.hstack([preds, self.position, sret])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [rnd.choice([0.0, 1.0]) for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [rnd.choice([0.0, 1.0]) for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def _render(self, mode='human', close=False):
        # ... TODO
        pass


class BuySellMLTradingEnv(gym.Env):
    """ This gym implements a simple trading environment for reinforcement learning. """

    metadata = {'render.modes': ['human']}

    def __init__(self, input_source, to_predict, datetimes, clfs,
                 winlen=1, bars_per_episode=1000, traded_amt=10000, initial_balance=10000,
                 commission=0, slippage=0,
                 reward_type='cur_balance',  # 'balance', 'cur_balance', 'sortino'
                 max_position_time=3,
                 min_ratio_trades=8,
                 ):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            np.ones((len(clfs) + 2,)) * 0.0,
            np.ones((len(clfs) + 2,)) * 1.0,
        )
        self.input_source = input_source
        self.to_predict = to_predict
        self.datetimes = datetimes
        self.clfs = clfs
        self.winlen = winlen
        self.bars_per_episode = bars_per_episode
        self.traded_amt = traded_amt
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.initial_balance = initial_balance
        self.max_position_time = max_position_time
        self.min_ratio_trades = min_ratio_trades
        self.trades = []
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        return self.step(action)

    def _reset(self):
        return self.reset()

    # @jit
    def step(self, action):

        if (self.idx < self.end_idx) and (self.balance > 0):
            self.idx += 1
            done = False
        else:
            done = True

        comm_paid = 2 * self.commission

        ret = 0
        qty = 0
        if self.position == -1:  # long
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.idx] - self.to_predict[self.open_idx]) * qty - comm_paid - slip_paid
        elif self.position == 1:  # short
            qty = self.traded_amt / self.to_predict[self.idx]
            slip_paid = 2 * self.slippage * qty
            ret = (self.to_predict[self.open_idx] - self.to_predict[self.idx]) * qty - comm_paid - slip_paid

        # execute the action and get the reward
        if (action == 0) and (self.position == 0):  # buy

            self.position = -1
            self.open_idx = self.idx

        elif (action == 1) and (self.position == 0):  # sell

            self.position = 1
            self.open_idx = self.idx

        elif ((self.position != 0) and ((self.idx - self.open_idx) >= self.max_position_time)):  # close

            if self.position == -1:  # long

                self.balance += ret
                self.returns.append(ret)

            elif self.position == 1:  # short

                self.balance += ret
                self.returns.append(ret)

            # log trade
            self.trades += [(self.to_predict[self.open_idx], self.to_predict[self.idx], self.position, ret,
                             self.datetimes[self.open_idx],
                             self.datetimes[self.idx],
                             self.name if hasattr(self, 'name') else None
                             )]

            self.position = 0  # reset position to out of market

        else:
            pass

        self.prev_balance = self.balance

        info = {}

        sret = 0
        if ret > 0:
            sret = 1
        elif ret < 0:
            sret = -1

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, sret])

        if self.reward_type == 'sortino':

            if len(self.returns) > self.min_ratio_trades:
                reward = sortino_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'sharpe':

            if len(self.returns) > self.min_ratio_trades:
                reward = sharpe_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'omega':

            if len(self.returns) > self.min_ratio_trades:
                reward = omega_ratio(np.array(self.returns[-self.min_ratio_trades:]))
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0
            else:
                reward = 0
        elif self.reward_type == 'cur_balance':
            reward = ret
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
        elif self.reward_type == 'balance':
            if len(self.returns) > 0:
                reward = np.sum(self.returns)  # self.balance
            else:
                reward = 0
        elif self.reward_type == 'rel_balance':
            if len(self.returns) > self.min_ratio_trades:
                reward = np.sum(self.returns[-self.min_ratio_trades:])  # self.balance
            else:
                reward = 0
        else:
            reward = 0

        # reward = reward * len(self.returns)

        return observation, reward, done, info

    def reset(self):
        # reset and return first observation
        self.idx = np.random.randint(self.winlen, self.input_source.shape[0] - self.bars_per_episode)
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def reset2(self):
        # reset and return first observation
        self.idx = self.winlen
        self.end_idx = self.idx + self.bars_per_episode
        self.position = 0
        self.open_idx = 0
        self.balance = self.initial_balance
        self.prev_balance = self.balance
        self.returns = []
        self.trades = []

        inp = self.input_source[self.idx - self.winlen: self.idx, :].reshape(1, -1)
        preds = [clf.predict(inp)[0] for clf in self.clfs]
        observation = np.hstack([preds, self.position, 0])

        return observation

    def _render(self, mode='human', close=False):
        # ... TODO
        pass
