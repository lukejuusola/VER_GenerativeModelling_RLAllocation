import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import pdb
import csv


class Storage:
  def __init__(self, cap):
    self.capacity = cap
    self.stored = 0

  def store(self, amount):
    self.stored += amount
    self.stored = min(self.stored, self.capacity)

  def take(self, amount):
    if amount > self.stored:
      temp = self.stored
      self.stored = 0
      return temp
    else:
      self.stored -= amount
      return amount

  def reset(self):
    self.stored = 0



class network:
  def __init__(self, n, data_type):
    self.n = n

    # Type of data to load (real or generated).
    self.data_type = data_type
    # Load 10000 x 2days of generated solar generation examples
    self.load_data()
    # Create a private generator object to generate next sample
    self._ver_generator = self._data_generator(self.X)

  def load_data(self):
    raise NotImplementedError()

  def _data_generator(self, X):
    while True:
      # Select a random example from dataset
      idx = np.random.randint(0, X.shape[0])
      sample = X[idx, :]
      # Yield each value of the random chosen, as a time-series
      for value in sample:
        yield value

  def generate(self):
    # Generate new ver generation using generator object
    return next(self._ver_generator)

  def reset(self):
    self._ver_generator = self._data_generator(self.X)

class uniform_network(network):
  def generate(self):
    samples = np.random.uniform(size = self.n)
    #samples = np.random.normal(size = self.n)
    #samples = np.vstack((np.zeros(self.n), samples))
    #samples = np.apply_along_axis(np.max, 0, samples)
    assert np.all(samples >= 0)
    return np.sum(samples)

  def reset(self):
    return

class normal_network(network):
  def generate(self):
    samples = np.random.normal(size = self.n)
    samples = np.vstack((np.zeros(self.n), samples))
    samples = np.apply_along_axis(np.max, 0, samples)
    assert np.all(samples >= 0)
    return np.sum(samples)

  def reset(self):
    return

class solar_network(network):
  def __init__(self, n, data='real'):
    super().__init__(n, data)

  def load_data(self):

    if self.data_type == 'generated':
      # Pick 1/100 data file at random
      file_idx = np.random.randint(0, 100)
      self.file_path = f'/Volumes/Eluteng/cs159/gen_solar_data' \
                       f'/solar_random_labels_{file_idx}.csv'

      # Load all generated examples stored in a single file (10000 x 2 days)
      with open(self.file_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
      self.X = np.array(rows, dtype=float)

    elif self.data_type == 'real':
      self.file_path = 'generative_models/data/solar_2006.csv'

      # Load training examples
      with open(self.file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
      rows = np.array(rows, dtype=float)
      self.X = np.reshape(rows, (-1, 576))

    else:
      raise ValueError('Type of data to load not understood.')


class load_network(network):
  def __init__(self, n, data='real'):
    super().__init__(n, data)

  def load_data(self):
    if self.data_type == 'generated':
      raise ValueError('Load data has not been generated yet.')

    elif self.data_type == 'real':
      self.file_path = 'generative_models/data/load_2017_2days.csv'

      # Load training examples
      with open(self.file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
      rows = np.array(rows, dtype=float)
      self.X = np.reshape(rows, (-1, 576))

    else:
      raise ValueError('Type of data to load not understood.')


class VerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    high = 100
    self.action_space = spaces.Box(low = np.array([0.]), high = np.array([high]), dtype = np.float32)
    self.observation_space = spaces.Box(low = np.zeros(3), high = np.array([np.inf] * 3), dtype = np.float32)

    ### These will be neural network generated ###

    # Use 'generated' to use generated data, or 'real' to use original training
    # dataset
    self.ver = solar_network(1, 'generated')
    #self.ver = uniform_network(1)

    # Use 'generated' to use generated data, or 'real' to use original training
    # dataset
    self.load = load_network(1, 'real')
    #self.load = uniform_network(1)

    # Scale load so that mean(solar_generation) = 0.75 * mean(load)
    self._scale_load()

    self.storage = Storage(150)

    # Want to give the agent some historical information.
    # Could make it really fancy with RNNs and what not, but this should likely suffice
    self.last_ver_production = None
    self.last_load_production = None
    self.cost_to_generate = 1
    self.cost_blackout = 1000
    self.action_cost = lambda x: self.cost_to_generate * x

  def _scale_load(self):
    mean_load = np.mean(self.load.X)
    mean_ver = np.mean(self.ver.X)

    # Scale load to have mean(ver) = 0.75 * mean(load)
    self.load.X = self.load.X * (mean_ver / mean_load) * (4. / 3.)

  def get_state(self):
    return np.array([self.storage.stored, self.last_ver_production, self.last_load_production])

  def set_costs(self, generation, blackout):
    self.cost_to_generate = generation
    self.cost_blackout = blackout


  def step(self, action):
    if type(action) is np.ndarray:
      action = action[0]
    assert action >= 0, action
    #pdb.set_trace()
    # get current load
    cur_load = self.load.generate()
    self.last_load_production = cur_load

    # get current ver generation
    cur_ver = self.ver.generate()
    self.last_ver_production = cur_ver

    # Generate total cost
    # cost is price per created unit. Large drawback if load greater than
    # current energy, i.e., blackout
    cost = self.action_cost(action)

    # update state
    # fully covered by ver, we store extra ver and action
    if cur_ver > cur_load:
      extra_nrg = cur_ver - cur_load + action
      self.storage.store(extra_nrg)
    # not fully covered
    else:
      remaining_load = cur_load - cur_ver
      # remainder is covered by action, store what's left
      if action > remaining_load:
        extra_nrg = action - remaining_load
        self.storage.store(extra_nrg)
      # need to dip into storage
      else:
        needed_from_storage = remaining_load - action
        taken = self.storage.take(needed_from_storage)
        # stoarge did not meet requirements
        if taken < needed_from_storage:
          # Blackout!
          cost += self.cost_blackout

    assert cost >= 0, cost
    reward = -cost
    # return obs, reward, done, info
    return self.get_state(), reward, False, None

  def reset(self):
    self.storage.reset()
    self.ver.reset()
    self.load.reset()
    return self.get_state()

  def render(self, mode='human', close=False):
    print('rEnDeRiNg:', self.storage.stored)
