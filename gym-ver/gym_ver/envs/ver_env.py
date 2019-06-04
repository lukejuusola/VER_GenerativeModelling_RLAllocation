import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import Box2D
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


class queue:
  def __init__(self, maxsize):
    self.maxsize = maxsize
    self.q = []
    self.size = 0
    for i in range(maxsize):
      self.enqueue(0.)

  def dequeue(self):
    front = None
    if self.size > 0:
      front = self.q[0]
      self.q = self.q[1:]
      self.size -= 1
    return front

  def enqueue(self, to_add):
    if self.size >= self.maxsize:
      self.dequeue()
    self.q += [to_add]
    self.size += 1

  def tolist(self):
    return self.q


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


class cyclic_network(network):
  def __init__(self, n, mean = 10, amp = 5, shift = 0):
    self.n = n
    self.mean = mean
    self.amp = amp
    self.shift = shift
    # resolution timesteps/cycle
    res = 24
    self.cur_t = 0
    self.timestep = 2 * np.pi / res

  def generate(self):
    noise = np.random.normal(size = self.n)
    sample_mean = self.amp * np.sin(self.cur_t + self.shift) + self.mean
    self.cur_t += self.timestep
    samples = sample_mean + noise
    samples = np.maximum(samples, 0)
    return np.sum(samples)

  def reset(self):
    self.cur_t = 0


class data_network(network):
  def __init__(self, n, file_list):
    self.n = n
    # n = 1 first
    assert n == 1
    self.cur_file = [] # Current file that we are pulling data from


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
    self.memory_size = 3
    self.state_size = 2 * self.memory_size + 1
    self.observation_space = spaces.Box(low = np.zeros(self.state_size), high = np.array([np.inf] * self.state_size), dtype = np.float32)

    ### These will be neural network generated ###
    # Use 'generated' to use generated data, or 'real' to use original training
    # dataset
    self.ver = solar_network(1, 'generated')
    # self.ver = uniform_network(1)
    # self.ver = cyclic_network(1, mean=9, amp=4, shift=.5)

    # Use 'generated' to use generated data, or 'real' to use original training
    # dataset
    self.load = load_network(1, 'real')
    # self.load = uniform_network(1)

    # Scale load so that mean(solar_generation) = 0.75 * mean(load)
    self._scale_load()

    self.storage = Storage(150)

    # Want to give the agent some historical information.
    # Could make it really fancy with RNNs and what not, but this should likely suffice
    self.hist_ver = queue(self.memory_size)
    self.hist_load = queue(self.memory_size)
    self.cost_to_generate = 1
    self.cost_blackout = 1000
    self.action_cost = lambda x: self.cost_to_generate * x

  def _scale_load(self):
    mean_load = np.mean(self.load.X)
    mean_ver = np.mean(self.ver.X)

    # Scale load to have mean(ver) = 0.75 * mean(load)
    self.load.X = self.load.X * (mean_ver / mean_load) * (4. / 3.)

  def get_state(self):
    state = [self.storage.stored] + self.hist_ver.tolist() + self.hist_load.tolist()
    return np.array(state)

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
    self.hist_ver.enqueue(cur_load)

    # get current ver generation
    cur_ver = self.ver.generate()
    self.hist_ver.enqueue(cur_ver)


    # Generate total cost
    # cost is price per created unit. Large drawback if load greater than
    # current energy, i.e., blackout
    cost = self.action_cost(action)

    # update state
    blackout_flag = 0
    cur_demand = cur_load - cur_ver
    # fully covered by ver, we store extra ver and action
    if cur_ver > cur_load:
      extra_nrg = action - cur_demand
      self.storage.store(extra_nrg)
    # not fully covered
    else:
      # remainder is covered by action, store what's left
      if action > cur_demand:
        extra_nrg = action - cur_demand
        self.storage.store(extra_nrg)
      # need to dip into storage
      else:
        needed_from_storage = cur_demand - action

        taken = self.storage.take(needed_from_storage)
        # stoarge did not meet requirements
        if taken < needed_from_storage:
          # Blackout!
          blackout_flag = 1

    #assert cost >= 0, cost
    reward = - blackout_flag * self.cost_blackout - action + cur_demand
    # Maximize cur_demand - action without blacking out
    # I.e., try to match action exactly. Further penalize wasting energy over capacity?
    # return obs, reward, done, info
    return self.get_state(), reward, False, None

  def reset(self):
    self.storage.reset()
    self.ver.reset()
    self.load.reset()
    return self.get_state()

  def render(self, mode='human', close=False):
    print('rEnDeRiNg:', self.storage.stored)
