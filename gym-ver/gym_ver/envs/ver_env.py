import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import Box2D
import pdb

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

class Storage:
  def __init__(self, cap):
    self.capacity = cap
    self.stored = 0

  def store(self, amount):
    self.stored += amount
    self.stored = min(self.stored, self.capacity)

  def take(self, amount):
    if amount > self.stored:
      self.stored = 0
      return self.stored
    else:
      self.stored -= amount
      return amount

  def reset(self):
    self.stored = 0

    

class network:
  def __init__(self, n):
    self.n = n

  def generate(self):
    raise NotImplementedError()

  def reset(self):
    raise NotImplementedError()
  
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
    

def data_network(network):
  def __init__(self, n, file_list):
    self.n = n
    # n = 1 first
    assert n == 1
    self.cur_file = [] # Current file that we are pulling data from

    
class VerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    high = 100
    self.action_space = spaces.Box(low = np.array([0.]), high = np.array([high]), dtype = np.float32)
    self.memory_size = 3
    self.state_size = 2 * self.memory_size + 1
    self.observation_space = spaces.Box(low = np.zeros(self.state_size), high = np.array([np.inf] * self.state_size), dtype = np.float32)
    # These will be neural network generated
    self.ver = cyclic_network(1, mean = 9, amp = 4, shift = .5)
    self.load = cyclic_network(1)
    self.storage = Storage(10)
    
    # Want to give the agent some historical information.
    # Could make it really fancy with RNNs and what not, but this should likely suffice
    self.hist_ver = queue(self.memory_size)
    self.hist_load = queue(self.memory_size)
    self.cost_to_generate = 1
    self.cost_blackout = 1000
    self.action_cost = lambda x: self.cost_to_generate * x

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
