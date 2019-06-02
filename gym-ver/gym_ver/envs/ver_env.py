import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import Box2D
import pdb


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

  
class VerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    high = 100
    self.action_space = spaces.Box(low = np.array([0.]), high = np.array([high]), dtype = np.float32)
    self.observation_space = spaces.Box(low = np.zeros(3), high = np.array([np.inf] * 3), dtype = np.float32)
    # These will be neural network generated
    self.ver = uniform_network(1)
    self.load = uniform_network(1)
    self.storage = Storage(10)
    
    # Want to give the agent some historical information.
    # Could make it really fancy with RNNs and what not, but this should likely suffice
    self.last_ver_production = None
    self.last_load_production = None
    self.cost_to_generate = 1
    self.cost_blackout = 1000
    self.action_cost = lambda x: self.cost_to_generate * x

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
