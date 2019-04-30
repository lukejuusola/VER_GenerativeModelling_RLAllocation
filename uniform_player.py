import gym
import gym_ver
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    env = gym.make('ver-v0')
    state = env.reset()
    n_loaders = env.load.n
    total_cost = 0
    turns = 10000

    # to graph
    costs = []
    nrgs = []
    cum_avg_cost = []
    for i in range(1, turns):
        cur_nrg = state[0]
        to_gen = n_loaders - cur_nrg
        state, reward, done, info = env.step(to_gen)
        total_cost -= reward
        nrgs.append(cur_nrg)
        cum_avg_cost.append(total_cost / i)
        costs.append(-reward)
        if done:
            break


    # Graph
    print('Graphing')
    xs = np.arange(1, turns)
    #exp_cost_per_turn = n_loaders / 2
    cum_avg_cost = np.array(cum_avg_cost)
    #cum_avg_cost -= exp_cost_per_turn
    plt.plot(xs, cum_avg_cost)
    #plt.plot(xs, nrgs, color = 'red')
    plt.plot(xs, np.zeros(xs.shape[0]), color = 'orange')
    plt.show()
    #plt.hist(costs, bins = 100)
    #plt.show()
