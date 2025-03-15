import gym
import time
import argparse
import numpy as np

import torch

from libs import wrappers
from libs import dqn_model

import collections

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model file to load')
    parser.add_argument('-e', '--env', default=DEFAULT_ENV_NAME,
                        help='Enviroment name to use, default=' + DEFAULT_ENV_NAME)
    parser.add_argument('-r', '--record', help='Directory to store video recording')
    parser.add_argument('--no-visualize', default=True, action='store_false', dest='visualize',
                        help='Disable visualization of the game play')
    args = parser.parse_args()

    env = wrappers.make_env(args.env, True)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    counter = collections.Counter()

    while True:
        starts_ts = time.time()

        # GUI
        if args.visualize:
            env.render(mode='rgb_array')

        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        counter[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

        if args.visualize:
            delta = 1/FPS - (time.time() - starts_ts)
            if delta > 0:
                time.sleep(delta)

    print('Total reward: %.2f' % total_reward)
    print('Action counts:', counter)

    if args.record:
        env.env.close()