from libs import wrappers
from libs import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
# средняя награда для последних 100 эпизодов чтобы прекратить обучение
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
# роазмер обучающегот набора 
BATCH_SIZE = 32
# макс емкость буфера примеров
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
# как часто копир веса обуч модели в целевую
SYNC_TARGET_FRAMES = 1000
# кол-во переходов (состояние, действие, ...) для буфера примеров
REPLAY_START_SIZE = 10000

# скорость уменьш эпсилон
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
    )


# буфер примеров. 
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # ранд наблюдения из буфера примеров
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), np.array(next_states)
    

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        # эпс-жад метод для изучения среды
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        # кон эпизода
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward
    

def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device, dtype=torch.int64)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device, dtype=torch.bool)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # max возвр макс знач-я и инд-ы
    next_state_values = tgt_net(next_states_v).max(1)[0]
    # после кон эпизода нет состояния -> нет награды
    next_state_values[done_mask] = 0.0
    # detach не исп эти веса для вычисл градиентов
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable cuda')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME,
                        help='Name of the envitoment, default=' + DEFAULT_ENV_NAME)
    parser.add_argument('--reward', type=float, default=MEAN_REWARD_BOUND,
                        help='Mean reward boundary for stop of training, default=%.2f' % MEAN_REWARD_BOUND)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment='-' + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_reward = []
    best_mean_reward = None
    # счетчик кадров
    frame_idx = 0
    # отслеж скорость
    ts_frame = 0
    ts = time.time()

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_reward.append(reward)

            # кадр / сек
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            mean_reward = np.mean(total_reward[-100:])
            print('%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s' % (
                frame_idx, len(total_reward), mean_reward, epsilon, speed
            ))

            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_100', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)

            # сохр модель
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + '-best.dat')
                if best_mean_reward is not None:
                    print('Best mean reward updated %.3f -> %.3f, model saved' % (
                        best_mean_reward, mean_reward
                    ))
                best_mean_reward = mean_reward

            # задача решена
            if mean_reward > args.reward:
                print('Solved in %d frames!' % frame_idx)
                break

        # достаточно ли данных в буфере
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # копир Q в целевую Q
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()