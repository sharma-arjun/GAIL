import argparse
import sys
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import scipy.optimize
from gym import wrappers

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Value, ActorCritic
from replay_memory import Memory
from running_state import ZFilter

# from utils import *

torch.set_default_tensor_type('torch.DoubleTensor')
PI = torch.DoubleTensor([3.1415926])

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('--num-episodes', type=int, default=500, metavar='N',
                    help='number of episodes (default: 500)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

if args.use_joint_pol_val:
    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=0.0003)
else:
    policy_net = Policy(num_inputs, num_actions)
    old_policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
    opt_value = optim.Adam(value_net.parameters(), lr=0.0003)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_actor_critic(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std, v = ac_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params_actor_critic(batch, i_episode):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    action_means, action_log_stds, action_stds, values = ac_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    opt_ac.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    clip_epsilon = args.clip_epsilon*max(1.0 - float(i_episode)/args.num_episodes, 0)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    action_var = Variable(actions)
    # compute probs from actions above
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old, values_old = ac_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    ac_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_ac.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()

    vf_loss = (values - targets).pow(2.).mean()
    total_loss = policy_surr + vf_loss
    total_loss.backward()
    opt_ac.step()


def update_params(batch, i_episode, optim_epochs, optim_batch_size):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)


    opt_value.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    opt_policy.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    clip_epsilon = args.clip_epsilon*max(1.0 - float(i_episode)/args.num_episodes, 0)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    advantages = (advantages - advantages.mean()) / advantages.std()

    # backup params after computing probs but before updating new params
    for old_policy_param, policy_param in zip(old_policy_net.parameters(), policy_net.parameters()):
        old_policy_param.data.copy_(policy_param.data)

    optim_iters = int(math.ceil(args.batch_size/optim_batch_size))

    for _ in range(optim_epochs):
        perm = np.arange(actions.size(0))
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm)
        states = states[perm]
        actions = actions[perm]
        values = values[perm]
        targets = targets[perm]
        advantages = advantages[perm]
        cur_id = 0
        for _ in range(optim_iters):
            cur_batch_size = min(optim_batch_size, actions.size(0) - cur_id)
            state_var = Variable(states[cur_id:cur_id+cur_batch_size])
            action_var = Variable(actions[cur_id:cur_id+cur_batch_size])
            advantages_var = Variable(advantages[cur_id:cur_id+cur_batch_size])

            action_means, action_log_stds, action_stds = policy_net(state_var)
            log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

            action_means_old, action_log_stds_old, action_stds_old = old_policy_net(state_var)
            log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

            opt_value.zero_grad()
            value_var = value_net(state_var)
            value_loss = (value_var - targets[cur_id:cur_id+cur_batch_size]).pow(2.).mean()
            value_loss.backward()
            opt_value.step()

            opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:,0]
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            opt_policy.step()

            cur_id += cur_batch_size

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []
optim_epochs = 5
optim_batch_size = 64

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            if args.use_joint_pol_val:
                action = select_action_actor_critic(state)
            else:
                action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()
    if args.use_joint_pol_val:
        for _ in range(10):
            update_params_actor_critic(batch, i_episode)
    else:
        update_params(batch, i_episode, optim_epochs, optim_batch_size)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))

    if i_episode == args.num_episodes:
        break
