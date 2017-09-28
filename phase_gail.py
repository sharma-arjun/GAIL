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

#from models import Policy, Value, Reward, ActorCritic
from phase_mlp import PMLP
from replay_memory import Memory
from load_expert_traj import Expert
from running_state import ZFilter

# from utils import *

torch.set_default_tensor_type('torch.DoubleTensor')
PI = torch.DoubleTensor([3.1415926])

class Phase():
    def __init__(self):
        #self.phase_list = [0, math.pi/2, math.pi, 3*math.pi/2]
        self.n = 32
        #self.l = np.linspace(1,1.5,self.n/2) #hopper
	self.l = np.linspace(0.8,2.0,(self.n+2)/2) #walker
        #self.timer = 500
	self.timer = 0

    #def comp_phase(self):

    #    if self.timer == 0:
    #        self.phase = random.choice(self.phase_list)
    #        self.timer += 1
    #    elif self.timer == 2:
    #        self.phase = random.choice(self.phase_list)
    #        self.timer = 1
    #    else:
    #        self.timer +=1

    #    return self.phase

    #def comp_phase(self):
    #    self.phase = (self.timer % 96)*math.pi/48
    #    self.timer += 1
    #    return self.phase

    # hopper
    #def comp_phase(self, height, vel):
    #    if height <= 1.0:
    #        phase = 0
    #    elif height > 1.5:
    #        phase = math.pi
    #    else:
    #        for i in range(self.n/2-1):
    #            if height > self.l[i] and height <= self.l[i+1]:
    #                phase = (2*math.pi/self.n)*(i+1)

    #    if vel < 0:
    #        phase = 2*math.pi - phase

    #    return phase

    # walker
    def comp_phase(self, height, vel):
        if height <= 0.8:
            phase = 0
        elif height > 2.0:
            phase = math.pi
        else:
            for i in range(self.n/2):
                if height > self.l[i] and height <= self.l[i+1]:
                    phase = (2*math.pi/self.n)*(i)

        if vel < 0:
            phase = 2*math.pi - phase

        return phase

    #def comp_phase(self):
    #    phase = (2*self.timer*math.pi)/1000
    #    self.timer = (self.timer + 1) % 1000

    #    return phase

    #def comp_phase(self, height, vel):
    #    if height <= 0.8:
    #        phase = 0

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-path', default="hopper_expert_trajectories/", metavar='G',
                    help='path to the expert trajectory files')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
# parser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
#                     help='l2 regularization regression (default: 1e-3)')
# parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
#                     help='max kl value (default: 1e-2)')
# parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
#                     help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('--num-episodes', type=int, default=500, metavar='N',
                    help='number of episodes (default: 500)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of epochs over a batch (default: 5)')
parser.add_argument('--optim-batch-size', type=int, default=64, metavar='N',
                    help='batch size for epochs (default: 64)')
parser.add_argument('--num-expert-trajs', type=int, default=5, metavar='N',
                    help='number of expert trajectories in a batch (default: 5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='interval between saving policy weights (default: 100)')
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
    policy_net = PMLP(input_size=num_inputs, output_size=num_actions, hidden_size=100, dtype=torch.DoubleTensor, n_layers=2, final_layer_flag=0, policy_flag=1)
    old_policy_net = PMLP(input_size=num_inputs, output_size=num_actions, hidden_size=100, dtype=torch.DoubleTensor, n_layers=2, final_layer_flag=0, policy_flag=1)
    value_net = PMLP(input_size=num_inputs, output_size=1, hidden_size=100, dtype=torch.DoubleTensor, n_layers=2, final_layer_flag=0, policy_flag=0)
    reward_net = PMLP(input_size=num_inputs+num_actions, output_size=1, hidden_size=100, dtype=torch.DoubleTensor, n_layers=2, final_layer_flag=2, policy_flag=0)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
    opt_value = optim.Adam(value_net.parameters(), lr=0.0003)
    opt_reward = optim.Adam(reward_net.parameters(), lr=0.0003)

def select_action(state, phase):
    state = torch.from_numpy(state).unsqueeze(0)
    phase = torch.from_numpy(np.array([phase])).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state), Variable(phase))
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

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

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
    #vpredclipped = values_old + torch.clamp(values - values_old, -args.clip_epsilon, args.clip_epsilon)
    #vf_loss2 = (vpredclipped - targets).pow(2.)
    #vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    total_loss = policy_surr + vf_loss
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm(ac_net.parameters(), 40)
    opt_ac.step()


def update_params(gen_batch, expert_batch, i_episode, optim_epochs, optim_batch_size):
    criterion = nn.BCELoss()

    # generated trajectories
    rewards = torch.Tensor(gen_batch.reward)
    masks = torch.Tensor(gen_batch.mask)
    actions = torch.Tensor(np.concatenate(gen_batch.action, 0))
    states = torch.Tensor(gen_batch.state)
    phases = torch.Tensor(gen_batch.phase).unsqueeze(1)
    #next_phases = torch.Tensor(gen_batch.next_phase)
    #print next_phases.shape
    values = value_net(Variable(states), Variable(phases))

    # expert trajectories
    list_of_expert_states = []
    for i in range(len(expert_batch.state)):
        list_of_expert_states.append(torch.Tensor(expert_batch.state[i]))
    expert_states = torch.cat(list_of_expert_states,0)

    list_of_expert_actions = []
    for i in range(len(expert_batch.action)):
        list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
    expert_actions = torch.cat(list_of_expert_actions, 0)

    list_of_masks = []
    for i in range(len(expert_batch.mask)):
        list_of_masks.append(torch.Tensor(expert_batch.mask[i]))
    expert_masks = torch.cat(list_of_masks, 0)

    list_of_phases = []
    for i in range(len(expert_batch.phase)):
        list_of_phases.append(torch.Tensor(expert_batch.phase[i]))
    expert_phases = torch.cat(list_of_phases, 0).unsqueeze(1)

    #list_of_next_phases = []
    #for i in range(len(expert_batch.next_phase)):
    #    list_of_next_phases.append(torch.Tensor(expert_batch.next_phase[i]))
    #expert_next_phases = torch.cat(list_of_next_phases, 0)
    #print expert_next_phases.size()

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    opt_value.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    opt_policy.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    opt_reward.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    clip_epsilon = args.clip_epsilon*max(1.0 - float(i_episode)/args.num_episodes, 0)

    # compute advantages
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
    #policy_net.backup()
    for old_policy_param, policy_param in zip(old_policy_net.parameters(), policy_net.parameters()):
        old_policy_param.data.copy_(policy_param.data)

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

    # update value, reward and policy networks
    optim_iters = int(math.ceil(args.batch_size/optim_batch_size))
    optim_batch_size_exp = int(math.ceil(expert_actions.size(0)/(optim_iters)))

    for _ in range(optim_epochs):
        perm = np.arange(actions.size(0))
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm)
        states = states[perm]
        actions = actions[perm]
        phases = phases[perm]
        values = values[perm]
        targets = targets[perm]
        advantages = advantages[perm]
        perm_exp = np.arange(expert_actions.size(0))
        np.random.shuffle(perm_exp)
        perm_exp = torch.LongTensor(perm_exp)
        expert_states = expert_states[perm_exp]
        expert_actions = expert_actions[perm_exp]
        expert_phases = expert_phases[perm_exp]
        cur_id = 0
        cur_id_exp = 0
        for _ in range(optim_iters):
            cur_batch_size = min(optim_batch_size, actions.size(0) - cur_id)
            cur_batch_size_exp = min(optim_batch_size_exp, expert_actions.size(0) - cur_id_exp)
            state_var = Variable(states[cur_id:cur_id+cur_batch_size])
            action_var = Variable(actions[cur_id:cur_id+cur_batch_size])
            phase_var = Variable(phases[cur_id:cur_id+cur_batch_size])
            advantages_var = Variable(advantages[cur_id:cur_id+cur_batch_size])
            expert_state_var = Variable(expert_states[cur_id_exp:cur_id_exp+cur_batch_size_exp])
            expert_action_var = Variable(expert_actions[cur_id_exp:cur_id_exp+cur_batch_size_exp])
            expert_phase_var = Variable(expert_phases[cur_id_exp:cur_id_exp+cur_batch_size_exp])

            # update reward net
            opt_reward.zero_grad()

            # backprop with expert demonstrations
            o = reward_net(torch.cat((expert_state_var, expert_action_var),1), expert_phase_var)
            loss = criterion(o, Variable(torch.zeros(expert_action_var.size(0),1)))
            loss.backward()

            # backprop with generated demonstrations
            o = reward_net(torch.cat((state_var, action_var),1), phase_var)
            loss = criterion(o, Variable(torch.ones(action_var.size(0),1)))
            loss.backward()
    
            opt_reward.step()

            # compute old and new action probabilities
            action_means, action_log_stds, action_stds = policy_net(state_var, phase_var)
            log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

            action_means_old, action_log_stds_old, action_stds_old = old_policy_net(state_var, phase_var)
            log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

            # update value net
            opt_value.zero_grad()
            value_var = value_net(state_var, phase_var)
            value_loss = (value_var - targets[cur_id:cur_id+cur_batch_size]).pow(2.).mean()
            value_loss.backward()
            opt_value.step()

            # update policy net
            opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:,0]
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            opt_policy.step()

            # set new starting point for batch
            cur_id += cur_batch_size
            cur_id_exp += cur_batch_size_exp

running_state = ZFilter((num_inputs,), clip=5)
#running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []
optim_epochs = args.optim_epochs
optim_batch_size = args.optim_batch_size

expert = Expert(args.expert_path, num_inputs)
print 'Loading expert trajectories ...'
expert.push()
print 'Expert trajectories loaded.'
phase_obj = Phase()

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    true_reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        #state = running_state(state)
        phase = phase_obj.comp_phase(env.env.model.data.qpos[1,0], env.env.model.data.qvel[1,0])

        reward_sum = 0
        true_reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            if args.use_joint_pol_val:
                action = select_action_actor_critic(state)
            else:
                action = select_action(state, phase)
            reward = -math.log(reward_net(torch.cat((Variable(torch.from_numpy(state).unsqueeze(0)), action), 1), Variable(torch.from_numpy(np.array([phase])).unsqueeze(0))).data.numpy()[0,0])
            action = action.data[0].numpy()
            next_state, true_reward, done, _ = env.step(action)
            next_phase = phase_obj.comp_phase(env.env.model.data.qpos[1,0], env.env.model.data.qvel[1,0])
            reward_sum += reward
            true_reward_sum += true_reward

            #next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward, phase, next_phase)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
            phase = next_phase

        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum
        true_reward_batch += true_reward_sum

    reward_batch /= num_episodes
    true_reward_batch /= num_episodes
    gen_batch = memory.sample()
    expert_batch = expert.sample(size=args.num_expert_trajs)
    if args.use_joint_pol_val:
        for _ in range(10):
            update_params_actor_critic(gen_batch, expert_batch, i_episode)
    else:
        update_params(gen_batch, expert_batch, i_episode, optim_epochs, optim_batch_size)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward {}\tAverage reward {}\tLast true reward {}\tAverage true reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch, true_reward_sum, true_reward_batch))

    if i_episode % args.save_interval == 0:
        f_w = open('checkpoints/policy_' + str(args.env_name) + '_ep_' + str(i_episode) + '_batch_' + str(args.batch_size) + '_epochs_' + str(args.optim_epochs)  + '_exptraj_' + str(args.num_expert_trajs) + '_reward_' + str(true_reward_batch) + '.pth', 'wb')
        checkpoint = {'running_state':running_state}
        if args.use_joint_pol_val:
            checkpoint['policy'] = ac_net
        else:
            checkpoint['policy'] = policy_net
        torch.save(checkpoint, f_w)

    if i_episode == args.num_episodes:
        break
