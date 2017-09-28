import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

def kn(p, n):
	return ((np.floor((4*p)/(2*math.pi)) + n - 1) % 4).astype(int)

def spline_w(p):
	return ((4*p)/(2*math.pi)) % 1

def compute_multipliers(w,p):
    list_of_w = [torch.zeros(p.shape), torch.zeros(p.shape), torch.zeros(p.shape), torch.zeros(p.shape)]
    w2 = torch.pow(w,2)
    w3 = torch.pow(w,3)

    for n in range(4):
        ind = kn(p,n)
        if n == 0:
            weight = w2 - 0.5*w - 0.5*w3 
        elif n == 1:
            weight = 1 - 2.5*w2 + 1.5*w3
        elif n == 2:
            weight = 0.5*w + 2*w2 - 1.5*w3
        elif n == 3:
            weight = 0.5*(w3 - w2)
        for j in range(len(ind)):
            list_of_w[ind[j,0]][j,0] = weight[j,0]

    return list_of_w[0], list_of_w[1], list_of_w[2], list_of_w[3]


def init_fanin(tensor):
	fanin = tensor.size(1)
	v = 1.0 / np.sqrt(fanin)
	init.uniform(tensor, -v, v)


class PMLP(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, dtype, n_layers=1, batch_size=1, scale=1.0, final_layer_flag=0, policy_flag=0):
		super(PMLP, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.scale = scale # scale output of actor from [-1,1] to range of action space [-scale,scale]. set to 1 for critic
		self.final_layer_flag = final_layer_flag # 1 for actor, 0 for critic (since critic range need not be restricted to [-1,1])
                self.dtype = dtype
                self.policy_flag = policy_flag
                if policy_flag:
                   self.action_log_std = nn.Parameter(torch.zeros(1, output_size))

		#self.control_gru_list = []
		self.control_hidden_list = []
		self.control_h2o_list = []

	
		#self.gru_00 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_00 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_0 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_10 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_10 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_1 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_20 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_20 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_2 = nn.Linear(self.hidden_size, self.output_size).type(dtype)
		#self.gru_30 = nn.GRUCell(self.input_size, self.hidden_size)
		self.l_30 = nn.Linear(self.input_size, self.hidden_size).type(dtype)
		self.h2o_3 = nn.Linear(self.hidden_size, self.output_size).type(dtype)


                init_fanin(self.l_00.weight)
                init_fanin(self.l_10.weight)
                init_fanin(self.l_20.weight)
                init_fanin(self.l_30.weight)

                init.uniform(self.h2o_0.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_0.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_1.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_1.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_2.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_2.bias,-3e-3, 3e-3)
                init.uniform(self.h2o_3.weight,-3e-3, 3e-3)
                init.uniform(self.h2o_3.bias,-3e-3, 3e-3)


		if n_layers == 2:
			
			#self.gru_01 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_11 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_21 = nn.GRUCell(self.hidden_size, self.hidden_size)
			#self.gru_31 = nn.GRUCell(self.hidden_size, self.hidden_size)
			self.l_01 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_11 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_21 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)
			self.l_31 = nn.Linear(self.hidden_size, self.hidden_size).type(dtype)

                        init_fanin(self.l_01.weight)
                        init_fanin(self.l_11.weight)
                        init_fanin(self.l_21.weight)
                        init_fanin(self.l_31.weight)
		
		self.control_hidden_list.append([self.l_00, self.l_10, self.l_20, self.l_30])
		if n_layers == 2:
			self.control_hidden_list.append([self.l_01, self.l_11, self.l_21, self.l_31])

		self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

		#self.alpha = []
		#for i in range(4):
		#	self.alpha.append(Alpha(n_layers))

		#self.init_controls(self.control_hidden_list, self.control_h2o_list, self.alpha)
		#self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)
		#if n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)

		self.hidden_list = []
		self.h2o_list = []
		self.phase_list = []

		# to initialize grad of control hidden and h2o ... I need to do this stupid thing ...
		dummy_x = Variable(torch.zeros(batch_size, input_size), requires_grad=False).type(dtype)
		dummy_y = Variable(torch.zeros(batch_size, output_size), requires_grad=False).type(dtype)
		dummy_criterion = nn.MSELoss()

		if n_layers == 1:
			for l, h2o in zip(self.control_hidden_list[0], self.control_h2o_list):
				dummy_h = F.relu(l(dummy_x))
				dummy_o = h2o(dummy_h)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		if n_layers == 2:
			for l0, l1, h2o in zip(self.control_hidden_list[0], self.control_hidden_list[1], self.control_h2o_list):
				dummy_h0 = F.relu(l0(dummy_x))
				dummy_h1 = l1(dummy_h0)
				dummy_o = h2o(dummy_h1)
				dummy_loss = dummy_criterion(dummy_o, dummy_y)
				dummy_loss.backward()

		# reset to zero after dummy pass
		#self.h_0 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)
		#if n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)

	def forward(self,x,phase):
		#w = self.weight_from_phase(phase, self.control_hidden_list, self.control_h2o_list)
		#hiddens = []
		#l0 = nn.Linear(self.input_size, self.hidden_size).type(self.dtype)
		#hiddens.append(l0)
		#h2o = nn.Linear(self.hidden_size, self.output_size).type(self.dtype)
		#if self.n_layers == 2:
		#	l1 = nn.Linear(self.hidden_size, self.hidden_size).type(self.dtype)
		#	hiddens.append(l1)

		#self.set_weight(w, hiddens, h2o)
		#self.hidden_list.append(hiddens)
		#self.h2o_list.append(h2o)
		#self.phase_list.append(phase)
                
                control_hidden_list = self.control_hidden_list
                control_h2o_list = self.control_h2o_list

                w = torch.from_numpy(spline_w(phase.data.numpy()))
                w0,w1,w2,w3 = compute_multipliers(w,phase.data.numpy())
                w0_h = Variable(w0.repeat(1,self.hidden_size).type(self.dtype))
                w1_h = Variable(w1.repeat(1,self.hidden_size).type(self.dtype))
                w2_h = Variable(w2.repeat(1,self.hidden_size).type(self.dtype))
                w3_h = Variable(w3.repeat(1,self.hidden_size).type(self.dtype))

                w0_o = Variable(w0.repeat(1,self.output_size).type(self.dtype))
                w1_o = Variable(w1.repeat(1,self.output_size).type(self.dtype))
                w2_o = Variable(w2.repeat(1,self.output_size).type(self.dtype))
                w3_o = Variable(w3.repeat(1,self.output_size).type(self.dtype))
                
	        h_0 = F.relu(w0_h*control_hidden_list[0][0](x) + w1_h*control_hidden_list[0][1](x) + w2_h*control_hidden_list[0][2](x) + w3_h*control_hidden_list[0][3](x))
                if self.n_layers == 2:
        	        h_1 = F.relu(w0_h*control_hidden_list[1][0](h_0) + w1_h*control_hidden_list[1][1](h_0) + w2_h*control_hidden_list[1][2](h_0) + w3_h*control_hidden_list[1][3](h_0))
                        o = w0_o*control_h2o_list[0](h_1) + w1_o*control_h2o_list[1](h_1) + w2_o*control_h2o_list[2](h_1) + w3_o*control_h2o_list[3](h_1)
                        if self.final_layer_flag == 1:
                            o = F.tanh(o)
                        elif self.final_layer_flag == 2:
                            o = F.sigmoid(o)

                else:
                        o = w0_o*control_h2o_list[0](h_0) + w1_o*control_h2o_list[1](h_0) + w2_o*control_h2o_list[2](h_0) + w3_o*control_h2o_list[3](h_0)
                        if self.final_layer_flag == 1:
                            o = F.tanh(o)
                        elif self.final_layer_flag == 2:
                            o = F.sigmoid(o)

		#h_0 = F.relu(l0(x))
		#if self.n_layers == 2:
		#	h_1 = F.relu(l1(h_0))
		#	if self.tanh_flag:
		#		o = F.tanh(h2o(h_1))
		#	else:
		#		o = h2o(h_1)
		#else:
		#	if self.tanh_flag:
		#		o = F.tanh(h2o(h_0))
		#	else:
		#		o = h2o(h_0)

                if self.policy_flag:
                    action_log_std = self.action_log_std.expand_as(o)
                    action_std = torch.exp(action_log_std)

                    return self.scale*o, action_log_std, action_std

                else:
             	    return self.scale*o

	def reset(self):
		#self.h_0 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		#if self.n_layers == 2:
		#	self.h_1 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		
		self.hidden_list = []
		self.h2o_list = []
		self.phase_list = []

		#self.init_controls(self.control_hidden_list, self.control_h2o_list, self.alpha)


	def weight_from_phase(self, phase, control_hidden_list, control_h2o_list):
		weight = {}
		w = spline_w(phase)
                for n in range(len(control_hidden_list)):
        		for key in control_hidden_list[0][0]._parameters.keys():
	        		weight[key + '_' + str(n)] = control_hidden_list[n][kn(phase, 1)]._parameters[key].data + w*0.5*(control_hidden_list[n][kn(phase, 2)]._parameters[key].data - control_hidden_list[n][kn(phase, 0)]._parameters[key].data) + w*w*(control_hidden_list[n][kn(phase, 0)]._parameters[key].data - 2.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data + 2*control_hidden_list[n][kn(phase, 2)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data - 1.5*control_hidden_list[n][kn(phase, 2)]._parameters[key].data + 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 0)]._parameters[key].data)


                for key in control_h2o_list[0]._parameters.keys():
        		weight[key] = control_h2o_list[kn(phase, 1)]._parameters[key].data + w*0.5*(control_h2o_list[kn(phase, 2)]._parameters[key].data - control_h2o_list[kn(phase, 0)]._parameters[key].data) + w*w*(control_h2o_list[kn(phase, 0)]._parameters[key].data - 2.5*control_h2o_list[kn(phase, 1)]._parameters[key].data + 2*control_h2o_list[kn(phase, 2)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_h2o_list[kn(phase, 1)]._parameters[key].data - 1.5*control_h2o_list[kn(phase, 2)]._parameters[key].data + 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 0)]._parameters[key].data)

		return weight


	def set_weight(self, w, hiddens, h2o):
		count = 0
		for l in hiddens:
			l._parameters['weight'].data = w['weight_' + str(count)]
			l._parameters['bias'].data = w['bias_' + str(count)]
			count += 1

		h2o._parameters['weight'].data = w['weight']
		h2o._parameters['bias'].data = w['bias']

	#def init_controls(self, list_of_hidden, list_of_h2o, alpha):
	#	for i in range(len(alpha)):
	#		for j in range(len(list_of_hidden)):
	#			l = list_of_hidden[j][i]
	#			alpha[i]._parameters['weight_' + str(j)] = l._parameters['weight'].data.clone()
	#			alpha[i]._parameters['bias_' + str(j)] = l._parameters['bias'].data.clone()

	#			#initialize alpha grads as zero here using shape ...
	#			alpha[i]._grad['weight_' + str(j)] = torch.zeros(l._parameters['weight'].data.size()).type(self.dtype)
	#			alpha[i]._grad['bias_' + str(j)] = torch.zeros(l._parameters['bias'].data.size()).type(self.dtype)

	#		h2o = list_of_h2o[i]
	#		alpha[i]._parameters['weight'] = h2o._parameters['weight'].data.clone()
	#		alpha[i]._parameters['bias'] = h2o._parameters['bias'].data.clone()
	#		alpha[i]._grad['weight'] = torch.zeros(h2o._parameters['weight'].data.size()).type(self.dtype)
	#		alpha[i]._grad['bias'] = torch.zeros(h2o._parameters['bias'].data.size()).type(self.dtype)


	def update_control_gradients(self):
		for hiddens, phase in zip(self.hidden_list, self.phase_list):
			w = spline_w(phase)
			count = 0
			for l in hiddens:
				for key in l._parameters.keys():
					self.control_hidden_list[count][kn(phase,0)]._parameters[key].grad.data += l._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
					self.control_hidden_list[count][kn(phase,1)]._parameters[key].grad.data += l._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
					self.control_hidden_list[count][kn(phase,2)]._parameters[key].grad.data += l._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
					self.control_hidden_list[count][kn(phase,3)]._parameters[key].grad.data += l._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)
				count += 1

		for h2o, phase in zip(self.h2o_list, self.phase_list):
			w = spline_w(phase)
			for key in h2o._parameters.keys():
				self.control_h2o_list[kn(phase,0)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
				self.control_h2o_list[kn(phase,1)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
				self.control_h2o_list[kn(phase,2)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
				self.control_h2o_list[kn(phase,3)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)


		#for i in range(len(self.control_hidden_list)):
		#	for alpha, l in zip(self.alpha, self.control_hidden_list[i]):
		#		for key in l._parameters.keys():
		#			l._parameters[key].grad.data += alpha._grad[key + '_' + str(i)]

		#for alpha, h2o in zip(self.alpha, self.control_h2o_list):
		#	for key in h2o._parameters.keys():
		#		h2o._parameters[key].grad.data += alpha._grad[key]
