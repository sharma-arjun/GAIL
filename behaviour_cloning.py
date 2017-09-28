import math
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models import Policy
import torch.optim as optim


def save_checkpoint(state, filename):
	torch.save(state, filename)

save_file_name = 'mlp_'

f = h5py.File('../ego_trial/state_data.h5','r')

policy = Policy(57,45)

criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=0.0001)

inputs = np.zeros((370*14,57))
targets = np.zeros((370*14,45))
count = 0

for j in range(15):
	if j == 5:
		continue
	for t in range(370):
		inputs[count,:] = np.asarray(f[str(j)][str(t)])[0:57]
		#inputs[count,57] = inputs[t,57] % (2*math.pi)
		targets[count,:] = np.asarray(f[str(j)][str(t+1)])[0:45] - inputs[count,0:45]
		#targets[count,45] = np.asarray(f[str(j)][str(t+1)])[57] % (2*math.pi)
		count += 1

# shuffle inputs and targets
ind = np.arange(370*14)
np.random.shuffle(ind)
inputs = inputs[ind]
targets = targets[ind]

batch_size = 32

for k in range(1000):#epochs
	print 'Epoch:', k
	print '*****'
	epoch_avg_loss = 0
	batch_pointer = 0

	j = 0
	while batch_pointer < inputs.shape[0]:
		epoch_batch_size = min(batch_size, inputs.shape[0]- batch_pointer)

		x = Variable(torch.from_numpy(inputs[batch_pointer:batch_pointer+epoch_batch_size, :]).float(), requires_grad = False)
		y = Variable(torch.from_numpy(targets[batch_pointer:batch_pointer+epoch_batch_size, :]).float(), requires_grad= False)
		optimizer.zero_grad()
		outputs,_,_ = policy.forward(x)
		
		loss = criterion(outputs, y)
		epoch_avg_loss += loss.data[0]
		print 'Epoch {} Batch {} Loss {}'.format(k, j, loss.data[0])

		loss.backward(retain_variables=False)
		optimizer.step()

		j += 1
		batch_pointer += epoch_batch_size

	epoch_avg_loss /= j
	print 'Average epoch loss: ', epoch_avg_loss
	print '\n'

filename = 'checkpoints/' + save_file_name + 'epoch_' + str(k+1) + '_loss_' + str(epoch_avg_loss) + '_checkpoint.pth'
save_checkpoint({'epoch':k+1, 'state_dict':policy.state_dict(), 'optimizer':optimizer.state_dict()}, filename)
