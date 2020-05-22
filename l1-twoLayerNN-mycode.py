import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np 
import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10 
# 64 num of input, define the dimentsion of each layer

# we first use numpy to achieve the learing process.
# training data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
W1 = np.random.randn(D_in, H) # the matrix W1
W2 = np.random.randn(H, D_out) # the matrix W2

learning_rate = 1e-6
for it in range(500):
	# forward pass
	h = x.dot(W1) # N*H
	h_relu = np.maximum(h,0) # N*H
	y_pred = h_relu.dot(W2) # N*D_out

	# compute loss
	loss = np.square(y_pred-y).sum()
	print(it, loss)

	# backward pass
	# compute the gradient( loss对参数求导，d loss/d W1,链式求导)
	grad_y_pred = 2.0*(y_pred-y)
	grad_W2 = h_relu.T.dot(grad_y_pred)
	grad_h_relu = grad_y_pred.dot(W2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h<0] = 0
	grad_W1 = x.T.dot(grad_h)

	# update weights of W1 and W2, why not += ?
	W1 -= learning_rate * grad_W1
	W2 -= learning_rate * grad_W2

# the model is : 
h = x.dot(W1) # N*H
h_relu = np.maximum(h,0) # N*H
y_pred = h_relu.dot(W2) # N*D_out	
# let's check the y_pred
y_pred-y # the value is very samll


# now let's use function in torch to achieve this learning process.

# training data
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()
W1 = torch.randn(D_in, H).cuda() # the matrix W1
W2 = torch.randn(H, D_out).cuda() # the matrix W2

learning_rate = 1e-6
for it in range(500):
	# forward pass
	h = x.mm(W1) # N*H
	h_relu = h.clamp(min=0) # N*H
	y_pred = h_relu.mm(W2) # N*D_out

	# compute loss
	loss = (y_pred-y).pow(2).sum().item()
	print(it, loss)

	# backward pass
	# compute the gradient( loss对参数求导，d loss/d W1,链式求导)
	grad_y_pred = 2.0*(y_pred-y)
	grad_W2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(W2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h<0] = 0
	grad_W1 = x.t().mm(grad_h)

	# update weights of W1 and W2, why not += ?
	W1 -= learning_rate * grad_W1
	W2 -= learning_rate * grad_W2

# now let's use torch to achieve this learning process.

# training data
# why I got errors when I set these tensors to cuda?
# the W1.grad = None, after loss.backward()
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()
W1 = torch.randn(D_in, H, requires_grad = True).cuda() # the matrix W1,分配内存给梯度，需要指定
W2 = torch.randn(H, D_out, requires_grad = True).cuda() # the matrix W2

learning_rate = 1e-6
for it in range(500):
	# forward pass
	y_pred = x.mm(W1).clamp(min=0).mm(W2)

	# compute loss
	loss = (y_pred-y).pow(2).sum() # computation graph
	print(it, loss.item())

	# backward pass
	# compute the gradient
	loss.backward() # I guess this step need to be dealed with cuda to solve the bug

	# update weights of W1 and W2, why not += ?
	with torch.no_grad(): 
	# don't let computation graph to occupy memory, forget W1,W2's gradient.
		W1 -= learning_rate * W1.grad
		W2 -= learning_rate * W2.grad
		W1.grad.zero_()
		W2.grad.zero_()

# let's use nn intesd of self-define
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10 
# training data
# why I got errors when I set these tensors to cuda?
# the W1.grad = None, after loss.backward()
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H, bias=False),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out, bias=False),
	)
# torch.nn.init.normal_(model[0].weight) # set the parameter to normal distribution to improve acc
# torch.nn.init.normal_(model[2].weight) # however, sometimes this make things worse!

model = model.cuda()

learning_rate = 1e-4
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # Adan doesn't need normal initialization.
# this is tricky, too large-explosion, too small-slow convergence
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) , SGD need normal initialization.

for it in range(500):
	# forward pass
	y_pred = model.forward(x) # model.forward()

	# compute loss
	loss = loss_fn(y_pred, y) # computation graph
	print(it, loss.item())

	# backward pass
	# compute the gradient
	loss.backward() 

	# update model parameters
	optimizer.step()

	model.zero_grad()
	optimizer.zero_grad()

# let's use nn.Module that more complex than sequential module
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(TwoLayerNet, self).__init__()
		# define the model architecture
		self.linear1 = torch.nn.Linear(D_in, H, bias = False)
		self.linear2 = torch.nn.Linear(H, D_out, bias = False)

	def forward(self,x):
		y_pred = self.linear2(self.linear1(x).clamp(min=0))
		return y_pred

model = TwoLayerNet(D_in, H, D_out).cuda()
loss_fn = nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for it in range(500):
	# Forward pass
	y_pred = model(x) # model.forward
	# compute loss
	loss = loss_fn(y_pred, y)
	print(it, loss.item())

	optimizer.zero_grad()
	# Backward pass
	loss.backward()
	# update model parameters
	optimizer.step()
