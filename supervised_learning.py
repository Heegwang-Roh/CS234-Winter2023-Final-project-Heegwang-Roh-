import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Softmax(nn.Module):
	
	def __init__(self, input_dim, output_dim):
		super(Softmax, self).__init__()
		self.linear1 = nn.Linear(input_dim, output_dim)
	
	def forward(self, x):
		x = x.to(torch.float32)
		x = F.softmax(self.linear1(x))
		return x

class NeuralNet(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(NeuralNet, self).__init__()
		self.linear1 = nn.Linear(input_dim, 500)
		self.linear2 = nn.Linear(500, output_dim)
	
	def forward(self, x):
		x = x.to(torch.float32)
		x = F.relu(self.linear1(x))
		x = F.softmax(self.linear2(x))
		return x
		
def linear_regression(X, y):
	dim = X.shape[0]
	X = np.append(X, np.ones((dim, 1)), axis=1)
	y = y.reshape(dim, 1)
	
	theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

	return theta
	
def run_linear_regression(features, rewards, doses):
	pred_arms = []
	dim = features.shape[0]
	X = np.append(features, np.ones((dim, 1)), axis=1)
	theta = linear_regression(features, doses)
	pred_doses = np.dot(X, theta)
	for dose in pred_doses:
		if dose < 21: pred_arms.append(0)
		elif dose > 49: pred_arms.append(2)
		else: pred_arms.append(1)
	pred_arms = np.array(pred_arms)
	
	# calculate number of each arm chosen
	unique, counts = np.unique(pred_arms, return_counts = True)
		
	# check model accuracy
	correct = (torch.from_numpy(rewards) == torch.from_numpy(pred_arms)).sum().item()
	acc = correct / len(pred_arms)
	
	return dict(zip(unique, counts)), acc
	
	
def run_softmax(features, rewards):

	input_dim = features.shape[1]
	output_dim = 3	
	
	softmax = Softmax(input_dim, output_dim)
	optimizer = torch.optim.Adam(softmax.parameters(), lr = 0.05)
	criterion = nn.CrossEntropyLoss()
	data = TensorDataset(torch.from_numpy(features), torch.from_numpy(rewards))
	train_loader = DataLoader(dataset = data, batch_size = 20)
	
	epochs = 20
	print("Running Softmax classifier......")
	for epoch in range(epochs):
		for x, y in train_loader:
			optimizer.zero_grad()
			y_pred = softmax(x)
			loss = criterion(y_pred, y)
			loss.backward()
			optimizer.step()
	print("Done!")
	
	pred_model = softmax(torch.from_numpy(features))
	_, y_pred = pred_model.max(1)
	
	# calculate number of each arm chosen
	unique, counts = np.unique(y_pred.numpy(), return_counts = True)
	
	# check model accuracy
	correct = (torch.from_numpy(rewards) == y_pred).sum().item()
	acc = correct / len(data)
	
	return dict(zip(unique, counts)), acc
	
	
def run_neuralnet(features, rewards):

	input_dim = features.shape[1]
	output_dim = 3	
	
	neuralnet = NeuralNet(input_dim, output_dim)
	optimizer = torch.optim.Adam(neuralnet.parameters(), lr = 0.05)
	criterion = nn.CrossEntropyLoss()
	data = TensorDataset(torch.from_numpy(features), torch.from_numpy(rewards))
	train_loader = DataLoader(dataset = data, batch_size = 10)
	
	Loss = []
	epochs = 20
	print("Running NN classifier......")
	for epoch in range(epochs):
		for x, y in train_loader:
			optimizer.zero_grad()
			y_pred = neuralnet(x)
			loss = criterion(y_pred, y)
			Loss.append(loss)
			loss.backward()
			optimizer.step()
	print("Done!")
	
	pred_model = neuralnet(torch.from_numpy(features))
	_, y_pred = pred_model.max(1)
	
	# calculate number of each arm chosen
	unique, counts = np.unique(y_pred.numpy(), return_counts = True)
	
	# check model accuracy
	correct = (torch.from_numpy(rewards) == y_pred).sum().item()
	acc = correct / len(data)
	
	return dict(zip(unique, counts)), acc