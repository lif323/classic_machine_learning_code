import torch 
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

def plane_function(x, y):
	return x + 2 * y

# 线性可分数据集
def gen_data(num):
	# x [-100, 100]
	# y [-100, 100]
	# x + 2y + 4 = 0
	datas = list()
	labels = list()
	for _ in range(num):
		x = np.random.uniform(-100, 100)
		y = np.random.uniform(-100, 100)
		if plane_function(x, y) == 0:
			continue
		datas.append([x, y])
		labels.append(1 if plane_function(x, y) > 0 else -1)
	datas = np.array(datas)
	labels = np.array(labels)
	labels = labels.reshape([labels.shape[0], 1])
	labels = labels.astype(np.float32)
	return datas, labels

class Perceptron(nn.Module):
	def __init__(self):
		super(Perceptron, self).__init__()
		self.w = torch.tensor([[0,0]], dtype=torch.float32)
		self.b = torch.tensor(0, dtype=torch.float32)
		
	def forward(self, x):
		pred = torch.sum(self.w * x) + self.b
		return pred 

def opti(model, features, labels, learning_rate=1e-3):
	wrong_num = 0
	for num, (f, l) in enumerate(zip(features, labels)):
		f = torch.from_numpy(f)
		l = torch.from_numpy(l)
		pred = model(f)
		if pred * l <= 0:
			wrong_num += 1
			model.w = model.w + learning_rate * l * f	
			model.b = model.b + learning_rate * l	
	return wrong_num
	
if __name__ == "__main__":
	features, labels = gen_data(1000)
	model = Perceptron()
	epoch = 100
	for e in range(epoch):
		wrong_num = opti(model, features, labels)
		print("w: ", model.w.numpy(), "b: ", model.b.numpy(), "num: ", wrong_num)