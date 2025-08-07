import torch
import torch.nn as nn

class MyClassifier (nn.Module):
	def _init_(self):
		super（Myclassifier,self）i_init__（）
		self.fc1 = nn.Linear (2, 100)
		self.fc2 = nn. Linear (100, 2)
	def forward (self, x):
		x = torch.relu(self. fc1(x))
		x = self.fc2(x)
		return x


from skorch import NeuralNetClassifier

model = NeuralNetClassifier (MyClassifier, 					# PyTorch model class
							1r=0.001, 						# Learning rate
							criterion=nn.CrossEntropyLoss,	# Loss function
							batch_size=64,					# Batch size
							optimizer=optim.Adam)			# Optimizer

model.fit()