import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

max_epoch = 200 # Anything more than 200 epochs practically overfits due to the ANN's precise approximation ability
lr = 0.01
n_h = 3

data = load_breast_cancer()
X, y = data.data, data.target

n_s, n_f = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()


class LogReg(nn.Module):
	def __init__(self, n_f, n_h):
		super(LogReg, self).__init__()
		self.l1 = nn.Linear(n_f, n_h)
		self.l2 = nn.Linear(n_h, 1)
	def forward(self, X):
		return torch.sigmoid(self.l2(torch.relu(self.l1(X))))

model = LogReg(n_f, n_h)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def eval(model, data_set, test_set):
	y_hat = model(data_set)
	n_c = torch.sum(torch.round(y_hat)==test_set.view(-1, 1)).item()
	n_s = test_set.shape[0] 
	return n_c/n_s * 100

for epoch in range(max_epoch):
	y_hat = model(X_train)
	l = loss(y_hat, y_train.view(-1, 1))
	l.backward()
	optimizer.step()
	optimizer.zero_grad()
	print(f"EPOCH: {epoch+1}, TRAIN ACC: {eval(model, X_train, y_train)}, TEST ACC: {eval(model, X_test, y_test)}")


FILE = "../torch_models/wisc-breast-cancer.pt"
torch.save(model, FILE)