import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset


import os 
import pandas as pd
import numpy as np
import skimage.io as io 
from skimage.color import rgba2rgb, gray2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #  12GB NVIDIA Tesla K80 GPU for decreased training times and parallelism. [https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiRq73l0a7yAhXVuZ4KHTnSBzYQFnoECA8QAw&url=https%3A%2F%2Fmedium.com%2F%40oribarel%2Fgetting-the-most-out-of-your-google-colab-2b0585f82403&usg=AOvVaw3BlldNaIvLIsquIc-LqMIn]

# Hyperparams
max_epoch = 10
lr = 0.01
decrement_rate = 10

class get_data(Dataset):
	def __init__(self, root_dir, csv_file, transforms):
		self.root_dir = root_dir
		self.csv_file = pd.read_csv(csv_file, header=None)
		self.transforms = transforms
	def __len__(self):
		return len(self.csv_file)
	def __getitem__(self, index):
		image_dir = os.path.join(self.root_dir, str(self.csv_file.iloc[index, 0])) # the 0th index indicates the file name
		image = io.imread(image_dir)
		if len(image.shape) == 2: 
			image = gray2rgb(image)
		else:
			if image.shape[2] == 2:
				image = gray2rgb(image)
			elif image.shape[2] == 4:
				image = rgba2rgb(image)

		label = self.csv_file.iloc[index, 1] # the 1st index indicates the label
		label = label.astype(np.float32)

		if self.transforms:
			image = self.transforms(image.copy())

		return (image, label)


root_dir = "../brain_cancer/Brain Tumor/Brain Tumor"
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip(), 
	transforms.RandomVerticalFlip(), transforms.RandomCrop(224, padding_mode='reflect', padding=4)])

train_test_val = get_data(root_dir, "../brain_cancer/labels.csv", transform)

train_val, test_set = torch.utils.data.random_split(lengths=[3200,562], dataset=train_test_val)

train_set, val_set = torch.utils.data.random_split(lengths=[2500, 700], dataset=train_val)


print(train_set.__len__())
print(val_set.__len__())
print(test_set.__len__())

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128)

# DONT FORGET TO DOWNLOAD RESNET 50 MODEL.
model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
	param.requires_grad = False

model.fc = nn.Sequential(
	nn.Linear(model.fc.in_features, 1), # Sigmoid indicates n_class of 1. 
	nn.Sigmoid()
)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.BCELoss()

def eval(model, loader):
	model.eval()
	n_c = 0
	n_s = 0
	with torch.no_grad():
		for (images, labels) in loader:
			images, labels = images.to(device), labels.to(device)
			y_hat_pred = model(images)
			n_c += torch.sum(torch.round(y_hat_pred) == labels.view(-1, 1)).item()
			n_s += labels.size()[0]

	return n_c/n_s * 100

current_acc = 0.0

for epoch in range(max_epoch):
	for (images, labels) in train_loader: # train loop
		model.train()
		images, labels = images.to(device), labels.to(device)
		y_hat = model(images)
		l = loss(y_hat, labels.view(-1, 1))
		l.backward()
		optimizer.step()
		optimizer.zero_grad()
		print(l)

	# Validation 
	acc = eval(model, val_loader) 
	if acc <= current_acc:
		for g in optimizer.param_groups:
			g['lr'] /= decrement_rate

	current_acc = acc
	print(f"EPOCH: {epoch+1}, TRAIN ACC: {eval(model, train_loader)}, VAL ACC: {acc}, TEST ACC: {eval(model, test_loader)}")


FILE = "../torch_models/brain-cancer.py"
torch.save(model, FILE)



