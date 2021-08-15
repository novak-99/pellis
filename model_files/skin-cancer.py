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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #  12GB NVIDIA Tesla K80 GPU for decreased training times and parellelism. [https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiRq73l0a7yAhXVuZ4KHTnSBzYQFnoECA8QAw&url=https%3A%2F%2Fmedium.com%2F%40oribarel%2Fgetting-the-most-out-of-your-google-colab-2b0585f82403&usg=AOvVaw3BlldNaIvLIsquIc-LqMIn]

# Hyperparams
max_epoch = 10
lr = 0.01
decrement_rate = 10

class get_data(Dataset):
	def __init__(self, root_dir, csv_file, transforms):
		self.root_dir = root_dir
		self.csv_file = pd.read_csv(csv_file, header=None)
		self.n_class = len(pd.read_csv(csv_file).iloc[0]) - 1 # disregard img name
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

		label = np.empty(self.n_class)
		for i in range(self.n_class):
			label[i] = self.csv_file.iloc[index, i+1]
		label = label.astype(np.float32)

		if self.transforms:
			image = self.transforms(image.copy())

		return (image, torch.from_numpy(label))


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip(), 
	transforms.RandomVerticalFlip(), transforms.RandomCrop(224, padding_mode='reflect', padding=4)])

root_dir = "../skin-cancer-data"
train_val = get_data(root_dir, os.path.join(root_dir, "train.csv"), transform)
test_set = get_data(root_dir, os.path.join(root_dir, "test.csv"), transform)

print(train_val.__len__())
train_set, val_set = torch.utils.data.random_split(lengths=[2400, 815], dataset=train_val)



train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128)

model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
	param.requires_grad = False

model.fc = nn.Sequential(
	nn.Linear(model.fc.in_features, 3)
)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

def eval(model, loader):
  model.eval()
  n_c = 0
  n_s = 0
  for (images, labels) in loader:
    images, labels = images.to(device), labels.to(device)
    images = images.float()
    y_hat_pred = model(images)
    __, y_hat_label = torch.max(y_hat_pred, 1)
    n_c += torch.sum(y_hat_label == labels.argmax(-1)).item()
    n_s += labels.size()[0]

  return n_c/n_s * 100

current_acc = 0.0

for epoch in range(max_epoch):
  for (images, labels) in train_loader: # train loop
    model.train()
    images, labels = images.to(device), labels.to(device)
    images = images.float()
    y_hat = model(images)
    l = loss(y_hat, labels.argmax(-1))
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