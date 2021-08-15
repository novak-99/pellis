import cv2 
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import os
import csv

model = torch.load("../../../torch_models/brain-cancer.pt", map_location=torch.device('cpu'))
model.eval()

class load_image(Dataset):
	def __init__(self, img, transforms):
		self.img = img
		self.transforms = transforms
	def __len__(self):
		return 1
	def __getitem__(self, index):
		img = self.img
		if self.transforms:
			img = self.transforms(img.copy())
		return (img, [[]]) # emtpy label, not needed.

def pred(y_hat):
	if torch.round(y_hat).item() == 0:
		return "Benign"
	else:
		return "Malignant"

def is_ds(dir):
	ds = ".DS_Store"
	if ds in dir:
		return True
	else:
		return False


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip(), 
	transforms.RandomVerticalFlip(), transforms.RandomCrop(224, padding_mode='reflect', padding=4)])

for folder in os.listdir("."):
	if folder == "data": 
		f = open("results.txt", "w")
		for file in os.listdir(folder):
			if not is_ds(file):
				img_dir = os.path.join(folder, file)
				frame = cv2.imread(img_dir)
				ds = load_image(frame, transform)
				dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1)
				y_hat = [[]]
				for (img, __) in dl:
					y_hat = model(img)
				f.write(f"Prediction for image {img_dir}: {pred(y_hat)}\n")

		break
