import cv2 
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import os
import csv

model = torch.load("../../../torch_models/skin-cancer-5.pt", map_location=torch.device('cpu'))
typeModel = torch.load("../../../torch_models/ham10000.pt", map_location=torch.device('cpu'))
model.eval()
temp = cv2.imread("templates/ex-lesion.png", 0)

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

def invasivePred(y_hat):
	y_hat = torch.softmax(y_hat, dim=1)
	print(y_hat)
	if torch.all(torch.eq(torch.round(y_hat), torch.tensor([[1,0,0]]))):
		return "Benign"
	elif torch.all(torch.eq(torch.round(y_hat), torch.tensor([[0,1,0]]))):
		return "Malignant"
	else:
		return "Trivial/Misc Object"

def typePred(y_hat_type):
	y_hat_type = torch.softmax(y_hat_type, dim=1)
	print(y_hat_type)
	label_dict = {0:"Bowen's Disease", 1:'Basal Cell Carcinoma', 2:'Benign Leratosis-Like Lesion', 3:'Dermatofibroma', 4:'Melanoma', 5:'Melanocytic Nevi', 6:'Vascular Lesions'}
	return label_dict[y_hat_type.argmax(-1).item()] 

def is_ds(dir):
	ds = ".DS_Store"
	if ds in dir:
		return True
	else:
		return False

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomHorizontalFlip(), 
	transforms.RandomVerticalFlip(), transforms.RandomCrop(224, padding_mode='reflect', padding=4)])
cap = cv2.VideoCapture(0) # if you don't have a webcam, this is also fine. There will be an option to just put imgs in a dir

localize = False
while True:
	ret, frame = cap.read()
	w, h = int(cap.get(3)), int(cap.get(4))
	if localize:
		# Localization
		r_matrix = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), temp, cv2.TM_SQDIFF)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(r_matrix)
		bottom_right = (min_loc[0] + temp.shape[1], min_loc[1] + temp.shape[0])
		cv2.rectangle(frame, min_loc, bottom_right, 2)
	else:
		ds = load_image(frame, transform)
		dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1)
		y_hat = [[]]
		for (img, __) in dl:
			y_hat = model(img)

		prediction = invasivePred(y_hat)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, prediction, (w-1200, h-50), font, 2, 0, 4, cv2.LINE_AA) 

		if prediction != "Trivial/Misc Object":
			# Skin Cancer Type
			for (img, __) in dl:
				y_hat = model(img)
				y_hat_type = typeModel(img)
			typePrediction = typePred(y_hat_type)
			cv2.putText(frame, "Type: " + typePrediction, (w-1200, h-20), font, 1, 0, 2, cv2.LINE_AA) 


	cv2.imshow("Pellis - Skin Cancer Detection", frame)
	if cv2.waitKey(1) == ord('s'):
		localize = not localize
		print("Switching modes...")

	if cv2.waitKey(1) == ord('q'): # This is for small scale datasets only, so we can be somewhat inefficient
		print("Terminating...")
		cap.release()
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
						f.write(f"Prediction for image {img_dir}: {invasivePred(y_hat)}\n")

				break
		break

cv2.destroyAllWindows() 
