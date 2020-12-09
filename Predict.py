
import io
import json

# import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
# from torchvision import datasets, models

# importing the libraries
#test

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms , models
from torch.autograd import Variable
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

normalize = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
                                
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('aerialmodel.pth')
model.eval()
model

def predict_image(image_path, model):
    print("Predict")
    model.eval()
    print("inside predict_image")
    image = Image.open(image_path)
    image_tensor = normalize(image).float()
    width, height = image.size
    image = image.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    image = np.array(image)
    image = image.transpose((2, 0, 1))
    image = image/255
    image[0] = (image[0] - 0.485)/0.229
    image[1] = (image[1] - 0.456)/0.224
    image[2] = (image[2] - 0.406)/0.225
    image = image[np.newaxis,:]
    image = torch.from_numpy(image)
    image = image.float()
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    print(classes.item())
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    output_c = output.cpu()
    index = output_c.data.numpy().argmax()   
    print(index)
    return str(index)
    



