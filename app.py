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
import Model
import Predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('aerialmodel.pth')
model.eval()
model

app = Flask(__name__)

@app.route('/predict', methods=['POST']) #this is how Rest API is built
def predict():
    if request.method == 'POST':
        file = request.files['file']
        imagefile = request.files.get('file', '')
        imagefile.save('ImageTest/test_image.jpg')
        image_path = 'ImageTest/test_image.jpg'
        model.eval()
        class_index = Predict.predict_image(image_path, model)
        Class_Name = Model.trainloader.dataset.classes[int(class_index)]
        Class_Name = str(Class_Name)
        return jsonify({'class_index': class_index, 'Class_Name': Class_Name})
        

if __name__ == '__main__':
    app.run()
