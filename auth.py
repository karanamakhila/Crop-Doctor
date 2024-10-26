# from crypt import methods
from email import message
from typing import Collection
import pymongo
from flask import Blueprint , render_template , url_for , redirect , request , session
import bcrypt
import random
import smtplib
import math
# from PIL import Image
from bson import Binary
import io
from flask import Flask
from flask_pymongo import PyMongo
from pymongo import MongoClient
# from scipy.misc import imsave,imread,imresize
import numpy as np
import keras.models
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import re
# import Sys
import os
import base64
from keras.preprocessing import image
# from load import *
import os
import torch
import dill as dill
from torchvision import models
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.datasets import ImageFolder  # for working with classes and images
import torch.nn.functional as F # for functions for calculating loss
import torch.nn as nn           # for creating  neural networks
import numpy as np 
from PIL import Image
from resnet import ResNet
import wikipedia
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
# resnet = ResNet()

classes = open('class.txt',"r").read().split(',')
client=pymongo.MongoClient("mongodb+srv://Karanam_Akhila:An12$shi@cluster0.h1vwyen.mongodb.net/?retryWrites=true&w=majority")
db=client["Farmers"]
collection=db["FarmerDetails"]
EMAIL_ADDRESS = config.EMAIL_ADDRESS
EMAIL_PASSWORD = config.EMAIL_PASSWORD

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): 
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out  
model = ResNet9(3,38)
PATH = 'C:/Users/karan/Downloads/CropDoctor-main/CropDoctor-main/plant-disease-model.pth'
# PATH='C:/Users/karan/Resnet/RESNET50_PLANT_DISEASE.h5'
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
model.eval()
model.double()


auth=Blueprint("auth",__name__)

@auth.route("/",methods=['GET','POST'])
def main():
    return render_template("Main.html")
    
    

@auth.route("/login",methods=['GET','POST'])
def login():
    if request.method=="POST":
        if request.form.get("buttons")=="register":
            return redirect("/register")
        fullname=request.form.get("fname")
        emailid=request.form.get("email")
        password=request.form.get("password")
        data=collection.find_one({"Email":emailid})
        if data:
            if password==data["password"]:
                return redirect("/detect")
            else:
                msg="Incorrect Password"
                return render_template("login.html",msg=msg)
        else:
            msg="Info Not Found!! Please Register"
            return render_template("login.html",msg=msg)
    return render_template("login.html")

def send_otp(email):
    # Generate a 6-digit OTP
    otp = random.randint(100000, 999999)
    session['otp'] = otp  # Store OTP in session

    # Email content
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg['Subject'] = "CropDoctor - Your OTP Code"
    body = f"Your OTP is: {otp}"
    msg.attach(MIMEText(body, 'plain'))

    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, email, text)
        server.quit()
        return True
    except Exception as e:
        print("Error:", e)
        return False




# @auth.route("/register",methods=['GET','POST'])
# def register():
#     if request.method=="POST":
#         fullname=request.form.get("fname")
#         emailid=request.form.get("email")
#         phone=request.form.get("phno")
#         password1=request.form.get("password1")
#         password2=request.form.get("password2")
#         if(password1==password2):
#             dict={"Name":fullname,"Email":emailid,"Phone Number":phone,"password":password1}
#             collection.insert_one(dict)
#             return redirect("/detect")
#         else:
#             msg="Incorrect Confirm Password"
#             return render_template("register.html",msg=msg)
#     return render_template("register.html")
@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form.get("fname")
        emailid = request.form.get("email")
        phone = request.form.get("phno")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        
        email_regex = r"[^@]+@[^@]+\.[^@]+"
        password_regex = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"

        if not re.match(email_regex, emailid):
            msg = "Invalid Email Format"
            return render_template("register.html", msg=msg)

        if not re.match(password_regex, password1):
            msg = "Password must include uppercase, lowercase, digit, special character, and be 8+ characters."
            return render_template("register.html", msg=msg)

        if password1 != password2:
            msg = "Passwords do not match."
            return render_template("register.html", msg=msg)
        
        if collection.find_one({"Email": emailid}):
            msg = "User already exists, please log in."
            return render_template("register.html", msg=msg)

        session['user_data'] = {"Name": fullname, "Email": emailid, "Phone Number": phone, "password": password1}

        if send_otp(emailid):
            return redirect("/verify_otp")
        else:
            msg = "Error sending OTP. Please try again."
            return render_template("register.html", msg=msg)
    
    return render_template("register.html")

@auth.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        entered_otp = request.form.get("otp")
        
        # Check if OTP matches
        if 'otp' in session and int(entered_otp) == session['otp']:
            user_data = session.pop('user_data', None)  # Retrieve and remove user data from session
            collection.insert_one(user_data)  # Save user data in the database
            session.pop('otp', None)  # Remove OTP from session
            return redirect("/detect")  # Success redirection
        
        msg = "Invalid OTP. Please try again."
        return render_template("verify_otp.html", msg=msg)

    return render_template("verify_otp.html")


@auth.route("/detect",methods=['GET','POST'])
def detect():
    if request.method=="POST":
        file=request.files['some_img']
        filename=file.filename
        file_path=os.path.join(r'C:/Users/karan/Downloads/CropDoctor-main/CropDoctor-main/static',filename)
        file.save(file_path)
        detection = predict_image(file_path)
        detection_split = detection.split('___')
        plant, disease = detection_split[0], detection_split[1]
        print(file_path)
        print(disease)
        if disease=='healthy':
            res="This is an healthy "+plant+" leaf"
            return render_template("result.html",img=filename,di="nothing",res=res)
        else:
            return render_template("result.html",img=filename,di=disease,res=wikipedia.summary(plant+disease),sentences=250)
    
    return render_template("home.html")
def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

def predict_image(image):
        print(image)
        image = np.double(
            Image.open(image).convert("RGB").resize((256, 256))
        )
        image = image/255.0
        img = transforms.ToTensor()(image).double()
        xb = to_device(img.unsqueeze(0),"cpu")
        yb = model(xb)
        _, preds  = torch.max(yb, dim=1)
        return classes[preds[0].item()]
