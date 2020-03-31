from tkinter import *

import numpy as np 
import cv2
from PIL import Image

import torch

root = Tk()
root.title("Digit Prediction")

net = torch.load('net.model')
net.eval()

frame = Frame(root, width=250, height=50)
frame.pack()

canvas = Canvas(root, width=150, height=150, bg='white')

var = StringVar()

def extract_image(canvas):
    # save postscipt image as a tmp file
    canvas.postscript(file='tmp.eps') 
    # read that file into an image
    img = Image.open('tmp.eps') 
    img.save('tmp.png')

def click(click_event):
    global prev
    prev = click_event

def move(move_event):
    global prev
    canvas.create_line(prev.x, prev.y, move_event.x, move_event.y, width=10)

    prev = move_event

def clear():
    canvas.delete('all')
    var.set("")

def predict():
    extract_image(canvas)

    img = cv2.imread('tmp.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.expand_dims(img, 2)
    img = (255. - img) # invert the image because of pytorch mnist dataset
    img = img / 255. # normalize
    img = np.transpose(img, (2, 0, 1)) # convert to (1, 28, 28)
    img = np.array([img]) # convert to (1, 1, 28, 28)

    img = torch.from_numpy(img).float() 
    
    preds = net(img) # feed through network
    prediction = torch.argmax(preds)

    prediction = int(prediction.numpy())

    var.set("The prediction is {}".format(prediction))

# used to free draw
canvas.bind('<Button-1>', click)
canvas.bind('<B1-Motion>', move)

title = Label(root, text="Draw a digit 0-9!", font=("Helvetica", 15)).pack()

canvas.pack(padx=20, pady=20)
label = Label(root, textvariable=var, font=("Helvetica", 10)).pack()

b = Button(root, text="Predict", bg='#000', fg='#fff', command=predict)
b.pack(pady=10)

c = Button(root, text="Clear", bg='#000', fg='#fff', command=clear)
c.pack(padx=5, pady=10)

mainloop()