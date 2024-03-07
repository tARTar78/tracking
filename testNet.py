from PIL import Image
import torch
from torchvision import transforms
from efficientnet.models.efficientnet import EfficientNet, params
from torch import nn
from collections import OrderedDict
import cv2
import sys
import os
import matplotlib.pyplot as plt
import urllib
import numpy as np


def drawRect(contours, frame):
    rectangles = []
    res_rect = []

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        rectangles.append([x, y, x + w, y + h])
    rectangles.sort()

#     объединение

    for i in range(len(rectangles)):
        if not rectangles[i]:
            continue
            
        x, y, X, Y = rectangles[i]
            
        for j in range(len(rectangles)):
            if (i == j) | (not rectangles[j]):
                continue
            x_j, y_j, X_j, Y_j = rectangles[j]
            
            if (x_j <= x) & (X_j >= x):
                if ((y_j <= y) & (Y_j >= y)) | ((y_j <= Y) & (Y_j >= Y)):
                    x = x_j
                    y = min(y_j, y)
                    X = max(X_j, X)
                    Y = max(Y_j, Y)
                    rectangles[j] = []
                    
            elif (x_j <= X) & (X_j >= X):
                if ((y_j <= y) & (Y_j >= y)) | ((y_j <= Y) & (Y_j >= Y)):
                    x = min(x_j, x)
                    y = min(y_j, y)
                    X = X_j
                    Y = max(Y_j, Y)
                    rectangles[j] = []
            
        rectangles[i] = [x, y, X, Y]

# фильтрация
    for i in range(len(rectangles)):
        if not rectangles[i]:
            continue

        x, y, X, Y = rectangles[i]
        y -= 50

        if (X - x <= 25) | (Y - y <= 25):
            rectangles[i] = []
            continue

        for j in range(len(rectangles)):
            if (i == j) | (not rectangles[j]):
                continue
            x_j, y_j, X_j, Y_j = rectangles[j]
            y_j -= 50

            if (x_j <= x) & (X_j >= X) & (y_j <= y) & (Y_j >= Y):
                rectangles[i] = []
                break
            if (x <= x_j) & (X >= X_j) & (y <= y_j) & (Y >= Y_j):
                rectangles[j] = []
                continue

    return rectangles

def changeFrame(rectangles, frame):
    bright_matr = np.ones(frame.shape, dtype = "uint8") * 50
#     higher_contrast_matr = np.ones(frame.shape, dtype = "uint8") * 2.

    brighter_frame = cv2.add(frame, bright_matr)
#     higher_frame = np.uint8(cv2.multiply(np.float64(brighter_frame), higher_contrast_matr))

    for i in range(len(rectangles)):
        if rectangles[i]:
            x, y, X, Y = rectangles[i]
#             frame[y:Y, x:X] = higher_frame[y:Y, x:X]

            model.eval()
            with torch.no_grad():
                pil_image = Image.fromarray(cv2.cvtColor(frame[y:Y, x:X], cv2.COLOR_BGR2RGB))

                frame_copy = tfms(pil_image).unsqueeze(0)
                res = model(frame_copy)

            if (X - x < 120):
              cv2.rectangle(frame, (x - 3, y - 50), (x + 120, y), (255, 255, 255), -2)
            else:
              cv2.rectangle(frame, (x - 3, y - 50), (X, y), (255, 255, 255), -2)

            frame[y:Y, x:X] = brighter_frame[y:Y, x:X]
            # cv2.rectangle(frame, (x - 3, y), (X, Y + 45), (255, 255, 255), -2)
            # cv2.rectangle(frame, (x - 3, y - 50), (X, y), (255, 255, 255), -2)

            cv2.putText(frame, f'squirrel: {res[0][0].item():.3f}', (x, y - 35), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 0), thickness =1, lineType = cv2.LINE_AA)
            cv2.putText(frame, f'bird: {res[0][1].item():.3f}', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 0), thickness =1, lineType = cv2.LINE_AA)
            cv2.putText(frame, f'trash: {res[0][2].item():.3f}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 0), thickness =1, lineType = cv2.LINE_AA)
    

checkpoint = torch.load('/content/experiments/birds_squirrels/best.pth')
model = EfficientNet(1.0, 1.0, 0.2)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 3), nn.Softmax(dim=1))

state_dict = checkpoint['model']
model_dict = model.state_dict()
new_state_dict = OrderedDict()
matched_layers, discarded_layers = [], []
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]

    if k in model_dict and model_dict[k].size() == v.size():
        new_state_dict[k] = v
        matched_layers.append(k)
    else:
        discarded_layers.append(k)

model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

tracker = cv2.TrackerMIL_create()
video = cv2.VideoCapture("/content/x2mate_com_Videos_for_Cats_to_Watch_Birds_and_Squirrel_Fun_in_December.mp4")

video_w = int (video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int (video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_w, video_h))

ok, frame = video.read()
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prevFrame = grayFrame
while True:
    for i in range(10):
        ok, frame = video.read()
        if not ok:
            break
    if not ok:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayFrame, prevFrame)
    retval, img = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    rectangles = drawRect(contours, frame)
    #print(rectangles)
    changeFrame(rectangles, frame)
    
    prevFrame = grayFrame
    video_out.write(frame)
video_out.release()

