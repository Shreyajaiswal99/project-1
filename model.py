## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p = 0.1)
        # output = (32, 220, 220)
        # Maxpooled = (32, 110 , 110)
        
        
        self.conv2 = nn.Conv2d(32, 64 ,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout(p = 0.2)
        # output = (64, 108 ,108)
        # Maxpooled output = (64, 54, 54)
        self.fc3 = nn.Linear(64*54*54, 9000)
        self.drop3 = nn.Dropout(p = 0.4)
        self.fc4 = nn.Linear(9000, 2500)
        self.drop4 = nn.Dropout(p = 0.4)
        self.fc5 = nn.Linear(2500, 136)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        
        ## TODO: Define the feedforward behavior of this model

        ## x is the input image and, as an example, here you may choose to include a pool/conv step
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        x = F.relu(self.fc4(x))
        x = self.drop4(x) 
               
        x = F.relu(self.fc5(x))
        
        
        return x
                      
                       
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
