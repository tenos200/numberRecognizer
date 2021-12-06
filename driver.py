import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
import os
import pygame as pg


# 1.design model (input, output size, forward pass)
class Model(nn.Module):
    def __init__(self, input_size, output_size, class_digits):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)
        #activation function 
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(output_size, class_digits)
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out 

def runModel():
    #0.1
    #we define the dimensions of the input 28*28 and all possible labels from 0-9 i.e, 10 labels
    input_size = 784 
    class_digits = 10 
    output_size = 100

    # 0. Load the data
    trainset = datasets.MNIST(r'./data', download=True, train=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST(r'./data', download=True, train=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    #instantiate model
    model = Model(input_size, output_size, class_digits)
    # 2.construct loss and optimizer
    learning_rate = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    #we can use SGD to Adam depening of what we want to do there
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 3.training loop
    epochs = 50 
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            # - forward pass: compute prediction
            images = images.reshape(-1, 28*28)
            output = model(images)
            loss = criterion(output, labels)
            # - backward pass: gradients
            loss.backward()
            # - update weights
            optimizer.step()
            print(f'Epoch: {epoch+1} / {epochs}, Training_loss = {loss.item():.4f}')

    #test model
    with torch.no_grad():
        correct = 0
        samples = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28*28)
            output = model(images)
            _, predictions = torch.max(output, 1)
            samples += labels.shape[0]
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / samples
    print(f'Accuracy = {accuracy}')
    torch.save(model, './models/NNmodel.pth')


def pygameRunner(model):
    
    run = True
    drawing = False 
    width = 400
    height = 400
    pg.init()
    clock = pg.time.Clock()

    #set up the size of the window
    window = pg.display.set_mode((width, height))
    #set background color to black
    window.fill((0,0,0))
    #set window title
    pg.display.set_caption('MNIST recognizer')

    #pygame event loop
    pg.display.update()
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                #pg.image.save(window, 'test.jpg')
                pg.quit()
                run = False
                return False
            if event.type == pg.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pg.MOUSEBUTTONUP:
                drawing = False 
            if event.type == pg.MOUSEMOTION:
                if drawing:
                    x, y = pg.mouse.get_pos()
                    #sort a better way to draw
                    for i in range(5):
                        window.set_at((x, y),(255, 255, 255))
                        window.set_at((x-i, y-i),(255, 255, 255))
                        window.set_at((x+i, y+i),(255, 255, 255))

        pg.display.flip()
        clock.tick(60)


if __name__ == "__main__":                                   
    try:
        #run function                                            
        model = torch.load('./models/NNmodel.pth')
        model.eval()
        print('Loaded successfully!')
        ret = pygameRunner(model)
        if not ret:
            os._exit(1)

        '''
        with torch.no_grad():
            output = model(tester)
            predictions = torch.max(output, 1)
            print(f'Prediction = {predictions}')
        '''
    #if the model is not loaded then we train the model
    except:
        print('failed')
        #runModel()                                                    
