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

def run():
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3.training loop
    epochs = 10 
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

if __name__ == "__main__":                                   
    #run function                                            
    if torch.load('./models/NNmodel.pth'):
        model = torch.load('./models/NNmodel.pth')
        model.eval()
        print('Loaded!')
        #we want to use this a representation of what the input will be later on 
        
        test = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], 
                ]
        input_test = torch.tensor(test).float()
        tester = input_test.reshape(-1, 28*28)
        with torch.no_grad():
            output = model(tester)
            predictions = torch.max(output, 1)
            print(f'Prediction = {predictions}')
    else:
        run()                                                    
