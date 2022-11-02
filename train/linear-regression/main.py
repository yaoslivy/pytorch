import numpy as np
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.nn.parameter import Parameter

# y = wx + b 
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.randn(1))
        self.bias = Parameter(torch.randn(1))

    def forward(self, input): 
        return (input * self.weight) + self.bias


if __name__ == "__main__":
    # Randomly generate datasets.
    w = 2 
    b = 3 
    xlim = [-10, 10]
    x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)
    y_train = [w * x + b + random.randint(0, 2) for x in x_train]
    plt.plot(x_train, y_train, 'bo')
    plt.savefig("./test.svg")
    
    # Train model.
    model = LinearModel()
    lr = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2, momentum=0.9)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    epochs = 1000
    for epoch in range(epochs):
        input = torch.from_numpy(x_train)
        output = model(input)
        # Gradient zeroed, calculate loss, backpropagation and optimized
        model.zero_grad()
        loss = nn.MSELoss()(output, y_train)
        print("epoch:", epoch+1, " loss:", loss)
        loss.backward()
        optimizer.step()
    
        
    for parameter in model.named_parameters():
        print(parameter)


    
    
