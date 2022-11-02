import torch.nn as nn
import torch
import torch.optim as optim


# Define one model
class MyModel(nn.Module):
    def __init__(self) -> None:
        pass
    
    def forward(self):
        pass


if __name__ == "__main__":
    #The model that needs to be trained.
    net = MyModel()
    #Define one loss function, learning rate and optimize function.
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #The number of times all data needs to be trained.
    epochs = 30

    traindata = torch.tensor([]) # Hypothetical training data
    for epoch in range(epochs):
        for i, data in enumerate(traindata):
            # One batch size data to train.
            inputs, labels = data 
            #First, the gradient needs to be zeroed，if not, the gradient of the first time will be added to the second calculation.
            optimizer.zero_grad()
            #Get the output of the model.
            outputs = net(inputs)
            #Calculate thr difference between the predicted and true values.
            loss = criterion(outputs, labels)
            #The backpropagation is performed and then optimized according to the gradient.
            loss.bachward()
            optimizer.step()
            
            

