from torchvision import torch
import torchvision.models as models


if __name__ == "__main__":
    # https://pytorch.org/vision/stable/models.html
    # Load pre-train models
    googlenet = models.googlenet(pretrained=True)

    # Extract the input parameters of the classification layer
    fc_in_features = googlenet.fc.in_features
    print("fc_in_features:", fc_in_features)

    # Extract the output parameters of the classification layer
    fc_out_features = googlenet.fc.out_features
    print("fc_out_features:", fc_out_features)

    # Modify the number of output classification of the pre-train model 
    googlenet.fc = torch.nn.Linear(fc_in_features, 10)


    

