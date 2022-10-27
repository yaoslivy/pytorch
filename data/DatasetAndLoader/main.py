from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


if __name__ == "__main__" :
    # Generate data randomly
    data_tensor = torch.randn(10, 3)
    target_tensor = torch.randint(2, (10,))  # target is 0 or 1

    my_dataset = MyDataset(data_tensor, target_tensor)

    print('Dataset size:', len(my_dataset))
    # Get data by index.
    print('tensor_data[0]: ', my_dataset[0])
     
    # Build a dataloader
    tensor_dataloader = DataLoader(dataset= my_dataset,   # Required
               batch_size= 2,
               shuffle= True,
               num_workers= 0)
    
    # Loop output all.
    for data, target in tensor_dataloader:
        print(data, target)


    print("One batch tensor data:", iter(tensor_dataloader).next())
