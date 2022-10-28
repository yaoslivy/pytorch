import torchvision

from torchvision import datasets, transforms


if __name__ == "__main__":
    mnist_dataset = torchvision.datasets.MNIST(root='./data',
                               train=True,
                               transform=None,
                               target_transform=None,
                               download=True)

    print(len(mnist_dataset.train_data))
    print(mnist_dataset[0])

    print("**transform operation**")
    my_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5), (0.5))
                                      ])
    # Read MNIST dataset and perform data transformation at the same time
    mnist_dataset = datasets.MNIST(root='./data',
                                   train=False,
                                   transform=my_transform,
                                   target_transform=None,
                                   download=True)
    item = mnist_dataset.__getitem__(0)
    print(type(item[0]))
