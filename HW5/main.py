import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%matplotlib inline
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('figure/random_images.png')
    plt.show()

def imshow2(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('figure/random_images2.png')
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### START CODE HERE ###
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3,128 ) # first hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear( 128,64 ) #second hidden layer
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64 , 10) 
        self.relu = nn.ReLU()
        ### END CODE HERE ###

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### START CODE HERE ###
        self.conv1 = nn.Conv2d(3,6 ,5 )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d( 6,16 ,5 )
        self.fc1 = nn.Linear(16 * 5 * 5,120 )
        self.fc2 = nn.Linear(120 ,84 )
        self.fc3 = nn.Linear( 84, 10)
        ### END CODE HERE ###

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# get some random training images
if __name__ == '__main__':

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_values = []
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            ### START CODE HERE ###

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            ### END CODE HERE ###
            running_loss += loss.item()
            #print(loss_values)
            # print statistics
            if i % 2000 == 1999:  # print every 2000 mini-batches
            # epoch loss for our plot
                epoch_loss = (running_loss / 2000)
            #adding to our array of epoch_loss
                loss_values.append(epoch_loss)
                print(loss_values)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    plt.style.use('ggplot')
    plt.plot(loss_values, label = 'Training loss')
    plt.legend()
    plt.savefig('figure/training_graph.png')
    plt.show()
    print(loss_values)
    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            ### START CODE HERE ###
            
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ### END CODE HERE ###

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    transform_CNN = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset_CNN = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_CNN)
    trainloader_CNN = torch.utils.data.DataLoader(trainset_CNN, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset_CNN = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_CNN)
    testloader_CNN = torch.utils.data.DataLoader(testset_CNN, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # functions to show an image

    # get some random training images
    dataiter = iter(trainloader_CNN)
    images, labels = dataiter.next()

    # show images
    imshow2(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    net_CNN = Net_CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net_CNN.parameters(), lr=0.001, momentum=0.9)

    loss_values = []
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader_CNN, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            ### START CODE HERE ###

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            out = net_CNN(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            ### END CODE HERE ###

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
            # epoch loss for our plot
                epoch_loss = (running_loss / 2000)
            #adding to our array of epoch_loss
                loss_values.append(epoch_loss)
                print(loss_values)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
    plt.style.use('ggplot')
    plt.plot(loss_values, label = 'Training loss')
    plt.legend()
    plt.savefig('figure/training_graph_2.png')
    plt.show()
    print(loss_values)
    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader_CNN:
            images, labels = data
            
            ### START CODE HERE ###
            
            # calculate outputs by running images through the network
            outputs = net_CNN(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ### END CODE HERE ###

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))