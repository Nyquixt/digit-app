import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

from models import Net

transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2, drop_last=True)  
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2, drop_last=True)

net = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

net.cuda()
net.train()

for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[Epoch: %d, Batch: %5d] Loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

        break
    break

net.eval()
net.to('cpu')
with torch.no_grad():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print('Test Accuracy of the network on the 10000 test images: {} %'.format(accuracy))


if accuracy > 85:
    torch.save(net, 'net.model')
