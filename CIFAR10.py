import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from LeNet5 import LeNet5
from ResNet18 import ResNet18

def main():
    cifar = CIFAR10('./data/', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    train_batchsize = 64
    cifar_train = DataLoader(cifar, batch_size=train_batchsize, shuffle=True)
    test_batchsize = 256
    cifar_test = DataLoader(cifar, batch_size=test_batchsize, shuffle=False)
    # x, label = next(iter(cifar_train))
    # print("x shape: {}, label shape: {}.".format(x.shape, label.shape))

    device = torch.device('cuda')
    model = LeNet5().to(device)
    cretion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range (1000):
        model.train()
        for batch_idx, (x, lable) in enumerate(cifar_train):
            x, lable = x.to(device), lable.to(device)
            logits = model(x)
            loss = cretion(logits, lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(cifar_train.dataset),
                    100. * batch_idx  / len(cifar_train), loss.item()
                ))
        with torch.no_grad():
            correct_nums = 0
            total_nums = 0
            model.eval()
            for x, lable in cifar_test:
                x, lable = x.to(device), lable.to(device)
                x = model(x)
                pred = x.argmax(dim=1)
                correct_nums += torch.eq(pred, lable).float().sum().item()
                total_nums += x.size(0)
            accuracy = correct_nums / total_nums
            print("epoch: {}\tloss: {}\taccuracy: {}".format(epoch, loss.item(), accuracy))

if __name__ == '__main__':
    main()
