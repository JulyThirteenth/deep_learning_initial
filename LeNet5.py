import torch
from torch import nn

class FlattenLayer(nn.Module):
    def __init__(self) -> None:
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class LeNet5(nn.Module):
    """
    model for cifar dataset classfication
    """
    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        '''
        input shape: [batch_size, 3, 32, 32]
        output shape: [batch_size, 16, 5, 5]
        '''
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        '''
        input shape: [batch_size, 16, 5, 5]
        output shape: [10]
        '''
        self.net2 = nn.Sequential(
            FlattenLayer(), 
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

def main():
    model = LeNet5()
    print(model)
    tmp = torch.randn(4,3,32,32)
    print(tmp.shape)
    tmp = model(tmp)
    print(tmp.shape)


if __name__ == '__main__':
    main()