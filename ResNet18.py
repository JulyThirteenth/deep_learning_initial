import torch
from torch import nn

class FlattenLayer(nn.Module):
    def __init__(self) -> None:
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class ResBlk(nn.Module):
    '''
    resnet block
    '''
    def __init__(self, ch_in, ch_out, stride) -> None:
        '''
        input shape: [b, ch_in, h, w]
        output shape: [b, ch_out, h, w]
        '''
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out)
        )
 
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        tmp = self.extra(x)
        out = tmp + out
        return out

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            ResBlk(64, 128, stride=2),
            ResBlk(128, 256, stride=2),
            ResBlk(256, 512, stride=2),
            ResBlk(512, 512, stride=2)
        )
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            FlattenLayer(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.out(x)
        return x


def main():
    model = ResNet18()
    print(model)
    tmp = torch.randn(4,3,32,32)
    print(tmp.shape)
    tmp = model(tmp)
    print(tmp.shape)


if __name__ == '__main__':
    main()