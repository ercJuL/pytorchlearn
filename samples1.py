from functools import reduce
import torch.utils.data as tud
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

toPILImage = ToPILImage()


def myshow(im_data):
    im = toPILImage(im_data)
    im.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = tv.datasets.CIFAR10(
    root='data/',
    train=True,
    download=True,
    transform=transform
)

trainloader = tud.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

testset = tv.datasets.CIFAR10(
    root='data/',
    train=False,
    download=True,
    transform=transform
)

testloader = tud.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, reduce(lambda i, j: i * j, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3
        return x


if __name__ == '__main__':
    # 显示一条数据
    # (data, lable) = trainset[100]
    # print(classes[lable])
    # myshow(data)

    # 用数据加载器加载一bach数据
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print('  '.join('%11s' % classes[labels[i]] for i in range(4)))
    # myshow(tv.utils.make_grid(images))

    # 实例化网络，定义损失函数
    net = Net()
    print(net)
    criterion = nn.CrossEntropyLoss()  # 交叉损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 训练网络
    for epoch in range(2):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # 输入数据
            inputs, lables = data
            inputs, lables = Variable(inputs), Variable(lables)
            # 梯度清零
            optimizer.zero_grad()
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印log信息
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # 每2000个batch打印一次训练状态
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Trainging')

    # 测试网络
    dataiter = iter(testloader)
    images, lables = dataiter.next()
    print('实际的label: ', ' '.join('%08s' % classes[lables[j]] for j in range(4)))
    myshow(tv.utils.make_grid(images / 2 - 0.5).resize((400, 100)))
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)  # 得分最高的类
    print('预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    correct = 0  # 预测正确图片数
    total = 0  # 总共的图片数
    for data in testloader:
        images, lables = data
        outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        total += lables.size(0)
        correct += (predicted == lables).sum()
    print('1000张测试集中的准确率为：%d %%' % (100 * correct / total))
