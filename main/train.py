import torch
from torch.nn import DataParallel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from vgg import *
from utils import *


def train(net, device, train_dataset_loader, criterion, optimizer, epoch):
    net.train()
    for cur_epoch in range(epoch):
        ll = 0
        for batch_id, data in enumerate(train_dataset_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            ll += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
#             print('epoch = %s,  batch_id = %s,  loss = %s' % (cur_epoch, batch_id, loss.item()))
#             print(pred.view(-1), labels.view(-1))
        print('epoch = %s,  loss = %s' % (cur_epoch, ll / len(train_dataset_loader)))
        torch.save(net.module.state_dict(), 'model/first_train/' + str(cur_epoch) + '.pt')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    cd = CancerDetection()
    train_path = '../../cancer_detection/train'
    test_path = '../../cancer_detection/test'
    cd.load_original_data(train_path, test_path)
    trainset = TrainDataset(cd.train_data)
    train_dataloader = DataLoader(trainset, batch_size=10, shuffle=True)
    net = vgg11()
    net = DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, device, train_dataloader, criterion, optimizer, 10)
