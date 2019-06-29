import torch
from torch.nn import DataParallel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from vgg import *
from utils import *


def valid(net, validset):
    net.eval()
    tot = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(validset):
            image, label = data
            image = image.view(1, 1, 512, 512).to(device)
            output, features = net(image)
            output = output.cpu().numpy().reshape(4)
            output = np.exp(output) / np.exp(output).sum()
            pred = output.argmax()
            tot += 1
            if pred == label:
                correct += 1
    print(correct, tot, 1.0 * correct / tot)


def train(net, device, train_dataset_loader, validset, criterion, optimizer, epoch):
    for cur_epoch in range(epoch):
        net.train()
        ll = 0
        for batch_id, data in enumerate(train_dataset_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = (net(inputs))[0]
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            ll += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
#             print('epoch = %s,  batch_id = %s,  loss = %s' % (cur_epoch, batch_id, loss.item()))
#             print(outputs)
#             print(pred.view(-1), labels.view(-1))
        print('epoch = %s,  loss = %s' % (cur_epoch, ll / len(train_dataset_loader)))
        torch.save(net.module.state_dict(), 'model/test_train/' + str(cur_epoch) + '.pt')
        valid(net, validset)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    cd = CancerDetection()
    train_path = '../../cancer_detection/train'
    test_path = '../../cancer_detection/test'
    cd.load_original_data(train_path, test_path)
    trainset = TrainDataset(cd.train_data)
    validset = ValidDataset(cd.valid_data)
    train_dataloader = DataLoader(trainset, batch_size=24, shuffle=True)
    net = vgg19_bn()
    net = DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
    train(net, device, train_dataloader, validset, criterion, optimizer, 100)
