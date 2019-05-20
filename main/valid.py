from vgg import *
import torch
import numpy as np
from utils import *
from dataset import *


def load_model(path, device):
    net = vgg11()
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    return net


model_path = '../experiment/new_retrain/20%_0.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
net = load_model(model_path, device)
net.eval()
print(net)
train_path = '../../cancer_detection/train'
test_path = '../../cancer_detection/test'
cd = CancerDetection()
cd.load_original_data(train_path, test_path)
validset = ValidDataset(cd.valid_data)
tot = 0
correct = 0
with torch.no_grad():
    for i, data in enumerate(validset):
        image, label = data
        image = image.view(1, 3, 512, 512).to(device)
        output = net(image)
        output = output.cpu().numpy().reshape(4)
        output = np.exp(output) / np.exp(output).sum()
        pred = output.argmax()
        tot += 1
        if pred == label:
            correct += 1
print(correct, tot, 1.0 * correct / tot)
