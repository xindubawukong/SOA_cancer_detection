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


model_path = 'model/first_train/9.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
net = load_model(model_path, device)
net.eval()
print(net)
train_path = '../../cancer_detection/train'
test_path = '../../cancer_detection/test'
cd = CancerDetection()
cd.load_original_data(train_path, test_path)
validset = TestDataset(cd.test_data)
res = []
with torch.no_grad():
    for i, dd in enumerate(validset):
        image, data = dd
        print(data.id)
        image = image.view(1, 1, 512, 512).to(device)
        output = net(image)
        output = output.cpu().numpy().reshape(4)
        output = np.exp(output) / np.exp(output).sum()
        pred = output.argmax()
        res.append((data.id, pred + 1))
cd.generate_submit_file(res, 'submit.csv')
