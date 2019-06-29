from vgg import *
import torch
import numpy as np
import sys
from utils import *
from dataset import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_model(path, device):
    net = vgg19_bn()
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    return net

def get_features(set, path):
    res = {}
    features_list = []
    with torch.no_grad():
        for i, dd in enumerate(set):
            image, data = dd
            # print(data.id)
            image = image.view(1, 1, 512, 512).to(device)
            output, features = net(image)
            # output = output.cpu().numpy().reshape(4)
            # output = np.exp(output) / np.exp(output).sum()
            # pred = output.argmax()
            features_list.append((features.cpu().numpy().reshape(1, -1))[0])

            if data.id in res:
                # res[data.id].append(pred)
                res[data.id].append(i)
            else:
                # res[data.id] = [pred]
                res[data.id] = [i]
    reduced_features_list = composition.fit_transform(features_list)
    # print(res)
    print(len(res))
    ans = []
    # for id, vote in res.items():
    #     num = np.zeros(4)
    #     for x in vote:
    #         num[x] += 1
    #     pred = num.argmax()
    #     ans.append((id, pred+1))
    # cd.generate_submit_file(ans, 'submit.csv')
    for id, vote in res.items():
        for x in vote:   
            new_tuple = (id,)+tuple(reduced_features_list[x])
            ans.append(new_tuple)
    cd.generate_submit_file(ans, path)
    
num_features = int(sys.argv[1])
model_path = 'model/test_train/58.pt'
if sys.argv[2] == 'tsne':
    composition = TSNE(n_components=num_features)
else:
    composition = PCA(n_components=num_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
net = load_model(model_path, device)
net.eval()
# print(net)
train_path = '../../cancer_detection/train'
test_path = '../../cancer_detection/test'
cd = CancerDetection()
cd.load_original_data(train_path, test_path)
validset = TestDataset(cd.test_data)
trainset = TestDataset(cd.train_data)
# res = {}
# features_list = []
# with torch.no_grad():
#     for i, dd in enumerate(validset):
#         image, data = dd
#         print(data.id)
#         image = image.view(1, 1, 512, 512).to(device)
#         output, features = net(image)
#         # output = output.cpu().numpy().reshape(4)
#         # output = np.exp(output) / np.exp(output).sum()
#         # pred = output.argmax()
#         features_list.append((features.cpu().numpy().reshape(1, -1))[0])

#         if data.id in res:
#             # res[data.id].append(pred)
#             res[data.id].append(i)
#         else:
#             # res[data.id] = [pred]
#             res[data.id] = [i]
# reduced_features_list = composition.fit_transform(features_list)
# # print(res)
# print(len(res))
# ans = []
# # for id, vote in res.items():
# #     num = np.zeros(4)
# #     for x in vote:
# #         num[x] += 1
# #     pred = num.argmax()
# #     ans.append((id, pred+1))
# # cd.generate_submit_file(ans, 'submit.csv')
# for id, vote in res.items():
#     for x in vote:   
#         ans.append((id, reduced_features_list[x][0], reduced_features_list[x][1]))
# cd.generate_submit_file(ans, 'test_feature.csv')
get_features(trainset, 'train_features.csv')
get_features(validset, 'test_features.csv')



