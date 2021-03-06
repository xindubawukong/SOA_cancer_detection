import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from utils import *


# Train dataset for pytorch input.
class TrainDataset(Dataset):

    def __init__(self, data_list):
        self.images = []
        for data in data_list:
            image_path = data.image_path
            self.images.append((image_path, data.subtype))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path, label = self.images[item]
        image = cv2.imread(image_path)
        image = get_gray(image)
        image = cv2.resize(image, (512, 512))
        image = Image.fromarray(np.uint8(image))
        color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        image = color_jitter(image)
        if np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        num_rotate = np.random.randint(0, 4)
        image = image.rotate(90 * num_rotate)
        image = np.asarray(image)
        image = image[None, :]
#         image = image.transpose((2, 0, 1))
        image = (image / 255.0).astype(np.float32)
        return torch.from_numpy(image), torch.Tensor([label]).long()


# Valid dataset for pytorch input.
class ValidDataset(Dataset):

    def __init__(self, data_list):
        self.images = []
        for data in data_list:
            self.images.append(data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        data = self.images[item]
        image_path = data.image_path
        label = data.subtype
        image = cv2.imread(image_path)
        image = get_gray(image)
        image = cv2.resize(image, (512, 512))
        image = image[None, :]
        image = (image / 255.0).astype(np.float32)
        return torch.from_numpy(image), torch.Tensor([label]).long()


# Test dataset for pytorch input.
class TestDataset(Dataset):

    def __init__(self, data_list):
        self.images = []
        for data in data_list:
            self.images.append(data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        data = self.images[item]
        image_path = data.image_path
        image = cv2.imread(image_path)
        image = get_gray(image)
        image = cv2.resize(image, (512, 512))
        image = image[None, :]
        image = (image / 255.0).astype(np.float32)
        return torch.from_numpy(image), data


if __name__ == '__main__':
    train_path = '../../cancer_detection/train'
    test_path = '../../cancer_detection/test'
    cd = CancerDetection()
    cd.load_original_data(train_path, test_path)
    trainset = TrainDataset(cd.train_data)
    validset = ValidDataset(cd.valid_data)
    for image, label in validset:
        print(image.size(), image.dtype, label)
        break
