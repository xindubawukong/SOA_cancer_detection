import os
import csv
import cv2
import random


# Return a numpy array with size [m * n * 3]
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return res


class Data(object):

    # For test data, the sub_type variable is set None.
    def __init__(self, id, age, HER2, P53, sub_type, image_path):
        self.id = id
        self.age = age
        self.HER2 = HER2
        self.P53 = P53
        self.subtype = sub_type
        self.image_path = image_path


class CancerDetection(object):

    def __init__(self):
        self.train_data = []
        self.valid_data = []
        self.test_data = []

    # Load all images from train and test folder. Split train data into train and valid.
    def load_original_data(self, train_path, test_path):
        csv_reader = csv.reader(open(os.path.join(train_path, 'feats.csv'), 'r'))
        content = []
        for line in csv_reader:
            content.append(line)
        all_data = []
        for item in content[1:]:
            path = os.path.join(train_path, 'images', item[0])
            for cur_dir, dirs, files in os.walk(path):
                for file in files:
                    if file[:5] != 'new1_':
                        continue
                    image_path = os.path.join(cur_dir, file)
                    data = Data(id=item[0], age=int(item[1]), HER2=int(item[2]), P53=bool(item[3]),
                                sub_type=int(item[4]) - 1, image_path=image_path)
                    all_data.append(data)
        random.shuffle(all_data)
        self.train_data = all_data[:825]
        self.valid_data = all_data[825:]

        csv_reader = csv.reader(open(os.path.join(test_path, 'feats.csv'), 'r'))
        content = []
        for line in csv_reader:
            content.append(line)
        self.test_data = []
        for item in content[1:]:
            path = os.path.join(test_path, 'images', item[0])
            for cur_dir, dirs, files in os.walk(path):
                for file in files:
                    if file[:5] != 'new1_':
                        continue
                    image_path = os.path.join(cur_dir, file)
                    data = Data(id=item[0], age=int(item[1]), HER2=int(item[2]), P53=bool(item[3]),
                                sub_type=None, image_path=image_path)
                    self.test_data.append(data)

        print('All data loaded.\nNumber of each dataset:\nTrain: %d\nValid: %d\nTest: %d\n'
              % (len(self.train_data), len(self.valid_data), len(self.test_data)))

    # Generate a submit file to path. The result should be a list. Each item of the list should be a tuple, with first
    # element is the patient id, and second element is the cancer sub-type.
    def generate_submit_file(self, result, path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(result)


if __name__ == '__main__':
    train_path = '../../cancer_detection/train'
    test_path = '../../cancer_detection/test'
    cd = CancerDetection()
    cd.load_original_data(train_path, test_path)
    cd.generate_submit_file([('qweqwe', 1), ('asdfasdf', 0)], 'submit.csv')
