import cv2


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return res


if __name__ == '__main__':
    print(read_image('temp.jpg').shape)
