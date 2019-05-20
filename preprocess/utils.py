import cv2
from PIL import Image
import numpy as np

def read_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if size != (0, 0):
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    m, n, k = img.shape
    mask = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if (img[i][j][0] == img[i][j][1]) & (img[i][j][1] == img[i][j][2]):
                continue
            mask[i][j] = 1
            img[i][j] = [0, 0, 0]
                
    cv2.imwrite('why.png', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int)
    
    return img, mask


def get_gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return res


if __name__ == '__main__':
    print(read_image('temp.jpg').shape)
