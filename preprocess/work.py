from utils import *
import numpy as np
from queue import Queue


def bfs(img, belong, x0, y0, K):
    m, n, _ = img.shape
    dx = [-1, 0, 0, 1]
    dy = [0, -1, 1, 0]
    q = Queue()
    q.put((x0, y0))
    belong[x0, y0] = K
    while not q.empty():
        x, y = q.get()
        for i in range(4):
            xx = x + dx[i]
            yy = y + dy[i]
            if xx < 0 or xx >= m or yy < 0 or yy >= n:
                continue
            if img[xx, yy].sum() > 0 and belong[xx, yy] ==0:
                q.put((xx, yy))
                belong[xx, yy] = K


def getROI(img):
    m, n, _ = img.shape
    belong = np.zeros((m, n))
    print(belong.shape)
    K = 0
    for i in range(m):
        for j in range(n):
            if img[i, j].sum() > 0 and belong[i, j] == 0:
                K += 1
                bfs(img, belong, i, j, K)
    print(belong)


if __name__ == '__main__':
    img = read_image('temp.jpg')
    print(img.shape)
    getROI(img)
