
# coding: utf-8

# In[1]:


from utils import *
import numpy as np
import six
if six.PY2:
	from Queue import Queue
else:
	from queue import Queue

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


def bfs(img, belong, x0, y0, K):
    m, n = img.shape
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
            if img[xx, yy] > 0 and belong[xx, yy] ==0:
                q.put((xx, yy))
                belong[xx, yy] = K


def getROI(img):
    m, n = img.shape
    belong = np.zeros((m, n))
    print(belong.shape)
    K = 0
    for i in range(m):
        for j in range(n):
            if img[i, j] > 0 and belong[i, j] == 0:
                K += 1
                bfs(img, belong, i, j, K)
    return (belong, K)

def show(img):
    plt.figure(figsize=(15, 10))
    plt.imshow(img, cmap='gray')
    plt.show()


# In[22]:

def convert_img(path, size):
	img, del_mask = read_image(path, size)
	# print(del_mask.nonzero())
	#show(img)
	#print(img)
	belong, K = getROI(img)


	# In[23]:


	num = np.zeros(K + 1).astype(np.int64)
	belong = belong.astype(np.int64)
	for i in range(belong.shape[0]):
		for j in range(belong.shape[1]):
			if belong[i, j] > 0:
				num[belong[i, j]] += 1


	# In[24]:


	t = num.argmax()
	ROI = (belong == t)
	f = img * ROI
	g = img * (ROI == False)
	minx, miny = 1000, 1000
	maxx, maxy = 0, 0
	for i in range(ROI.shape[0]):
	    for j in range(ROI.shape[1]):
		    if ROI[i][j] == 1:
			    minx = min(i, minx)
			    miny = min(j, miny)
			    maxx = max(i, maxx)
			    maxy = max(j, maxy)
	#print(minx)
	#print(miny)
	#print(maxx)
	#print(maxy)

	# ===============================================================
	# cv2.imwrite('ff.png', f) 	
	# cv2.imwrite('gg.png', g) 				
	# ===============================================================


	# In[25]:


	#show(img)
	#show(f)
	#show(g)


	# In[26]:


	num = np.zeros(256).astype(np.int64)
	#f_gray = get_gray(f)
	#g_gray = get_gray(g)
	m, n = img.shape
	for i in range(m):
		for j in range(n):
			num[g[i, j]] += 1


	# In[8]:


	threhold = [0, 0, 0]
	for k in range(3):
		maxxx = 0
		if k == 2:
			end = 256
		else:
			end = (k+1)*100
		for i in range(k*100, end):
			if num[i] > maxxx:
				maxxx = num[i]
				threhold[k] = i
		
	# print(threhold)
	#print(num[threhold[1]])
	#print(num)


	# In[27]:


	mask = np.zeros([m,n])
	for i in range(m):
		for j in range(n):
			#if ( threhold[0] >= f[i, j] >=0) | (threhold[1] >= f[i, j] >= 194) | (threhold[2] >= f[i, j] > 200):
			#if (threhold[1] >= f[i, j] > 100):
			if f[i, j] in threhold[1:]:
				if f[i, j] == 0:
					continue
				mask[i , j] = 1
				
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	dilation = cv2.dilate(mask, kernel)  
	#closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 闭运算
	f_binary = f*(1-dilation)
	#show(f_binary)


	# In[28]:


	dilation = dilation.astype(np.uint8)
	#mask = np.zeros((m,n))
	mask = (dilation==1) | (del_mask==1)
	# mask = del_mask
	mask = mask.astype(np.uint8)
	f_binary = f_binary.astype(np.uint8)
	#img = img.astype(np.uint8)
	# dst2 = cv2.inpaint(f_binary, mask, 1, cv2.INPAINT_NS)
	dst2 = cv2.inpaint(f_binary, mask, 1, cv2.INPAINT_TELEA)
	#cv2.imwrite('dst123.png', dst2[minx:maxx][miny:maxy])
	#dst2 = cv2.resize(dst2[minx:maxx][miny:maxy], size, interpolation=cv2.INTER_LANCZOS4)
	# cv2.imwrite('dst.png', dst2)
	#show(dst)
	#show(dst2)

	#=============================================== added by dxy ==========================================

	thresh = 0.4

	left = 0
	while (True):
		rate = 1.0 * (dst2[left, :] > 0).sum() / dst2.shape[1]
		if rate < thresh:
			left += 1
		else:
			break
	
	right = dst2.shape[0] - 1
	while (True):
		rate = 1.0 * (dst2[right, :] > 0).sum() / dst2.shape[1]
		if rate < thresh:
			right -= 1
		else:
			break
	
	up = 0
	while (True):
		rate = 1.0 * (dst2[:, up] > 0).sum() / dst2.shape[0]
		if rate < thresh:
			up += 1
		else:
			break
	
	down = dst2.shape[1] - 1
	while (True):
		rate = 1.0 * (dst2[:, down] > 0).sum() / dst2.shape[0]
		if rate < thresh:
			down -= 1
		else:
			break
	
	dst2 = dst2[left+10:right-10, up+10:down-10]

	# cv2.imwrite('result_' + path, dst2)
	return dst2


if __name__ == '__main__':
    # convert_img('test1.png', (512, 512))
    # convert_img('test2.png', (512, 512))
    # convert_img('test3.png', (512, 512))
    # convert_img('test4.png', (512, 512))
    # convert_img('test5.png', (512, 512))
	convert_img('1.bmp', (512, 512))
