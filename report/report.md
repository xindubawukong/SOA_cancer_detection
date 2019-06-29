# SOA Final Project Report



Team member:

| Name          | Student Number |
| ------------- | -------------- |
| Xiangyun Ding | 2016011361     |
| Yifan Yin     | 2016011368     |





## 1 Overview

### Background Introduction

Throughout the world, breast cancer is one of the leading causes of female death. Molecular subtyping of breast cancer has become common practice to understand prognosis of disease, and to design a treatment plan. The subtype indicates the severity of the cancer and influences the treatment plan. This project is to develop an automated method to classify the molecular subtype of breast cancer based on ultrasound images and clinical diagnostic data. 

Given hundreds of medical records of breast cancer patients. Each medical record is associated with several ultrasound images and some clinical diagnostic data. The clinical diagnostic data contains the following fields:

![image-20190629184655325](/Users/xdbwk/Desktop/thu32/soa/SOA_cancer_detection/report/report.assets/image-20190629184655325.png)

The competition details can be viewed on <https://biendata.com/competition/detection/>.

### Submitted Files List

```
.
├── code
│   ├── classification
│   └── preprocess
└── report.pdf
```

The code folder contains all code for our tow steps( pre-processing and classification).



## 2 Method

### Ultrosound Image Pre-processing

![image-20190629185109992](/Users/xdbwk/Desktop/thu32/soa/SOA_cancer_detection/report/report.assets/image-20190629185109992.png)

As shown in this figure, artifacts such as fiducial markers added by radiologists degraded the learning process. In order to remove useless annotations, we researched many pre-processing method. Our final pre-processing method is as follows:

1 Apply 2D connected component algorithm on the binary image, resulting in a labelled image with K components.

2 Identify the texture region as the largest 2D connected component in the labelled image and subtracted it from the image to get the “possible artifact” regions.

3 Plot the histogram of the “possible artifact” regions and divided the histogram into three parts [0,100], [101,200], [201,255], respectively.

4 Take the histogram peaks in each of the three parts as the intensity levels of the artifacts and subtract from the original image to generate the artifact-removed image.

We then used the `Inpaint` function in OpenCV to restore the images. The restore method is Navier-Stokes. After removing annotations and restoring, now the images are like this:

![image-20190629185752523](/Users/xdbwk/Desktop/thu32/soa/SOA_cancer_detection/report/report.assets/image-20190629185752523.png)

To reach best learning result, we then run data normalization and augmentation. All images were resized to size $512\times512$ to fit the input of neural networks, and then  were randomly flipped, rotated or color jittered. Finally, the four classes were made to have equal numbers of images.

### Classification

The input is a $512\times512$ size image and three clinical diagnostic data: age, HER2 and P53. We are expected to classify each image to one of the four classes.

All deep neural networks were implemented with <a href='<https://pytorch.org/>'>Pytorch</a>.

#### SVM Classifier

First, we trained a simple SVM classifier with only the clinical diagnostic data, and reached the f1-score of 0.56. The code can be found in `code/classification/SVM.py`.

#### Simple Neural Network

We trained a simple VGG network(the model is VGG_19 with batch-normalization) with only the ultrasound images, and reached the f1-score of 0.39. This result is even lower than SVM. For one patient with several ultrasound images, we predict the label of each image and vote to the final result.

#### Combination of SVM and Neural Network

We tried this method to combine the clinial diagnostic data and ultrasound images, which can be stepped as follows:

1 Extracted the features of the images(vectors of length 512) from the feature-layers output of our VGG neural network.

2 Dimensionality reduction to vectors of length 5. According to experiments results, PCA outperforms t-SNE. This may because t-SNE performs better for visualization.

3 Combine this vector with the three clinical diagnostic data to vectors of length 8. These vectors were then sent to train the SVM classifier.

The best result of this method is 0.625.

#### Final Network Structure

According to the advice of the teaching assistant, we tried another method which only use the neural network.

![image-20190629201404174](/Users/xdbwk/Desktop/thu32/soa/SOA_cancer_detection/report/report.assets/image-20190629201404174.png)

This method reached the f1-score of 0.639, which outperforms the last method.

#### More Complex Network

We then tried another network DRN(Dilated Residual Networks), which captures both the global and local features without losing the perception field. This network outperformed VGG with f1-score 0.667. (Rank 6 currently).

![image-20190629201532990](/Users/xdbwk/Desktop/thu32/soa/SOA_cancer_detection/report/report.assets/image-20190629201532990.png)



## Work Division

Xiangyun Ding:

Build up neural networks and parameters tuning.



Yifan Yin:

Pre-processing and SVM classifier.



## Summary

Our team is finally rank 6 on the leaderboard. During this process, we learned a lot from SOA classes, Prof. Tang and Prof. Li, teaching assistants and our classmates. Thank you very much.

Seperatly using the clinical diagnostic data or ultrasound images didn't perform well on this task. All input data is meaningful for classification. Our first method to combine these two data sources has two steps to compress the input data( neural network for features and SVM classifier), which lost some meaningful information during the process. By directly plugging the clinical diagnostic data into the neural network, we abtained a good model for classification. At last, we replaced VGG by DRN, which performed best.

This project is more than a homework for SOA class. It is a precious asset for our college life.



## References

[1] Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

[2] Yu, F., Koltun, V. and Funkhouser, T., 2017. Dilated residual networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 472-480).

[3] Chi, J., Walia, E., Babyn, P., Wang, J., Groot, G. and Eramian, M., 2017. Thyroid nodule classification in ultrasound images by fine-tuning deep convolutional neural network. *Journal of digital imaging*, *30*(4), pp.477-486.