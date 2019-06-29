# coding=utf-8
import pandas as pd
import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import stochastic_gradient
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


def concat_csv_data(data, features, columns):
    for name in columns:
        new_list = []
        for index in features.values:
            index = index[0]
            index = list(data['id']).index(index)
            new_list.append(data[name][index])
        features[name] = new_list
    return features


num_pic_features = int(sys.argv[1])
pic_features_columns = [str(x) for x in range(num_pic_features)]
other_features_columns = ['age', 'HER2', 'P53']
features_columns = pic_features_columns.copy()
features_columns.extend(other_features_columns)

train_data_columns = ['id', 'age', 'HER2', 'P53', 'label']
test_data_columns = ['id', 'age', 'HER2', 'P53']
pic_csv_columns = ['id']
pic_csv_columns.extend(pic_features_columns)

train_data = pd.read_csv(
    'train_feats.csv', engine='python',
    names=train_data_columns
)

train_features = pd.read_csv(
    'train_features.csv', engine='python',
    names=pic_csv_columns
)

data = concat_csv_data(train_data, train_features, ['age', 'HER2', 'P53', 'label'])
# print(data)
# 随机采用25%的数据用于测试，剩下的75%的数据用于训练集  random_state是随机数的种子，不同的种子会造成不同的随机采样结果，相同的种子采样结果相同
X_train, X_test, y_train, y_test = \
    train_test_split(data[features_columns],
                     data['label'], test_size=0.25)  
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
group_by_type_original = data.groupby('label').count()
# print(group_by_type_original)


# 使用SMOTE方法进行过抽样处理
model_smote = SMOTE()  # 建立SMOTE模型对象
x_smote, y_smote = model_smote.fit_sample(X_train, y_train)  # 输入数据并作过抽样处理
x_smote = pd.DataFrame(x_smote, columns=features_columns)  # 将数据转换为数据框并命名列名
y_smote = pd.DataFrame(y_smote, columns=['label'])  # 将数据转换为数据框并命名列名
smote = pd.concat([x_smote, y_smote], axis=1)  # 按列合并数据框
group_by_data_smote = smote.groupby('label').count()  # 对label做分类汇总
# print(group_by_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分布

X_train = smote[features_columns]
y_train = smote['label']

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导.
ss = StandardScaler()
ss = MinMaxScaler(feature_range=(0,1))
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

score = np.zeros(3)
model = []

# 3.使用逻辑斯蒂回归
lr = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, class_weight='balanced')  # 初始化LogisticRegression
lr.fit(X_train, y_train)  # 使用训练集对测试集进行训练
lr_y_predit = lr.predict(X_test)  # 使用逻辑回归函数对测试集进行预测
score[0] = lr.score(X_test, y_test) 
print('F1_macro of LR Classifier:%f' % f1_score(y_test, lr_y_predit, average='macro'))
model.append(lr)
print('Accuracy of LR Classifier:%f' % lr.score(X_test, y_test))
print(lr_y_predit)
print(classification_report(y_test, lr_y_predit, target_names=['1', '2', '3', '4'])) 


# 3.使用SVM训练
# svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
# svm.fit(X_train, y_train)  # 进行模型训练
# y_predict = svm.predict(X_test)
# print(y_predict)

svm = LinearSVC(multi_class='ovr', max_iter=10000, class_weight='balanced', dual=False)  # 初始化现行假设的支持向量机分类器 LinearSVC
svm.fit(X_train, y_train)  # 进行模型训练
y_predict = svm.predict(X_test)
print(y_predict)
score[1] = svm.score(X_test, y_test)
model.append(svm)
print('F1_macro of Linear SVC:', f1_score(y_test, lr_y_predit, average='macro'))
print('The Accuracy of Linear SVC is %f' % svm.score(X_test, y_test))  # 使用自带的模型评估函数进行准确性评测
print(classification_report(y_test, y_predict, target_names=['1', '2', '3', '4']))

# ada_real=AdaBoostClassifier(base_estimator=lr, learning_rate=0.5, n_estimators=300, algorithm='SAMME.R')
# ada_real.fit(X_train,y_train)
# y_predict = ada_real.predict(X_test)
# print(y_predict)
# score[2] = ada_real.score(X_test, y_test)
# model.append(ada_real)
# print('F1_macro of Ada real:', f1_score(y_test, lr_y_predit, average='macro'))
# print('The Accuracy of Ada real is %f' % ada_real.score(X_test, y_test))  # 使用自带的模型评估函数进行准确性评测
# print(classification_report(y_test, y_predict, target_names=['1', '2', '3', '4']))

test_data = pd.read_csv(
    'test_feats.csv', engine='python',
    names=test_data_columns)

test_features = pd.read_csv(
    'test_features.csv', engine='python',
    names=pic_csv_columns
)

test_features = concat_csv_data(test_data, test_features, other_features_columns)

id = test_data.iloc[:, 0]
transformed_test_data = ss.transform(test_features[features_columns])

using_model = model[score.argmax()]
predict = using_model.predict(transformed_test_data)

res = {}
for i, value in enumerate(test_features.values):
    index = value[0]
    if index in res:
        res[index].append(predict[i])
    else:
        res[index] = [predict[i]]

ans = []
for i in id:
    vote = res[i]
    num = np.zeros(4)
    for x in vote:
        num[x-1] += 1
    pre = num.argmax()
    ans.append((i, pre+1))

with open('submit-6.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ans)


