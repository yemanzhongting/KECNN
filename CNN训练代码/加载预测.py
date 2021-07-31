import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle

import keras.backend as K
from keras import Sequential
from keras.layers import Dense
import numpy as np

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall

#加载模型h5文件
# model = load_model("new.h5")
model = load_model("new.h5", custom_objects={'getPrecision': getPrecision,'getRecall':getRecall})  # 假设自定义的层的名字为AttLayer

model.summary()
print(model.summary())

a = np.load(r"C:\Users\20143\data.npy")
# print(model.predict(a[0]))

b = np.load(r"C:\Users\20143\data_label.npy")
a = np.load(r"C:\Users\20143\data.npy")
X, Y = shuffle(a, b)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split

# x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
x_test=X
y_test=Y
# print(y_test)
# print('---')
# print(x_test)

# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(8,6))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    # plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val, batch_size=16)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    print(conf_mat)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')


# =========================================================================================
# 最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
# labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
# 比如这里我的labels列表
labels = ['Estate','Tourist','Business','Medical','Railway','Road&Transport','Education','Catering','Shopping']
from keras.utils import to_categorical
y_test = to_categorical(y_test)
x_test= x_test.reshape((4313, 100, 1000, 1))
plot_confuse(model, x_test, y_test)