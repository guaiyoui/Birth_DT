# 参考这篇文章中的实现方式
# https://zhuanlan.zhihu.com/p/271296719

import pandas as pd # 数据处理
import numpy as np # 使用数组
import matplotlib.pyplot as plt # 可视化
from matplotlib import rcParams # 图大小
from termcolor import colored as cl # 文本自定义

from sklearn.tree import DecisionTreeClassifier as dtc # 树算法
from sklearn.model_selection import train_test_split # 拆分数据
from sklearn.metrics import accuracy_score # 模型准确度
from sklearn.tree import plot_tree # 树图

rcParams['figure.figsize'] = (25, 20)

# 读取数据
data = pd.read_excel('./data/data.xlsx') 

# 数据映射, 如果需要改映射就在这边改
gender_map = {'男': 1, '女': 0}
w01_map = {'没有上过学': 0, '文盲/半文盲': 1, '小学': 2, '初中': 3, '高中/中专/技校/职高': 4, '大专': 5, '大学本科': 6, '硕士': 7, '博士': 8}
birth_map = {'是': 1, '否': 0 , '不知道': 0}
qg2032_map = {'是': 1, '否': 0, '不适用': 0, '不知道': 0}
qg3011_map = {'不知道': 40}
qg6_map = {'不知道': 40}
qg401_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2}
qg402_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2}
qg403_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2}
qg404_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2}
qg405_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2, '不适用(没有晋升机会)': 0}
qg406_map = {'非常不满意': 0, '不太满意': 1, '一般':2, '比较满意': 3, '非常满意': 4, '不知道': 2}
qg603_map = {'是': 1, '否': 0}

data['gender_update'].replace(gender_map, inplace=True)
data['w01'].replace(w01_map, inplace=True)
data['qka205'].replace(birth_map, inplace=True)
data['qg2032'].replace(qg2032_map, inplace=True)
data['qg401'].replace(qg401_map, inplace=True)
data['qg402'].replace(qg402_map, inplace=True)
data['qg403'].replace(qg403_map, inplace=True)
data['qg404'].replace(qg404_map, inplace=True)
data['qg405'].replace(qg405_map, inplace=True)
data['qg406'].replace(qg406_map, inplace=True)
data['qg603'].replace(qg603_map, inplace=True)
data['qg3011'].replace(qg3011_map, inplace=True)
data['qg6'].replace(qg6_map, inplace=True)
data.replace(np.nan, 0, inplace=True)

data = data.astype('int')

x_head = list(data.columns[0:3])+list(data.columns[4:])
y_head = ['qka205']


x_var = data[x_head].values # 自变量
y_var = data[y_head].values # 因变量

# for i in x_head:
#     print(i, data[i].unique())

# 可以调整这边的criterion参数来选择不同的算法，比如entropy或者gini
model = dtc(criterion = 'gini', max_depth = 5)

model.fit(x_var, y_var)

feature_names = x_head
target_names = data['qka205'].unique().tolist()
print("特征名称: ", feature_names)
target_names = [str(i) for i in target_names]
print(target_names)

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)

# 图中每一个数值的信息，可以参考这篇文章：
# https://blog.csdn.net/qq_29367075/article/details/113354858

plt.savefig("tree_visualization.pdf", dpi=600, bbox_inches='tight')

# plt.savefig('tree_visualization.pdf', ppi=300) 
# plt.savefig('tree_visualization.jpg') 

# 计算准确率
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.2, random_state = 0)
tree = dtc(criterion = 'gini', max_depth = 5)
tree.fit(x_train, y_train)
pred_model = tree.predict(x_test)
print('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)))

feature_importance = tree.tree_.compute_feature_importances(normalize=False)

print('feature importance', feature_importance)
