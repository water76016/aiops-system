"""
数据清洗
1. 识别数据中的缺失值
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# 添加sklearn的Imputer填充器
# from sklearn.preprocessing import Imputer


# 皮马印第安人糖尿病数据集
PIMA_DATA = "../../../resources/data/pima.data"

# 识别数据中的缺失值
def find_na():
    # 自定义表头
    pima_columns_names = [
        'times_pregnant', 'plasma_glucose_concentration',
        'diastolic_blood_pressure', 'triceps_thickness',
        'serum_insulin', 'bmi', 'pedigree_function',
        'age', 'onset_diabetes'
    ]
    pima = pd.read_csv(PIMA_DATA, names=pima_columns_names)
    # 打印前5行数据
    print(pima.head())
    # 统计数量分布
    print(pima['onset_diabetes'].value_counts(normalize=True))

    col = 'plasma_glucose_concentration'
    plt.hist(pima[pima['onset_diabetes'] == 0][col], 10, alpha=0.5, label='non-diabetes')
    plt.hist(pima[pima['onset_diabetes'] == 1][col], 10, alpha=0.5, label='diabetes')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))
    plt.show()

    # # 显示相关性
    # print(sns.heatmap(pima.corr()))

    # 查看是否有空值
    print(pima.isnull().sum())
    # 查看数据的行数和列数
    print(pima.shape)
    # 查看数据的描述性统计
    print(pima.describe())

    # 删除存在缺失值的行
    pima_dropped = pima.dropna()

    # 需要试验的KNN模型参数
    knn_params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7]
    }

    # 进行knn拟合
    X_dropped = pima_dropped.drop('onset_diabetes', axis=1)
    y_dropped = pima_dropped['onset_diabetes']

    knn = KNeighborsClassifier()
    # 设置KNN模型，准备进行网格搜索
    grid = GridSearchCV(knn, knn_params)
    grid.fit(X_dropped, y_dropped)


    print(grid.best_score_, grid.best_params_)



if __name__ == '__main__':
    find_na()