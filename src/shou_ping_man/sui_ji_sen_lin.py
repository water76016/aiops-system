import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 假设df是你的数据集DataFrame，其中包含以下列：
# 'user_id', 'timestamp', 'feature1', 'feature2', ..., 'label'
# 其中'label'是一个二进制列，表示用户是否在某个时间段内点击（1为是，0为否）

# 这里我们假设df已经被正确加载和预处理
# 例如，你可能已经对时间戳进行了处理，提取了有用的时间特征（如小时、星期几等），并与其他特征合并

# 假设'feature1', 'feature2', ... 是你的特征列，'label'是目标列
X = df[['feature1', 'feature2', ...]]  # 替换为你的特征列名
y = df['label']  # 目标列

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器实例
# n_estimators是树的数量，你可以根据需要调整它
# 其他参数如max_depth、min_samples_split等也可以调整以优化模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 注意：这里的'feature1', 'feature2', ... 和 'label' 只是示例列名，你需要替换成你数据集中的实际列名