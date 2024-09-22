import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 示例：假设7天内的点击数据
click_times = pd.date_range(start='2024-09-12', periods=50, freq='12H')  # 用户的点击时间
df = pd.DataFrame({'timestamp': click_times, 'clicked': 1})  # 点击行为
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 生成7天内所有时间点（每10分钟一个点）
all_times = pd.date_range(start='2024-09-12', end='2024-09-18 23:59', freq='10T')
all_df = pd.DataFrame({'timestamp': all_times})

# 合成负样本
all_df['clicked'] = 0  # 初始化未点击
all_df = pd.concat([all_df, df])  # 合并点击和未点击数据

# 随机选择一部分负样本
negative_samples = all_df[all_df['clicked'] == 0]
n_neg_samples = min(len(df), 2 * len(df))  # 选择2倍正样本数量的负样本
sampled_negatives = negative_samples.sample(n=n_neg_samples, random_state=42)

# 合并正样本和随机选择的负样本
balanced_df = pd.concat([df, sampled_negatives])

# 特征工程
balanced_df['hour'] = balanced_df['timestamp'].dt.hour
balanced_df['minute'] = balanced_df['timestamp'].dt.minute
balanced_df['day_of_week'] = balanced_df['timestamp'].dt.dayofweek

# 准备训练数据
X = balanced_df[['hour', 'minute', 'day_of_week']]
y = balanced_df['clicked']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 生成未来一天的时间点
future_times = pd.date_range(start='2024-09-19 00:00', end='2024-09-19 23:50', freq='10T')
future_data = pd.DataFrame({'timestamp': future_times})
future_data['hour'] = future_data['timestamp'].dt.hour
future_data['minute'] = future_data['timestamp'].dt.minute
future_data['day_of_week'] = future_data['timestamp'].dt.dayofweek

# 预测未来时间点的点击情况
future_predictions = model.predict(future_data[['hour', 'minute', 'day_of_week']])
future_data['predicted_clicked'] = future_predictions

# 选择预测为点击的时间点
predicted_clicks = future_data[future_data['predicted_clicked'] == 1]
print(predicted_clicks)


print('-------------------------------------------------')
# 假设我们调整阈值为0.6
probabilities = model.predict_proba(future_data[['hour', 'minute', 'day_of_week']])[:, 1]
future_data['predicted_prob'] = probabilities

# 根据自定义阈值进行选择
threshold = 0.7
predicted_clicks = future_data[future_data['predicted_prob'] >= threshold]

print(predicted_clicks)
