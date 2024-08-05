import pandas as pd
from prophet import Prophet

# 示例数据
data = {
    'user_id': ['user1'] * 10 + ['user2'] * 5,
    'timestamp': pd.to_datetime(['2023-01-01 08:00', '2023-01-01 09:00', '2023-01-01 10:00', '2023-01-02 08:30',
                                 '2023-01-02 09:30', '2023-01-02 10:30', '2023-01-03 08:15',
                                 '2023-01-03 09:15', '2023-01-03 10:15', '2023-01-03 11:15',
                                 '2023-01-01 12:00', '2023-01-02 12:30', '2023-01-03 13:00',
                                 '2023-01-04 13:30', '2023-01-05 14:00'])
}
df = pd.DataFrame(data)

# 将时间戳转换为日期（对于Prophet）
df['ds'] = df['timestamp'].dt.date
df['y'] = 1  # 对于点击事件，我们可以简单地将y设为1（如果有多次点击，可能需要更复杂的数据处理）

# 创建Prophet模型实例
m = Prophet()

# 由于Prophet默认以天为单位，我们不需要额外设置频率
# 如果数据中包含多个用户，可以考虑为每个用户单独建模

# 拟合模型
m.fit(df)

# 创建未来数据框架
future = m.make_future_dataframe(periods=3)  # 预测未来1天

# 进行预测
forecast = m.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())