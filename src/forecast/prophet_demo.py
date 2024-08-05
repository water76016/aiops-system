# import pandas as pd
# from fbprophet import Prophet
#
# # 创建数据集（示例）
# df = pd.DataFrame({
#     'ds': pd.date_range(start='2023-01-01', periods=365, freq='D'),
#     'y': [np.random.normal(loc=100, scale=10) for _ in range(365)]  # 假设的每日访问量数据
# })
#
# # 初始化Prophet模型
# m = Prophet()
#
# # 拟合模型
# m.fit(df)
#
# # 创建未来日期的数据框（用于预测）
# future = m.make_future_dataframe(periods=30)  # 预测未来30天
#
# # 进行预测
# forecast = m.predict(future)
#
# # 查看预测结果
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
#
# # 可视化预测结果
# fig = m.plot(forecast)