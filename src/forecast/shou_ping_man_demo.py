import pandas as pd

import numpy as np

# 创建一个空的DataFrame
df = pd.DataFrame(index=pd.date_range(start='2023-01-01 00:00:00', end='2023-01-02 00:00:00', freq='5T'))

# # 使用resample方法填充0或1，这里我们假设5分钟后的状态是随机的（0或1）
df['status'] = df.resample('5T').apply(lambda x: 1 if pd.np.random.rand() > 0.5 else 0)

# # 如果需要填充缺失的数据，可以使用ffill或bfill
# df = df.resample('5T').ffill()

print(df)