import pandas as pd

# 创建一个示例DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}, index=['row1', 'row2', 'row3', 'row4'])
print('打印原始数据', df)
# 根据索引删除行
df_dropped = df.drop(labels=['row2', 'row4'], axis=0)
print(df_dropped)

# 或者，直接在原地修改（不推荐，因为会改变原始DataFrame）
df.drop(labels=['row2', 'row4'], axis=0, inplace=True)
print(df)