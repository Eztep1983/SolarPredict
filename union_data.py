import pandas as pd

df_x = pd.read_csv('x.csv')
df_y = pd.read_csv('y.csv')

merge_time = df_x.merge(df_y, left_on='period_end', right_on='date', how='left')



print(merge_time.head())
merge_time.dropna(inplace=True)

print(merge_time.info())

merge_time.to_csv('data_final.csv', index=False)