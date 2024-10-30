import pandas as pd

# Asegúrate de que las fechas en ambos DataFrames coincidan
df_X = pd.read_csv('x.csv')
df_Y = pd.read_csv('y.csv')

# Convertir las fechas a formato datetime
df_X['period_end'] = pd.to_datetime(df_X['period_end'])
df_Y['date'] = pd.to_datetime(df_Y['date'])

# Alinear los DataFrames por fecha
df_X.set_index('period_end', inplace=True)
df_Y.set_index('date', inplace=True)

# Unir los DataFrames para asegurarse de que las fechas coincidan
df_combined = df_X.join(df_Y, how='inner')

# Separar nuevamente los DataFrames
Y = df_combined[['W']].values
X = df_combined[['cloud_opacity', 'ghi', 'dni', 'air_temp']].values

print(X.shape)
print(Y.shape)
