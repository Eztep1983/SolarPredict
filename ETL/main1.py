import pandas as pd

''' ------- cargamos los datos de generacion de potencia activa -------- '''
df_SFV = pd.read_csv('DatosSFV.csv')

print(df_SFV.info())

df_SFV = df_SFV[['[dd.MM.yyyyHH:mm]', '[VA]']]

print(df_SFV.info())

df_SFV = df_SFV.rename(columns={'[dd.MM.yyyyHH:mm]': 'date', '[VA]': 'W'})

print(df_SFV.info())
print(df_SFV.head(20))

df_SFV['date'] = pd.to_datetime(df_SFV['date'], format='%d.%m.%Y%H:%M')
df_SFV.dropna(inplace=True)

print(df_SFV.info())
df_SFV.to_csv('y.csv', index=False)

''' ------- cargamos las variables de entrada para el modelo -------- '''
