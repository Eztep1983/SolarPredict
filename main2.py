import pandas as pd

''' ------- cargamos las variables de entrada -------- '''
df_Solcast = pd.read_csv('SolcastSeveral2Semestre2023.csv')

'''
Variables de interes: 
    * cloud_opacity
    * ghi
    * dni
    * air_temp
    * period_end
'''
df_Solcast = df_Solcast[['cloud_opacity', 'ghi', 'dni', 'air_temp', 'period_end']]

#df_SFV['date'] = pd.to_datetime(df_SFV['date'], format='%d.%m.%Y%H:%M')
#2023-07-01T00:05:00-05:00
df_Solcast['period_end'] = pd.to_datetime(df_Solcast['period_end'], format='ISO8601')

#2023-11-22T02:05:00-05:00

df_Solcast['period_end'] = df_Solcast['period_end'].dt.tz_convert(None).dt.floor('T')

df_Solcast['period_end'] = pd.to_datetime(df_Solcast['period_end'], format='%Y-%m-%d %H:%M')

print(df_Solcast.info())
print(df_Solcast.head(10))

df_Solcast.to_csv('x.csv', index=False)

