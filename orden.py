import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
x_data = pd.read_csv("x.csv")
y_data = pd.read_csv("y.csv")

# Asegúrate de que ambas columnas de tiempo tengan el mismo formato
x_data['period_end'] = pd.to_datetime(x_data['period_end'])
y_data['date'] = pd.to_datetime(y_data['date'])

# Extrae las características de x y la variable objetivo de y
X = x_data[['cloud_opacity', 'ghi', 'dni', 'air_temp']].values
y = y_data['W'].values

# Escalar los datos para mejorar el rendimiento del modelo
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
