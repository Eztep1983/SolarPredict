import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

df_X = pd.read_csv('x.csv')
df_Y = pd.read_csv('y.csv')

#CONVERTIR LAS FECHAS EN FORMATO DATETIME
df_X['period_end'] = pd.to_datetime(df_X['period_end'])
df_Y['date'] = pd.to_datetime(df_Y['date'])

#ALINEANDO LOS DATAFRAMES POR FECHA Y HACERLOS JOIN PARAUNIRLOS
df_X.set_index('period_end', inplace=True)
df_Y.set_index('date', inplace=True)
df_combined = df_X.join(df_Y, how='inner')


X = df_combined[['cloud_opacity', 'ghi', 'dni', 'air_temp']].values#SEPARANDO LAS CARACTERISTICAS
Y = df_combined[['W']].values #ESTA ES LA VARIABLE OBJETIVO XD

#ESCALAR LOS DATOS
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

#DIVIDIR EN CONJUNTO DE ENTRENAMIENTO Y PRUEBA 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#MODELO
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  #SALIDA PARA LA PRODUCCION DE ENERGIA SOLAR
])

#COMPILAR EL MODELO
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#ENTRENAMIENTO DEL MODELO
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=1)

#EVALUACION DEL MODELO
loss, mae = model.evaluate(X_test, y_test)
print("Mean Absolute Error en conjunto de prueba:", mae)

#REALIZAR PREDICCIONES
y_pred = model.predict(X_test)

#INVERTIR ESCALA PARA LOS VALORES REALES
y_pred_rescaled = scaler_Y.inverse_transform(y_pred)
y_test_rescaled = scaler_Y.inverse_transform(y_test)

#MOSTRAR LOS RESULTADOS
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Valores Reales', color='blue', alpha=0.6)
plt.plot(y_pred_rescaled, label='Predicciones', color='red', alpha=0.6)
plt.title("Comparación entre Producción Real y Predicha de Energía Solar")
plt.xlabel("Ejemplos")
plt.ylabel("Producción de Energía (W)")
plt.legend()
plt.show()
