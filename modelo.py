from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from orden import *

# Definir el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(Y.shape[1])
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X, Y, epochs=100, validation_split=0.2, verbose=1)

# Evaluar el modelo
loss = model.evaluate(X, Y)
print(f'Pérdida del modelo: {loss}')

# Hacer predicciones
predicciones = model.predict(X)
print(predicciones)
