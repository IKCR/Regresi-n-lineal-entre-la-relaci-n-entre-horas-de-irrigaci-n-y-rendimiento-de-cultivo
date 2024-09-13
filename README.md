# Regresion-lineal-entre-la-relacion-entre-horas-de-irrigacion-y-rendimiento-de-cultivo
# Se importan las librerias necesarias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Se definen los de datos de Horas de irrigación y Rendimiento de cosecha en toneladas
Irrigation_Hours = np.array([5, 12, 8, 15, 10, 20, 7, 18, 9, 22, 6, 13, 16, 14, 11, 19, 21, 8, 17, 12,
                             25, 9, 24, 13, 18, 6, 14, 9, 19, 4, 17, 11, 22, 3, 26, 7, 13, 10, 15, 8,
                             21, 5, 18, 12, 20, 16, 25, 9, 23, 14, 11, 30, 6, 27, 7, 24, 15, 28, 8, 29,
                             4, 31, 10, 32, 12, 33, 5, 34, 13, 35, 17, 36, 18, 37, 16, 38, 20, 39, 21, 40,
                             22, 42, 23, 43, 25, 44, 26, 46, 27, 47, 28, 48, 30, 49, 31, 50, 33, 51, 34, 53])

Crop_Yield = np.array([10, 18, 14, 22, 16, 25, 12, 21, 15, 27, 11, 19, 20, 17, 13, 23, 26, 14, 22, 18,
                       30, 15, 29, 20, 24, 11, 20, 13, 23, 8, 22, 16, 27, 6, 32, 10, 19, 15, 21, 12,
                       26, 9, 24, 18, 25, 23, 31, 13, 28, 19, 17, 37, 10, 34, 11, 30, 21, 35, 12, 36,
                       7, 39, 16, 41, 17, 42, 9, 43, 19, 44, 24, 45, 25, 46, 22, 47, 28, 48, 29, 50,
                       30, 53, 31, 54, 33, 55, 34, 57, 35, 58, 36, 60, 38, 61, 39, 62, 41, 63, 42, 65])

# Se definen las variables independiente y dependiente (x,y) respectivamente
x = Irrigation_Hours.reshape(-1,1)  # Variable independiente -> Horas de irrigación
# Reshape de las horas de irrigación para ajustar a la forma esperada por el modelo
# La funcion reshape convierte un arreglo de una dimension en una matriz columna
y = Crop_Yield  # Variable dependiente -> Rendimiento de la cosecha

# Se Divididen los datos en conjuntos de entrenamiento y prueba
# test_size=0.2 significa que el 20% de los datos se usará para prueba, y el 80% para entrenamiento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Se crea una instancia del modelo de Regresión Lineal
model = LinearRegression()

# Se ajusta el modelo a los datos de entrenamiento
model.fit(x_train, y_train)

# Se predice los rendimientos de la cosecha usando los datos de prueba
y_pred = model.predict(x_test)

# Se calcula el coeficiente de determinación (R²), que mide la precisión del modelo
r2 = model.score(x_test, y_test)
print("R2 Score:", r2)

# Se obtiene el coeficiente (pendiente de la línea de regresión)
coefficient = model.coef_[0]
print("Coefficient:", coefficient)

# Se obtiene la intersección (intercepto de la línea de regresión con el eje y)
intercept = model.intercept_
print("Intercept:", intercept)

# Se crea un gráfico de dispersión de las horas de irrigación vs. el rendimiento de la cosecha
plt.scatter(x_test, y_test, color='blue')

# Se grafica la línea de regresión ajustada sobre los datos de prueba
plt.plot(x_test, y_pred, color='red', linestyle = '-', linewidth = 2, label = 'Predicted Crop Yield')

# Etiquetas y título del gráfico
plt.xlabel('Irrigation Hours')  # Etiqueta del eje x
plt.ylabel('Crop Yield (tons)')  # Etiqueta del eje y
plt.title('Linear Regression')  # Título del gráfico

# Se muestra la leyenda y se grafica
plt.legend()
plt.show()
