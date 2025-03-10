import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generar valores de entrada
x = np.linspace(-10, 10, 100)

# Calcular la salida de la función sigmoid
y = sigmoid(x)

# Plotear la función
plt.plot(x, y)
plt.title('Función de Activación Sigmoid')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)
plt.show()
