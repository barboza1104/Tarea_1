import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

# Generar valores de entrada
x = np.linspace(-10, 10, 100)

# Calcular la salida de la función ReLU
y = relu(x)

# Plotear la función
plt.plot(x, y)
plt.title('Función de Activación ReLU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)
plt.show()
