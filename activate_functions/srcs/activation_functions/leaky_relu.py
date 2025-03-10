import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generar valores de entrada
x = np.linspace(-10, 10, 100)

# Calcular la salida de la función Leaky ReLU
y = leaky_relu(x)

# Plotear la función
plt.plot(x, y)
plt.title('Función de Activación Leaky ReLU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)
plt.show()
