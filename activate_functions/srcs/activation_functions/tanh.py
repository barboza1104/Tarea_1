import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

# Generar valores de entrada
x = np.linspace(-10, 10, 100)

# Calcular la salida de la función tanh
y = tanh(x)

# Plotear la función
plt.plot(x, y)
plt.title('Función de Activación Tangente Hiperbólica')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)
plt.show()
