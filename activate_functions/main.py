import sys
import os
import matplotlib.pyplot as plt

def main():
    # Agregar el directorio que contiene srcs al path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'srcs'))

    # Importar y ejecutar las funciones de activación
    from activation_functions.relu import relu
    from activation_functions.sigmoid import sigmoid
    from activation_functions.tanh import tanh
    from activation_functions.leaky_relu import leaky_relu

    funciones = [relu, sigmoid, tanh, leaky_relu]
    for func in funciones:
        try:
            func()
            plt.show() 
        except Exception as e:
            print(f"wao") 

if __name__ == "__main__":
    main()
