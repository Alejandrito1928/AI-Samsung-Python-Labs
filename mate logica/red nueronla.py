import numpy as np

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"}
]

def init_layers(nn_architecture):
    param_values = {}
    
    for layer_idx, layer in enumerate(nn_architecture):
        layer_num = layer_idx + 1
        
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        param_values['W' + str(layer_num)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        
        param_values['B' + str(layer_num)] = np.random.randn(
            layer_output_size, 1) * 0.1
            
    return param_values

parametros = init_layers(nn_architecture)

print("✅ Inicialización de Parámetros Completa.")
print(f"Dimensión de W5 (Salida con Sigmoid): {parametros['W5'].shape}")

def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def single_layer_forward_propagation(A_prev, W_curr, B_curr, activation= "relu"):
    Z_curr = W_curr @ A_prev + B_curr
    if activation == "relu":
        A_curr = relu(Z_curr)
    elif activation == "sigmoid":
        A_curr = sigmoid(Z_curr)
    else:
        raise Exception("No se ha implementado la funcion de activacion")
    return A_curr, Z_curr

def full_forward_propagation(X, param_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for layer_idx, layer in enumerate(nn_architecture):
        layer_num = layer_idx + 1
        A_prev = A_curr
        
        activation_function_curr = layer["activation"]
        W_curr = param_values['W' + str(layer_num)]
        B_curr = param_values['B' + str(layer_num)]
        
        A_curr, Z_curr = single_layer_forward_propagation(
            A_prev, W_curr, B_curr, activation_function_curr
        )

        memory['A' + str(layer_idx)] = A_prev
        memory['Z' + str(layer_num)] = Z_curr
        
    return A_curr, memory

def convert_prob_into_class(probs):
    probs_ = probs.copy()
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

# Definición de la matriz de entrada X (2 características, 6 muestras)
data_list = [[1, 1], [-4, 1], [-1, 4], [-2, 4], [3, -4], [3, -1]]
X = np.array(data_list).T

param_values = init_layers(nn_architecture) # Inicializa (W y B)

# Ejecuta el Forward Pass completo
A_final, memory = full_forward_propagation(X, param_values, nn_architecture)

print("--- RESULTADO DEL FORWARD PASS ---")
print(f"Dimensiones de la Matriz de Entrada X: {X.shape}")
print(f"Dimensiones de la Matriz de Salida (Predicción): {A_final.shape}")
print(f"Valores de la Predicción (Probabilidades):")
print(A_final)

# Convierte las probabilidades a la decisión binaria (0 o 1)
Clasificacion_Final = convert_prob_into_class(A_final)
print("\nClasificación Final (0 o 1):")
print(Clasificacion_Final)


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)).sum()
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat == Y).all(axis=0).mean() 





y = np.array([[1, 1,1]])
Y_hat = np.array([[0.65, 0.1,0.7]])
cost = get_cost_value(Y_hat, y)
print("-"*30)
print(f"\nValor de la Función de Costo: {cost}")


def single_layer_backward_propagation(dA_curr, W_curr, B_curr, Z_curr, A_prev, activation= "relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        dZ_curr = relu_backward(dA_curr, Z_curr)
    elif activation == "sigmoid":
        dZ_curr = sigmoid_backward(dA_curr, Z_curr)
    else:
        raise Exception("No se ha implementado la funcion de activacion")
    
    dW_curr = dZ_curr @ A_prev.T / m
    dB_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = W_curr.T @ dZ_curr
    
    return dA_prev, dW_curr, dB_curr

def full_backward_propagation(Y_hat, Y, memory, param_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev
        A_prev = memory['A' + str(layer_idx_prev)]
        Z_curr = memory['Z' + str(layer_idx_curr)]
        
        W_curr = param_values["W" + str(layer_idx_curr)]
        B_curr = param_values["B" + str(layer_idx_curr)]

        dA_prev, dW_curr, dB_curr = single_layer_backward_propagation(
            dA_curr, W_curr, B_curr, Z_curr, A_prev, activ_function_curr
        )

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["dB" + str(layer_idx_curr)] = dB_curr

    return grads_values

def update(param_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        layer_idx += 1
        param_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        param_values["B" + str(layer_idx)] -= learning_rate * grads_values["dB" + str(layer_idx)]
    return param_values

def train(X, Y, nn_architecture, epochs, learning_rate):
    param_values = init_layers(nn_architecture)
    
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, memory = full_forward_propagation(X, param_values, nn_architecture)
        
        cost = get_cost_value(Y_hat, Y)
        accuracy = get_accuracy_value(Y_hat, Y)
        
        cost_history.append(cost)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, memory, param_values, nn_architecture)
        
        param_values = update(param_values, grads_values, nn_architecture, learning_rate)
        
    return param_values, cost_history, accuracy_history

x = np.array([[1, -4, -1, -2, 3, 3], [1, 1, 4, 4, -4, 9]])
y = np.array([[1, 1, 1, 0, 0, 0]])
print(y)
print(x)
        
param_values, cost_history, accuracy_history = train(x, y, nn_architecture, 5000, 0.1)
print("✅ Entrenamiento Completo.")
print(f"Valor Final de la Función de Costo: {cost_history[-1]}")
cost_history[-1]
Y_hat

import matplotlib.pyplot as plt
plt.plot(cost_history)
plt.xlabel("Épocas")
plt.ylabel("Función de Costo")
plt.title("Evolución de la Función de Costo durante el Entrenamiento")
plt.show()