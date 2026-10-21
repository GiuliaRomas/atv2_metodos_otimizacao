import numpy as np

class ActivationFunction:
    def __init__(self, act_function):
        self.act_function = act_function

    def activate(self, v):
        return self.act_function(v)

class Model:
    def __init__(self, act_function, n_inputs, n_outputs, optimizer):
        self.act_function = act_function
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.optimizer = optimizer

        # pesos no formato (n_outputs, n_inputs) sorteados aleatoriamente em [0, 1)
        self.synaptic_weights = np.random.rand(n_outputs, n_inputs)

    def output(self, inputs):
        # v = W · x
        v = np.dot(self.synaptic_weights, np.array(inputs))

        # y = f(v)
        return list(map(self.act_function.activate, v))

    def learn(self, y_t, y_e, x):
        # e = y_t - y_e
        error = np.array(y_t) - np.array(y_e)

        # ∇W = -2 · e · xᵀ
        grad = -2 * np.outer(error, np.array(x))  

        # atualiza os pesos conforme o método do otimizador
        self.synaptic_weights = self.optimizer.step(self.synaptic_weights, grad)