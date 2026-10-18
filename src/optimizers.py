import numpy as np

class SGD:
    """
        Gradiente descendente estocástico
        x_k = x_{k-1} - η ∇f(x_{k-1})
    """

    def __init__(self, eta=0.01):
        self.eta = eta # taxa de aprendizagem

    def step(self, w, grad):
        # atualiza os pesos na direção oposta ao gradiente
        return w - self.eta * grad 


class Adam:
    """
        Adam: Adaptive Moment Estimation

        m_k = β1 m_{k-1} + (1-β1) ∇f                (primeiro momento)
        v_k = β2 v_{k-1} + (1-β2) (∇f)²             (segundo momento)

        m̂_k = m_k / (1-β1^k)
        v̂_k = v_k / (1-β2^k)

        x_k = x_{k-1} - η m̂_k / (√v̂_k + ε)
    """

    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.eta, self.beta1, self.beta2, self.eps = eta, beta1, beta2, eps
        self.m = None # primeiro momento
        self.v = None # segundo momento
        self.k = 0 # numero de atualizações (iterações)

    def _adam_step(self, grad):
        if self.m is None:
            # inicializa os momentos com zeros
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.k += 1

        # atualiza os momentos de primeira e segunda ordem
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # corrige o viés inicial dos momentos
        m_hat = self.m / (1 - self.beta1**self.k)
        v_hat = self.v / (1 - self.beta2**self.k)

        # calcula o passo adaptativo para cada peso
        return self.eta * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, w, grad):
        # subtrai o passo adaptativo dos pesos
        return w - self._adam_step(grad)


class AdamW(Adam):
    """
        AdamW: Adaptive Moment Estimation with Weight Decay

        x_k = x_{k-1} - η m̂_k / (√v̂_k + ε) - η λ x_{k-1}
    """

    def __init__(self, eta=0.01, lam=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(eta, beta1, beta2, eps)
        self.lam = lam # coeficiente de decaimento dos pesos

    def step(self, w, grad):
        # combina a atualização Adam com o decaimento dos pesos
        return w - self._adam_step(grad) - self.eta * self.lam * w