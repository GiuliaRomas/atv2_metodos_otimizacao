from fractions import Fraction
import numpy as np

def pi_fraction_formatter(x, pos):
    # aproxima x/π por uma fração com denominador ≤ 8
    frac = Fraction(x / np.pi).limit_denominator(8) 
    
    num = frac.numerator
    den = frac.denominator

    # formata os casos especiais e os múltiplos de π
    if num == 0:
        return "0"
    elif den == 1:
        if num == 1:
            return r"$\pi$"
        elif num == -1:
            return r"$-\pi$"
        else:
            return rf"${num}\pi$"
    else:
        # formata as frações de π em LaTeX
        if num == 1:
            return rf"$\frac{{\pi}}{{{den}}}$"
        elif num == -1:
            return rf"$-\frac{{\pi}}{{{den}}}$"
        else:
            return rf"$\frac{{{num}\pi}}{{{den}}}$"

def make_features(x, degree):
    # normaliza x para t ∈ [-1, 1], considerando x ∈ [0, 2π]
    t = (x - np.pi) / np.pi 

    # gera as features polinomiais [1, t², ..., t^degree]
    # organiza as features em colunas, gerando uma matriz de dimensão (len(x), degree + 1)
    return np.column_stack([t**p for p in range(0, degree + 1, 2)]) 

def my_activation_function(v):
    # função identidade: f(v) = v
    return v  