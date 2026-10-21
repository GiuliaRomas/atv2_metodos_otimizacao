# Métodos de Otimização — Atividade 2

Implementações em Python de métodos de otimização baseados em gradiente.

## Conteúdo

| Notebook | Descrição |
|---|---|
| `notebooks/PC3_b_gradiente_estocastico.ipynb` | PC.3 (b): gradiente estocástico na função de Himmelblau, com variação de parâmetros e trajetórias sobre as curvas de nível |
| `notebooks/PC5_c_AdamW.ipynb` | PC.5: Aproximação de $\cos(x)$ em $[0, 2\pi]$ por regressão linear, usando o otimizador AdamW (custo: MSE) |

## Principais resultados

- **Regressão do cosseno:** Adam (MSE de teste ≈ 7,6×10⁻⁷) e AdamW (≈ 7,8×10⁻⁷) convergiram mais rápido e com erro menor que o SGD (≈ 1,8×10⁻⁴).
- **Himmelblau (SGD):** η entre 0,005 e 0,01 converge bem, η ≥ 0,02 diverge, o ponto inicial define qual dos quatro mínimos é encontrado, e o decaimento da taxa reduz a oscilação quando há ruído no gradiente.

## Autores

- Giulia Romas
- Gabriel David