#### Exercício Aula 01 ####

# Mude a seed se quiser!
library(torch)

torch_manual_seed(1)

# Considere o tensor

(tx <- torch_randn(c(3,4,1)))

#### Item a) ####

# Qual a dimensão do vetor criado?

# Escreva um vetor ty que é funcão de tx

#### Item b) ####

# Seja

(ty <- 5*tx$view(c(3,1,4)) + torch_randn(3,1,4))

# Redimensione tx e ty de pelo menos três maneiras 
# (use view, reshape, squeeze e unsqueeze)

# Qual será o resultado de squeeze e unsqueeze aplicado nos vetores originais?

#### Item c) ####

# Acesse segunda e terceira linha da segunda página de tx

# Acesse primeira e última coluna da primeira página de ty

#### Item d) ####

# Qual será a dimensão de tx + ty?

# Qual será a dimensão de tx[1,..] + ty[1,..]?

# Seja

(tx_un <- tx$squeeze())
(ty_un <- ty$squeeze()$t())

# O que acontece se tentarmos fazer tx_un + ty_un? Explique porquê.

#### Item e) ####

# Qual será a dimensão do produto matricial entre tx_un e ty_un?

#### Item f) ####

# Seja

tx_matrix <- tx$flatten() |> as.matrix()
(tx_flat <- model.matrix(~tx_matrix) |> torch_tensor())
(ty_flat <- ty$flatten())

# Obtenha o resultado da regressao linear com linalg_lstsq

# Compare com o resultado de lm(as.matrix(ty_flat) ~ as.matrix(tx_flat))

# ========================================================================

# Exercício 1: Explorando Dimensões de Tensores
# Considere um tensor criado com torch_tensor(1:12).
# Qual seria a dimensão desse tensor?

# Exercício 2: Operações Básicas
# Dados dois tensores A e B, ambos com valores torch_tensor(1:4),
# qual seria o resultado de A * B?

# Exercício 3: Desafio de Redimensionamento
# Se você tiver um tensor de 12 elementos,
# quais dimensões você poderia usar para redimensioná-lo em
# uma matriz 2D sem causar erro? Mostre como você faria isso.

# Exercício 4: Slicing Avançado
# Considere um tensor 3D com dimensões (4, 4, 4).
# Como você acessaria apenas a segunda e terceira coluna da segunda "página"?
# Experimente várias formas de fazer isso.

# Exercício 5: Broadcasting
# Dados tensor_a5 de dimensão (3, 2) e tensor_b5 de dimensão (2,),
# qual seria o resultado de tensor_a5 + tensor_b5? E se tentássemos somar
# tensor_a5 com um tensor de dimensão (3,), isso causaria um erro?

# Exercício 6: Interpretando Resultados de Operações Matriciais
# Se você multiplicar um tensor A (dimensão 2x3) por um tensor B (dimensão 3x2),
# qual será a dimensão do resultado? Qual seria a dimensão se você transpuser
# o resultado?

# Exercício 7: Manipulação de Datasets com Tensores
# Após converter o dataset 'mtcars' para um tensor, que passos você
# tomaria para calcular a média da primeira coluna (mpg)? Escreva o código.

# Exercício 8: Predição de Decomposição de Matrizes
# Ao realizar uma decomposição QR em um tensor quadrado 4x4,
# quantas matrizes e de quais dimensões você espera receber?

# Exercício 9: Desafio Prático - Regressão Linear
# Dado o dataset 'cars', com duas colunas (x e y), como você utilizaria
# tensores para calcular os coeficientes de uma regressão linear de y em x?
# Dica: Considere a fórmula da regressão linear (β = (X'X)^-1 X'y).
# Calcule todas as formas que você conhece para fazer isso e
# compare os resultados.

