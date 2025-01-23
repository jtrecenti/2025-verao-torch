#### Exercício Aula 03 ####

# Mude a seed se quiser!
library(torch)
library(torchvision)
library(luz)
torch_manual_seed(1)

# Considere o dataset "iris"

View(iris)
?iris

# Considere como variável resposta "Species"

#### Item a) ####

# Crie o dataset e o dataloader
# (Cuidado para aplicar scale apenas nas variáveis contínuas)

ds_iris <- dataset(
  # desenvolva a função
)

dl_iris <- dataloader(
  # desenvolva a função
)

#### Item b) ####

# Crie um MLP 
# (Usem ao menos uma camada oculta com ReLU, sejam criativos!)

net <- nn_module(
  # desenvolva a rede
  # obs: termine com uma camada linear
)

result <- net |>
  setup(
    loss = nn_bce_with_logits_loss(),
    # pode mudar (mas use uma para classificação)
    optimizer = optim_adam # pode mudar
    # outros parametros: metrics, etc
  ) |>
  fit(
    # desenvolver parâmetros da função
  )

#### Item c) ####

# Obtenha as predições do modelo e a taxa de acerto

## =======================================================================

# Exercício 1: Criação de Dataset e DataLoader
# Crie um dataset utilizando o conjunto de dados 'mtcars' e
# defina um DataLoader com um tamanho de batch de 4.

# Bibliotecas necessárias
library(torch)
library(torchvision)

# Conjunto de dados 'cars_scale'
mtcars_scale <- scale(mtcars)

# Criação do Dataset e DataLoader
# Substitua os "##" com o código correto.
ds_mtcars <- ##(mtcars_scale)
dl_mtcars <- ##(ds_mtcars, batch_size = 10)

# Verifique o tamanho do seu dataset e dataloader
print(length(ds_cars))
print(length(dl_cars))


# Exercício 2: Construção de uma MLP
# Construa uma MLP com uma camada oculta de 8 neurônios e uma função de
# ativação ReLU. Ajuste usando o Luz. Utilize mtcars ou outra base de sua
# preferência.

# Exercício 5: Visualização de Predições
# Visualize as predições do seu modelo em um gráfico,
# comparando com os dados reais.

# Exercício 6: Construção de uma CNN
# Crie uma CNN  para a base torchvision::kmnist_dataset()

ds_kmnist <- torchvision::kmnist_dataset(
  ## estudar os parâmetros
)

