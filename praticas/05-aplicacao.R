#### Aplicação Aula 05 ####

# Essa aula tem como objetivo mostrar uma aplicação simples
# do uso de redes neurais e do torch em um problema de
# classificação e estimação de chance

# Contexto: estimação não paramétrica do ACE
# pela chamada "odds approach"

# Bibliotecas
library(torch)
library(luz)
library(dagitty)
library(ggdag)
library(ggplot2)
library(dplyr)
library(tidyr)

# Semente para reproducibilidade
set.seed(1)
torch_manual_seed(1)

#### STRUCTURAL CAUSAL MODEL (SCM) ####

grafo <- dagitty('dag { 
          X [exposure, pos ="1.000,2.000"]
          W1 [pos ="3.000,3.000"]
          W2 [pos ="3.000,2.000"]
          Y [outcome, pos ="5.000,2.000"]
          Z [pos ="3.000,1.000"]
          X -> { W1 W2 }
          { W1 W2 } -> Y
          Z -> { X Y }
        }')

ggdag(grafo) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  xlab("") + ylab("")

#### DADOS ####

generate_data <- function(n, sd=0.7) {
  Z <- rnorm(n, 0, sd)
  eps <- rbinom(n, 1, 0.8)
  X <- as.numeric(eps * (Z >= 0) + (1 - eps) * (Z < 0))
  U1 <- runif(n)
  W1 <- X*((U1<=1/8)*rnorm(n,0,sd) + (U1>1/8)*rnorm(n,4,sd)) + 
    (1-X)*rnorm(n,2,sd)
  U2 <- runif(n)
  W2 <- X*((U2<=1/8)*rnorm(n,0,sd) + (U2>1/8)*rnorm(n,4,sd)) + 
    (1-X)*rnorm(n,2,sd)
  Y <- rnorm(n, W1 + W2 + Z, sd)
  tibble(X, W1, W2, Y,Z) %>% as.data.frame()
}

n = 1000
dados <- generate_data(n)

ggplot(dados, aes(x = W1, y = W2, color = as.factor(X))) +
  geom_point(alpha = 0.5) +
  labs(color = "X") + 
  theme_minimal()

#### DATASET E DATALOADER ####

# Torch dataset
ds_torch <- dataset(
  
  name = "dados_torch",
  
  initialize = function(data) {
    # input data (scale)
    x <- data[, c('W1', 'W2')] %>% as.matrix() %>% scale()
    self$x <- torch_tensor(x)
    # target data
    y <- data$X %>% as.matrix()
    self$y <- torch_tensor(y)
  },
  
  .getitem = function(i) {
    list(self$x[i,], self$y[i])
  },
  
  .length = function() {
    dim(self$x)[1]
  }
)

n_train = as.integer(0.75*n)
train_indices <- sample(1:nrow(dados), n_train)

train_ds <- ds_torch(dados[train_indices, ])
valid_ds <- ds_torch(dados[setdiff(1:nrow(dados), train_indices), ])

train_dl <- train_ds %>% dataloader(batch_size = 32, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = 32, shuffle = FALSE)

#### MÓDULOS E MODELO ####

net <- nn_module(
  "Logit_Net",
  
  initialize = function() {
    self$fc1 <- nn_linear(2, 64)
    self$fc2 <- nn_linear(64, 32)
    self$fc3 <- nn_linear(32, 1)
    self$dropout <- nn_dropout(0.3)
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$fc3()
  }
)

model <- net %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_adam,
    metrics = luz_metric_binary_accuracy_with_logits()
  )

# Usando o "lr_finder"
rates_and_losses <- model %>%
  lr_finder(train_dl, start_lr = 1e-3, end_lr = 1)

rates_and_losses %>% plot()

fitted <- model %>%
  fit(
    train_dl,
    epochs = 20,
    valid_data = valid_dl,
    # callbacks!
    callbacks = list(
      luz_callback_early_stopping(patience = 3),
      luz_callback_lr_scheduler(
        lr_one_cycle,
        max_lr = 0.1,
        epochs = 20,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"
      )
    ),
    verbose = TRUE
  )

plot(fitted)

#### REGRESSÃO LOGÍSTICA ####

dados_train <- dados[train_indices, ]
dados_valid <- dados[setdiff(1:nrow(dados), train_indices), ]

log_reg <- glm(X ~ W1 + W2, data=dados_train, 
               family="binomial")
#summary(log_reg)

prob_train_pred <- predict(log_reg, type = "response")
prob_valid_pred <- predict(log_reg, 
                      newdata = dados_valid, 
                      type = "response")

y_train_pred <- ifelse(prob_train_pred > 0.5, 1, 0)
y_valid_pred <- ifelse(prob_valid_pred > 0.5, 1, 0)

acc_train <- mean(y_train_pred == dados_train$X)
acc_valid <- mean(y_valid_pred == dados_valid$X)

acc_train
acc_valid

#### PREDICOES DAS ODDS ####

# Modelo NN torch

dados_ds <- ds_torch(dados)

logit_pred <- predict(fitted, dados_ds$x)
cond_odds_nn <- torch_exp(logit_pred) %>% as.matrix()

# Modelo regressão logística

cond_odds_glm <- predict(log_reg, newdata = dados) %>% exp()

#### APLICAÇÃO EM INFERENCIA CAUSAL ####

# Algumas funções...

# Calcula a odds condicional
est_cond_odds <- function(X, do_x, odds) {
  ifelse(X == do_x, 1, ifelse(do_x == 1, odds, 1 / odds))
}

# Calcula (o inverso) da odds
est_odds <- function(X, do_x, p) {
  ifelse(X == do_x, 1, ifelse(do_x == 1, (1 - p) / p, p / (1 - p)))
}

# Algumas continhas para calcular o ACE

p = mean(dados$X==1)

dados$cond_odds_nn <- cond_odds_nn
dados$cond_odds_glm <- cond_odds_glm

dados <- dados %>%
  rowwise() %>%
  mutate(
    pratio_do1 = est_odds(X, 1, p),
    pratio_do0 = est_odds(X, 0, p),
    odds_do1_nn = est_cond_odds(X, 1, cond_odds_nn),
    odds_do0_nn = est_cond_odds(X, 0, cond_odds_nn),
    weight_do1_nn = odds_do1_nn * pratio_do1,
    weight_do0_nn = odds_do0_nn * pratio_do0,
    odds_do1_glm = est_cond_odds(X, 1, cond_odds_glm),
    odds_do0_glm = est_cond_odds(X, 0, cond_odds_glm),
    weight_do1_glm = odds_do1_glm * pratio_do1,
    weight_do0_glm = odds_do0_glm * pratio_do0
  ) %>%
  ungroup()

# Cálculo do efeito causal médio
mu1_est_nn <- sum(dados$Y * dados$weight_do1_nn, na.rm = TRUE) / sum(dados$weight_do1_nn, na.rm = TRUE)
mu0_est_nn <- sum(dados$Y * dados$weight_do0_nn, na.rm = TRUE) / sum(dados$weight_do0_nn, na.rm = TRUE)
ace_est_nn <- mu1_est_nn - mu0_est_nn
ace_est_nn

mu1_est_glm <- sum(dados$Y * dados$weight_do1_glm, na.rm = TRUE) / sum(dados$weight_do1_glm, na.rm = TRUE)
mu0_est_glm <- sum(dados$Y * dados$weight_do0_glm, na.rm = TRUE) / sum(dados$weight_do0_glm, na.rm = TRUE)
ace_est_glm <- mu1_est_glm - mu0_est_glm
ace_est_glm

#### LOOP PARA COMPARAÇÃO ####

n_simulations <- 10

results <- tibble(
  iteration = 1:n_simulations,
  ace_est_nn = numeric(n_simulations),
  ace_est_glm = numeric(n_simulations)
)

for (i in 1:n_simulations) {
  
  # Reproducibilidade
  set.seed(i+1)
  torch_manual_seed(i+1)
  
  # Para sabermos onde estamos no loop
  message("Running iteration ", i, " of ", n_simulations)
  
  # Gera dados
  dados <- generate_data(n)
  
  # Mesmo esqueminha pros dados
  train_indices <- sample(1:nrow(dados), n_train)
  dados_train <- dados[train_indices, ]
  dados_valid <- dados[setdiff(1:nrow(dados), train_indices), ]
  
  train_ds <- ds_torch(dados_train)
  valid_ds <- ds_torch(dados_valid)
  train_dl <- train_ds %>% dataloader(batch_size = 32, shuffle = TRUE)
  valid_dl <- valid_ds %>% dataloader(batch_size = 32, shuffle = FALSE)
  
  # Modelo NN
  fitted <- model %>%
    fit(
      train_dl,
      epochs = 10,
      valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 3),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.1,
          epochs = 10,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"
        )
      ),
      verbose = FALSE
    )
  
  # Odds preditas pela rede
  dados_ds <- ds_torch(dados)
  logit_pred <- predict(fitted, dados_ds$x)
  cond_odds_nn <- torch_exp(logit_pred) %>% as.matrix()
  
  # Regressão logística e odds pelo glm
  glm_final <- glm(X ~ W1 + W2, data = dados, family = "binomial")
  cond_odds_glm <- predict(glm_final) %>% exp()
  
  # Continhas para o ACE
  p <- mean(dados$X == 1)
  dados$cond_odds_nn <- cond_odds_nn
  dados$cond_odds_glm <- cond_odds_glm
  
  dados <- dados %>%
    rowwise() %>%
    mutate(
      pratio_do1 = est_odds(X, 1, p),
      pratio_do0 = est_odds(X, 0, p),
      odds_do1_nn = est_cond_odds(X, 1, cond_odds_nn),
      odds_do0_nn = est_cond_odds(X, 0, cond_odds_nn),
      weight_do1_nn = odds_do1_nn * pratio_do1,
      weight_do0_nn = odds_do0_nn * pratio_do0,
      odds_do1_glm = est_cond_odds(X, 1, cond_odds_glm),
      odds_do0_glm = est_cond_odds(X, 0, cond_odds_glm),
      weight_do1_glm = odds_do1_glm * pratio_do1,
      weight_do0_glm = odds_do0_glm * pratio_do0
    ) %>%
    ungroup()
  
  # NN ACE
  mu1_est_nn <- sum(dados$Y * dados$weight_do1_nn, na.rm = TRUE) / sum(dados$weight_do1_nn, na.rm = TRUE)
  mu0_est_nn <- sum(dados$Y * dados$weight_do0_nn, na.rm = TRUE) / sum(dados$weight_do0_nn, na.rm = TRUE)
  
  # GLM ACE
  mu1_est_glm <- sum(dados$Y * dados$weight_do1_glm, na.rm = TRUE) / sum(dados$weight_do1_glm, na.rm = TRUE)
  mu0_est_glm <- sum(dados$Y * dados$weight_do0_glm, na.rm = TRUE) / sum(dados$weight_do0_glm, na.rm = TRUE)
  
  # Salva o resultado
  results$ace_est_nn[i] <- mu1_est_nn - mu0_est_nn
  results$ace_est_glm[i] <- mu1_est_glm - mu0_est_glm
}

# Transforma os dados para plotar
results_long <- results %>%
  pivot_longer(
    cols = c(ace_est_nn, ace_est_glm),
    names_to = "method",
    values_to = "ace_estimate"
  )

# Valor verdadeiro do ACE
true_ace <- 3 

# Histograma
ggplot(results_long, aes(x = ace_estimate, fill = method)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 10) +
  geom_vline(aes(xintercept = true_ace), color = "black", linetype = "dashed", linewidth = 1) +
  scale_fill_manual(values = c("ace_est_nn" = "blue", "ace_est_glm" = "red")) +
  labs(
    title = "Histograms of ACE Estimates",
    x = "ACE Estimate",
    y = "Frequency",
    fill = "Method"
  ) +
  theme_minimal()




