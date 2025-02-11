# embeddings 

library(torch)

local_torch_manual_seed(1)
embedding <- nn_embedding(
  num_embeddings = 10,
  embedding_dim = 6 # no gpt-4o eu acho que é 3072
)

input <- torch_tensor(
  rbind(
    c(1, 2, 3, 4, 10, 10),
    c(4, 3, 2, 9, 5, 10)
  ),
  dtype = torch_long()
)

embedding(input)

# obs: embedding na mão
nn_embedding_na_mao <- nn_module(
  initialize = function(num_embeddings, embedding_dim) {
    pesos <- torch_randn(num_embeddings, embedding_dim) |> 
      nn_parameter()
    self$weight <- pesos
  },
  forward = function(x) {
    x$matmul(self$weight)
  }
)

local_torch_manual_seed(1)
nn <- nn_embedding_na_mao(10, 6)

# não funciona, pois precisamos fazer o one hot
#nn(input[1,..])

nnf_one_hot(input)$to(dtype = torch_float()) |> 
  nn()

embedding(input)

# /embedding

# análise de texto

frases <- c(
  "eu gosto de gatos",
  "eu gosto de cachorros",
  "eu gosto de gatos e cachorros"
)

tokenize <- function(sentenca) {
  unlist(stringr::str_split(sentenca, " "))
}

# não é torch
tokens <- unique(tokenize(frases))
vocab <- purrr::set_names(seq_along(tokens), tokens)

# lapply()
vectorized_sentences <- purrr::map(
  frases,
  \(x) as.integer(factor(tokenize(x), levels = tokens))
)

# ta acabando

max_length <- purrr::map_int(vectorized_sentences, length) |> 
  max()

pad_sequences <- purrr::map(
  vectorized_sentences,
  \(x) nnf_pad(x, c(0, max_length - length(x)), value = 7) 
)

input_sequences <- torch::torch_stack(pad_sequences)

# rnn -----------------------

rnn <- nn_rnn(
  input_size = 4,
  hidden_size = 3,
  batch_first = TRUE,
  num_layers = 1
)

embedding <- nn_embedding(
  num_embeddings = 7,
  embedding_dim = 4,
  padding_idx = 7
)


emb <- embedding(input_sequences)

torch_randn(2, 4, 1)

rnn(emb)

# lstm, gru -----------------

lstm <- nn_lstm(
  input_size = 4,
  hidden_size = 7,
  batch_first = TRUE
)


output_lstm <- lstm(emb)
# entrada: 3,6,4
# saída list()


# aplicação: séries temporais

vic_elec <- tsibbledata::vic_elec

# plano:
# 1. criar um dataset
# 2. criar um modelo que use lstm
# 3. pipeline do luz

demand_dataset <- dataset(
  initialize = function(x, n_timesteps, sample_frac = 1) {
    self$n_timesteps <- n_timesteps
    self$x <- torch_tensor((x - train_mean) / train_sd)

    n <- length(self$x) - self$n_timesteps

    self$starts <- sort(sample.int(
      n,
      size = n * sample_frac
    ))

  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1
    list(
      # pega uma pequena sequência aleatória da base de dados
      x = self$x[start:end],
      # guarda o próximo elemento para previsão
      y = self$x[end + 1]
    )
  },
  .length = function() {
    length(self$starts)
  }
)

# pré processamento!
demand_hourly <- vic_elec |> 
  tsibble::index_by(Hour = lubridate::floor_date(Time, "hour")) |> 
  dplyr::summarise(Demand = sum(Demand))

demand_train <- demand_hourly |> 
  dplyr::filter(lubridate::year(Hour) == 2012) |> 
  dplyr::as_tibble() |> 
  dplyr::select(Demand) |> 
  as.matrix()

demand_valid <- demand_hourly |> 
  dplyr::filter(lubridate::year(Hour) == 2013) |> 
  dplyr::as_tibble() |> 
  dplyr::select(Demand) |> 
  as.matrix()

demand_test <- demand_hourly |> 
  dplyr::filter(lubridate::year(Hour) == 2014) |> 
  dplyr::as_tibble() |> 
  dplyr::select(Demand) |> 
  as.matrix()

train_mean <- mean(demand_train)
train_sd <- sd(demand_train)

n_timesteps <- 7 * 24

train_ds <- demand_dataset(demand_train, n_timesteps)
valid_ds <- demand_dataset(demand_valid, n_timesteps)
test_ds <- demand_dataset(demand_test, n_timesteps)

# Dataloaders

bsize <- 128

train_dl <- train_ds |> 
  dataloader(batch_size = bsize, shuffle = TRUE)
valid_dl <- valid_ds |> 
  dataloader(batch_size = bsize)
test_dl <- test_ds |> 
  dataloader(batch_size = length(test_ds))

x <- train_dl |> 
  dataloader_make_iter() |> 
  dataloader_next()

# lstm
# "o cachorro, pensando sobre a resposta da vida, do universo e tudo mais, ..."

model <- nn_module(
  initialize = function(input_size, hidden_size, dropout = 0.2, num_layers = 1, rec_dropout = 0) {

    self$lstm <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

    self$dropout <- nn_dropout(dropout)

    self$num_layers <- num_layers

    self$output <- nn_linear(hidden_size, 1)

  },
  forward = function(x) {
    # TODO: entender o sentido desse dim(x)[2]
    self$lstm(x)[[2]][[1]][1,,] |> 
    #self$lstm(x)[[1]][,dim(x)[2],] |> 
      self$dropout() |> 
      self$output()
  }
)

m <- model(input_size = 1, hidden_size = 32, num_layers = 2, rec_dropout = 0)
m(x$x)


# passo do luz
# ...próxima aula...


input_size <- 1
hidden_size <- 32
num_layers <- 2
rec_dropout <- 0.2

library(luz)

model <- model |>
  setup(optimizer = optim_adam, loss = nn_mse_loss()) |>
  set_hparams(
    input_size = input_size,
    hidden_size = hidden_size,
    num_layers = num_layers,
    rec_dropout = rec_dropout
  )

# Learning Rate finder: novidade do Luz
rates_and_losses <- model |>
  lr_finder(train_dl, start_lr = 1e-3, end_lr = 1, accelerator = luz::accelerator(cpu = TRUE))

rates_and_losses |> plot()

fitted <- model |>
  fit(
    train_dl,
    epochs = 5,
    valid_data = valid_dl,
    # exemplos de callbacks: outra novidade do luz
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
    verbose = TRUE,
    accelerator = luz::accelerator(cpu = TRUE)
  )


demand_viz <- demand_hourly |>
  dplyr::filter(lubridate::year(Hour) == 2014, lubridate::month(Hour) == 12)

demand_viz_matrix <- demand_viz |>
  tibble::as_tibble() |>
  dplyr::select(Demand) |>
  as.matrix()

viz_ds <- demand_dataset(demand_viz_matrix, n_timesteps)
viz_dl <- viz_ds |>
  dataloader(batch_size = length(viz_ds))

preds <- predict(fitted, viz_dl, accelerator = luz::accelerator(cpu = TRUE))
preds <- preds$to(device = "cpu") |>
  as.matrix()

preds <- c(rep(NA, n_timesteps), preds)

pred_ts <- demand_viz |>
  tibble::add_column(forecast = preds * train_sd + train_mean) |>
  tidyr::pivot_longer(-Hour) |>
  tsibble::update_tsibble(key = name)

pred_ts |>
  feasts::autoplot() +
  ggplot2::scale_colour_manual(values = c("#08c5d1", "#00353f")) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "None")
