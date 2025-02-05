library(torch)

# novas!
library(luz)
library(torchvision)


cars_scale <- cars |> 
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

cars_matrix <- model.matrix(~speed, data = cars_scale)
xy <- cbind(cars_matrix, cars_scale$dist)

cars_tensor <- torch_tensor(xy)

xx <- cars_tensor[, 2]$unsqueeze(2)
yy <- cars_tensor[, 3]$unsqueeze(2)

mlp <- nn_sequential(
  nn_linear(1, 8),
  nn_relu(),
  nn_linear(8, 1)
)

optimizer <- optim_sgd(mlp$parameters, lr = 0.01)

loss <- nn_mse_loss()

for (t in 1:1000) {
  y_pred <- mlp(xx)
  l <- loss(y_pred, yy)
  optimizer$zero_grad()
  l$backward()
  optimizer$step()
}

library(ggplot2)

ggplot(cars_scale, aes(speed, dist)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)+
  geom_line(
    data = data.frame(
      speed = cars_scale$speed,
      dist = as.numeric(y_pred)
    ),
    colour = "darkgreen"
  )
  
# pacote luz: agora a coisa fica bem mais divertida e fácil

ds_cars_module <- dataset(
  name = "cars_dataset",
  initialize = function(da) {
    cars_scale <- da |> 
      dplyr::mutate(
        speed = scale(speed),
        dist = scale(dist)
      )
    cars_matrix <- model.matrix(~speed, data = cars_scale)

    xy <- cbind(cars_matrix, cars_scale$dist)

    self$x <- cars_tensor[,2]$unsqueeze(2)
    self$y <- cars_tensor[,3]$unsqueeze(2)
  },
  .getitem = function(idx) {
    list(self$x[idx,], self$y[idx,])
  },
  .length = function() {
    dim(self$x)[1]
  }
)

ds_cars <- ds_cars_module(cars)

ds_cars$.length()
ds_cars$.getitem(1)


## exemplo de implementação
# tensor_dataset(xx, yy)

dl_cars <- dataloader(ds_cars, batch_size = 10, shuffle = TRUE)

# não precisaremos rodar na prática
dl_cars |> 
  dataloader_make_iter() |> 
  dataloader_next()

# pegar artigo sobre sgd e regularização

# chegou a hora da felicidade

net <- nn_module(
  initialize = function(d_hidden) {
    self$net <- nn_sequential(
      nn_linear(1, d_hidden),
      nn_relu(),
      nn_linear(d_hidden, 1)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

resultado <- net |> 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |> 
  set_hparams(
    d_hidden = 8
  ) |> 
  set_opt_hparams(
    lr = 0.01
  ) |> 
  fit(
    dl_cars,
    epochs = 100,
    callbacks = list(
      luz_callback_early_stopping(
        "train_loss",
        min_delta = 0.001,
        patience = 3
      )
    )
)

y_pred <- predict(resultado, xx)

ggplot(cars_scale, aes(speed, dist)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)+
  geom_line(
    data = data.frame(
      speed = cars_scale$speed,
      dist = as.numeric(y_pred)
    ),
    colour = "darkgreen"
  )

# CNN -------------------------------------------

# torchvision: toolkit para trabalhar com imagens

# 1. datasets e dataloaders

train_ds <- mnist_dataset(
  "dados/",
  download = TRUE,
  train = TRUE,
  transform = transform_to_tensor
)

valid_ds <- mnist_dataset(
  "dados/",
  download = TRUE,
  train = FALSE,
  transform = transform_to_tensor
)

length(train_ds)
length(valid_ds)

train_dl <- dataloader(train_ds, batch_size = 32)
valid_dl <- dataloader(valid_ds, batch_size = 32)


xy <- train_dl$.iter()$.next()
x <- xy$x
y <- xy$y

plot(as.raster(as.matrix(x[2,1,,])))

# 2. arquitetura da rede

net <- nn_module(
  initialize = function() {

    # 28x28 -padding> 26x26 -padding> 24x24 -> (max pooling 2x2) -> 12x12
    # x64 canais de saída
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)

  },
  forward = function(x) {
    x |> 
      self$conv1() |> 
      nnf_relu() |> 
      self$conv2() |> 
      nnf_relu() |> 
      nnf_max_pool2d(2) |> 
      torch_flatten(start_dim = 2) |> 
      self$fc1() |> 
      nnf_relu() |> 
      self$fc2()
  }
)

# 3. setup e ajuste do modelo

fitted <- net |> 
  luz::setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz::luz_metric_accuracy()
    )
  ) |> 
  luz::fit(
    train_dl,
    valid_data = valid_dl,
    epochs = 1,
    # meu computador está bugado
    accelerator = luz::accelerator(cpu = TRUE)
  )

preds <- predict(
  fitted,
  valid_dl,
  accelerator = luz::accelerator(cpu = TRUE)
)

id <- 47
predict(
  fitted,
  valid_ds[id]$x$unsqueeze(1),
  accelerator = luz::accelerator(cpu = TRUE)
) |> 
  as.numeric() |> 
  which.max() |> 
  magrittr::subtract(1)

plot(as.raster(as.matrix(valid_ds[id]$x[1,,])))

# fim