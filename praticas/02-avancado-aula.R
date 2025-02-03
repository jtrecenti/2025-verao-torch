library(torch)

cars_scaled <- cars |> 
  dplyr::mutate(
    speed = scale(speed),
    dist = scale(dist)
  )

X <- model.matrix(~speed, data = cars_scaled)
X <- torch_tensor(X)

y <- cars_scaled[,2] |> 
  torch_tensor()

XtX <- X$t()$matmul(X)
XtX_inv <- XtX$inverse()
XtX_inv$matmul(X$t())$matmul(y)

lm(dist ~ speed, data = cars_scaled)

# descomposição de cholesky
# XtX = LL^t 

# a,b,c
# a = 1
# a + b = 3
# a + b + c = 5

#XtX b = Xt y
#L^t b = z
#LL^t b = Xt y
#L z = Xt y

L <- linalg_cholesky(XtX)

#Xty <- X$t()$matmul(y) # mm é igual matmul
Xty <- X$t()$mm(y)
torch_triangular_solve(Xty, L, upper = FALSE) |> 
  purrr::pluck(1) |> 
  torch_triangular_solve(L$t()) |> 
  purrr::pluck(1)

# decomposição QR

list_qr <- linalg_qr(X)

#a, b <- [1,2]

library(zeallot)

c(Q, R) %<-% linalg_qr(X)
torch_triangular_solve(Q$t()$mm(y), R)

# autograd e cálculo de gradientes -------------------------

t1 <- torch_tensor(10, requires_grad = TRUE)
t2 <- t1 + 2
t3 <- t2$square()
t4 <- t3 * 3

# dt4/dt1
t4$backward()

t1$grad

# t4 = 3 * (t1 + 2) ^ 2
# dt4 / dt1 = 1 * 2 * 3 * (t1 + 2) = 6 * (t1 + 2)
t1
6 * (t1 + 2)

# usar retain_grad()

t1 <- torch_tensor(10, requires_grad = TRUE)
t2 <- t1 + 2
t2$retain_grad()
t3 <- t2$square()
t3$retain_grad()
t4 <- t3 * 3
t4$retain_grad()

t4$backward()

t3$grad
t2$grad
t1$grad

# vamos então ajustar o beta da regressão com essa técnica!

# N
# for (i in 1:N) {
#   perda <- mse(modelo, perda)
#   perda$backward()
#   beta <- beta - alpha * beta$grad
# }

num_iterations <- 10000
lr <- 0.00001

beta <- torch_tensor(c(0, 1), requires_grad = TRUE)

# residuos <- X$mm(beta) - y
# loss <- residuos$square()$mean()

# reescrevendo

forward_pass <- function(X, beta) {
  X$matmul(beta)
}

mse <- function(beta) {
  residuos <- forward_pass(X, beta) - y
  loss <- residuos$square()$mean()
  loss
}

for (i in 1:num_iterations) {

  perda <- mse(beta)
  perda$backward()

  with_no_grad({
    beta$sub_(lr * beta$grad)
    beta$grad$zero_()
  })

}

beta

beta <- torch_tensor(c(0, 1), requires_grad = TRUE)
otimizador_2a_derivada <- optim_lbfgs(beta, lr = 0.0001)

atualizar_beta <- function() {
  otimizador_2a_derivada$zero_grad()
  perda <- mse(beta)
  perda$backward()
  perda
}

for(i in 1:10) {
  otimizador_2a_derivada$step(atualizar_beta)
}

beta

# primeira rede neural - MLP - Multi Layer Perceptron

#w1 <- torch_randn(1, 8, requires_grad = TRUE)
#b1 <- torch_zeros(1, 8, requires_grad = TRUE)

X1 <- X[,2]$unsqueeze(2)
lr <- 0.0001

# passo 1: parametros
d_in <- 1
d_hidden <- 8
d_out <- 1

w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)

w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

for (i in 1:1000) {
  # passo 2: forward pass
  # passo forward
  resultado_inicial <- X1$matmul(w1)$add(b1)
  # agora precisamos (!!!!) adicionar uma função NÃO linear
  nao_linear <- resultado_inicial$relu()
  # finalmente, chegamos na dimensão da variável resposta
  y_pred <- nao_linear$matmul(w2)$add(b2)

  loss <- (y_pred - y)$pow(2)$mean()

  # gradiente
  loss$backward()

  with_no_grad({

    w1$sub_(lr * w1$grad)
    w2$sub_(lr * w2$grad)
    b1$sub_(lr * b1$grad)
    b2$sub_(lr * b2$grad)

    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()

  })
}

loss

library(ggplot2)

ggplot(cars_scaled) +
  aes(speed, dist) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_line(
    data = data.frame(
      speed = cars_scaled$speed,
      dist = as.numeric(y_pred)
    )
  )

# módulos
lr <- 0.0001
mlp <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

otimizador <- optim_sgd(mlp$parameters, lr = lr)

loss <- nn_mse_loss()


for (i in 1:1000) {
  y_pred <- mlp(X1)
  l <- loss(y_pred, y)
  # zero o gradiente
  otimizador$zero_grad()
  # calculo o gradiente
  l$backward()
  # atualizo parâmetros
  otimizador$step()
}

mlp$parameters

mlp(X1)

ggplot(cars_scaled) +
  aes(speed, dist) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_line(
    data = data.frame(
      speed = cars_scaled$speed,
      dist = as.numeric(y_pred)
    ),
    colour = "darkgreen"
  )
