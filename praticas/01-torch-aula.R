library(torch)

# criando tensores -------------------------------------

t1 <- torch_tensor(1)

t2 <- torch_tensor(c(1, 2, 3))

class(t2)

t1$dtype
t1$device
t1$shape

t1_int <- t1$to(dtype = torch_int())
t1_cuda <- t1$to(device = "cuda")

# outras formas de criar tensores

torch_tensor(1:4)

torch_tensor(matrix(c(1,1,2,2), nrow = 2))

torch_tensor(array(c(1,1,2,2,3,3,4,4), dim = c(2,2,2)))

set.seed(123)
runif(10)

set.seed(123)

torch_randn(3,3)
torch_rand(4,4)

# se precisar fazer de forma reprodutível

local_torch_manual_seed(123)
torch_manual_seed(123)
torch_randn(3,3)

cars |> 
  as.matrix() |> 
  torch_tensor()

# o torch não gosta de data frames
cars |> 
  torch_tensor()

model.matrix(~ Petal.Length + Species, data = iris) |> 
  torch_tensor()

# aritmética

tensor1 <- torch_tensor(c(1, 2))
tensor2 <- torch_tensor(c(3, 4))

tensor1$add(tensor2)

# soma inplace
tensor1$add_(tensor2)

# atalhos

tensor1 + tensor2

# detalhe técnico
# torch:::`+.torch_tensor`

tensor1 * tensor2
tensor1 / tensor2

tensor1$dot(tensor2)

# indexação e slicing

tensor_a <- torch_tensor(
  array(1:27, dim = c(3,3,3))
)

tensor_a[1:2,,2]

tensor_a[1:2,..]

torch_tensor(array(1:81, dim = c(3,3,3,3)))[1,..]

print(torch_tensor(array(1:81, dim = c(3,3,3,3))), n = -1)

vetor <- c(1:10)
vetor[-1]

torch_tensor(1:10)[5:-1]

# mudar a forma de vetores e matrizes

tensor_1dim <- torch_arange(1, 20, 2)

tensor_2dim <- tensor_1dim$view(c(2, 5))

# se colocar dimensões que não funcionam, dá erro
tensor_1dim$view(c(2, 6))

tensor_1dim$storage()$data_ptr()
tensor_2dim$storage()$data_ptr()
tensor_1dim$add_(100)

tensor_1dim[3] <- 3000
tensor_1dim

tensor_1dim_2 <- torch_tensor(matrix(1:8, ncol = 2))
tensor_1dim_2_t <- tensor_1dim_2$t()
# nesse caso não funciona
tensor_1dim_2_t$view(c(4,2))
# alternativa: reshape
tensor_1dim_2_t$reshape(c(4,2))

# squeeze, unsqueeze

tensor_1dim <- torch_arange(1, 20, 2)

tensor_1dim$unsqueeze(2)

unsqueezed <- tensor_1dim$unsqueeze(2)

tensor_1dim$unsqueeze(1)$unsqueeze(3)

unsqueezed$squeeze()

tensor_1dim$unsqueeze(1)$unsqueeze(3)$squeeze()

## até aqui: vimos as operações básicas do torch

## Broadcasting

(t_a <- torch_rand(c(3,1)))
(t_b <- torch_rand(c(1,4)))

t_a + t_b

t_a_namao <- cbind(as.numeric(t_a), as.numeric(t_a), as.numeric(t_a), as.numeric(t_a)) |> 
  torch_tensor()

t_b_namao <- rbind(as.numeric(t_b), as.numeric(t_b), as.numeric(t_b)) |> 
  torch_tensor()

t_a_namao + t_b_namao

# 3,1
# 1,4

# ->
# 3,4
# 1,4

# ->
# 3,4
# 3,4

# 1,3,1
#   1,4
# ->
# 1,3,4
#   1,4
# ->
# 1,3,4
#   3,4
# ->
# 1,3,4
# 1,3,4

t_a$unsqueeze(1) + t_b

# Operações matriciais

t1 <- torch_randn(c(2,3))
t2 <- torch_randn(c(3,2))

torch_matmul(t1, t2)

t1$matmul(t2)

## não funciona!
t1 %*% t2


t1$t()

torch_t(t1)


t_quadrada <- torch_randn(c(3,3))

t_quadrada$det()
torch_det(t_quadrada)

t_quadrada$inverse()

torch_inverse(t_quadrada)
linalg_inv(t_quadrada)

# regressão linear!

mtcars_mat <- torch_tensor(as.matrix(mtcars))
X <- mtcars_mat[,2:-1]
y <- mtcars_mat[,1]

X <- X$to(device = "cuda")
y <- y$to(device = "cuda")

XtX <- X$t()$matmul(X)
XtX_inv <- XtX$inverse()

# resultados da aplicação das matrizes 
beta <- XtX_inv$matmul(X$t())$matmul(y)

lm(mpg ~ . - 1, data = mtcars)
