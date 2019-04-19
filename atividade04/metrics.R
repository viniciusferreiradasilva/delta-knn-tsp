# Métricas de desempenho.

# Erro quadrático médio.
mse <- function(y, predicted){
  return(mean(y - predicted)^2)
}

# Erro quadrático.
squaredError <- function(y, predicted){
  return((y - predicted)^2)}

# Erro absoluto médio.
ase <- function(y, predicted){
  return(mean(abs(y - predicted)))
}

# Erro absoluto.
absError <- function(y, predicted){
  return(abs(y - predicted))
}

# Theil's U.
tu <- function(y, predicted){
  return(sum((tail(y, -1) - tail(predicted, -1))^2) / sum(diff(tail(y, -1))^2))
}

# POCID.
pocid <- function(y, predicted){
  return(sum(diff(predicted) * diff(y) > 0) / length(y))
}