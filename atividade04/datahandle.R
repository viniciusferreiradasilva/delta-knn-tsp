# Funções para separação dos dados.

# Função de dada uma série e os valores de início e fim, retorna a janela referente na série.
slidingWindow <- function(x, start, end){
  return(x[start:end])
}

