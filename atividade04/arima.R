library('ggplot2')
library('forecast')
library('tseries')
source('metrics.R')
source('datahandle.R')

# Carrega o csv de todas as ações.
sp500 <- read.csv('../input/all_stocks_5yr.csv', header=TRUE, stringsAsFactors=FALSE)

# Recupera apenas as ações da NVDA.
nvda <- subset(sp500, Name=="NVDA")

# Converte os valores para Date.
nvda$Date <- as.Date(nvda$date)

# Cria uma série temporal com os dados de abertura.
open_ts <- ts(nvda[, c('open')], frequency=52)
# Identifica e substitui outliers usando suavização e decomposição da série.
nvda$open <- tsclean(log(open_ts))
# Calcula a média móvel para 7 dias do preço de abertura.
nvda$open_ma7 <- ma(nvda$open, order=7)
# Realiza 1 diff da série temporal para tornar não-estacionária.
nvda$diff_open <- c(diff(nvda$open), NA)


# Cria uma série temporal com os dados de fechamento.
close_ts <- ts(nvda[, c('close')], frequency=52)
# Identifica e substitui outliers usando suavização e decomposição da série.
nvda$close <- tsclean(log(close_ts))
# Calcula a média móvel para 7 dias do preço de fechamento.
nvda$close_ma7 <- ma(nvda$close, order=7)
# Realiza 1 diff da série temporal para tornar não-estacionária.
nvda$diff_close <- c(diff(nvda$close), NA)

# Normalizando o valor de volume pelo volume máximo.
nvda$volume <- (nvda$volume)/(max(nvda$volume))
# Cria uma série temporal com os dados de volume.
volume_ts <- ts(nvda[, c('volume')], frequency=52)
# Identifica e substitui outliers usando suavização e decomposição da série.
nvda$close <- tsclean(volume_ts)
# Calcula a média móvel para 7 dias do volume.
nvda$volume_ma7 <- ma(nvda$volume, order=7)

# Plota o gráfico dos valores real, média móvel semanal e mensal.
ggplot() +
       geom_line(data = nvda, aes(x = Date, y = nvda$open, colour = "Real")) +
       geom_line(data = nvda, aes(x = Date, y = nvda$open_ma7, colour = "Weekly Moving Average"))  +
       ylab('open')
