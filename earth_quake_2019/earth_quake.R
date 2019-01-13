# kaggle earthquake prediction - 2019
# author - praveen taneja

rm(list = ls())

library(data.table)
library(ggplo)
library(dygraphs)
library(magrittr)

trainData <- 'C:/Users/taneja/coding/kaggle_data/earth_quake_comp_2018/all/train.csv'

trainDT <- fread(trainData, nrows = 500000)
trainDT[, idx := 1:nrow(trainDT)]
setcolorder(trainDT, c('idx', 'acoustic_data', 'time_to_failure')) 

dygraph(trainDT) %>% 
  dyAxis(name = 'y', label = 'acoustic') %>% 
  dyAxis(name = 'y2', label = 'time') %>%
  dySeries('time_to_failure', axis = 'y2')