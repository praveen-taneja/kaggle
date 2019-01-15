# kaggle earthquake prediction - 2019
# author - praveen taneja

rm(list = ls())

library(data.table)
library(ggplot)
library(dygraphs)
library(magrittr)

source('C:/Users/taneja/win_gdrive/coding/pt_utils.R')

trainData <- 'C:/Users/taneja/coding/kaggle_data/earth_quake_comp_2018/all/train.csv'
testData <- 'C:/Users/taneja/coding/kaggle_data/earth_quake_comp_2018/all/test/seg_0012b5.csv'
chunkSize <- 150000
numTrainRows <- 629145480
numTrainingEpochs <- floor(629145480/150000)
signalNames <- c('acoustic_data', 'time_to_failure')
labelName <- 'time_to_failure'


featuresDT <- CalcFeatures(trainData
                           , numTrainingEpochs
                           , c('avg', 'min', 'max', 'stddev')
                           , chunkSize = chunkSize
                           , signalNames = signalNames
                           , labelName = labelName)
#trainDT <- CreateDataChunk(trainData, chunkSize, chunkSize*2, signalNames)

# dygraph(trainDT) %>% 
#   dyAxis(name = 'y', label = 'acoustic') %>% 
#   dyAxis(name = 'y2', label = 'time') %>%
#   dySeries('time_to_failure', axis = 'y2')


