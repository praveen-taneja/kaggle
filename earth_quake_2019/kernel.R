# kaggle earthquake prediction - 2019
# author = praveen taneja

# rm(list = ls())

library(data.table)
library(ggplot2)
library(dygraphs)
library(magrittr)
library(caret)
library(xgboost)
library(e1071)

# parameters
trainData <- '../input/train.csv'
testData <- '../input/test/seq_004cd2.csv'
sampleSubmission <- '../input/sample_submission.csv'

chunkSize <- 150000
numTrainRows <- 629145480
numTrainingEpochs <- ceiling(629145480/150000)
signalNames <- c('acoustic_data', 'time_to_failure')
labelName <- 'time_to_failure'


trainDT <- fread(trainData)
#trainDT <- trainDT[1:(numTrainingEpochs*chunkSize), ] # keep data for whole chunks
trainDT[, epochId := rep(1:numTrainingEpochs, each = chunkSize)]
#testDT <- trainDT[1:(chunkSize*3), ]
trainFeaturesDT <- trainDT[, list('avg' = mean(acoustic_data, na.rm = TRUE)
                                  , 'min' = min(acoustic_data, na.rm = TRUE)
                                  , 'max' = max(acoustic_data, na.rm = TRUE)
                                  , 'sd' = sd(acoustic_data, na.rm = TRUE)
                                  , 'kurtosis' = e1071::kurtosis(acoustic_data, na.rm = TRUE)
                                  , 'time_to_failure' = tail(time_to_failure,1)
), by = 'epochId']

trainFeaturesDT[, epochId := NULL]
#fwrite(trainFeaturesDT, 'trainFeatures.csv')
#DT1 <- fread('../trainFeatures.csv')

x.train <- trainFeaturesDT[, .SD, .SDcols = setdiff(names(trainFeaturesDT), 'time_to_failure')]
y.train <- as.numeric(trainFeaturesDT[, time_to_failure])

# model training
xgb.trControl <- caret::trainControl(
  method = "repeatedcv"
  , number = 4 # folds for resampling iters
  , repeats = 4
  , verboseIter = T
  , savePredictions = T
  #, summaryFunction = "mnLogLoss"
  #, classProbs = TRUE
)

xgb.grid <- expand.grid(nrounds = 100
                        , eta = c(0.1)
                        , gamma = c(1)
                        , max_depth = c(3)
                        , subsample = 0.5
                        , colsample_bytree = 0.7
                        , min_child_weight = 1)

xgb_fit <- caret::train(x = x.train
                        , y = y.train # need to convert to vector
                        , method = 'xgbTree'
                        , trControl = xgb.trControl
                        , tuneGrid = xgb.grid
                        #, metric = 
)

xgb_fit

preds <- predict(xgb_fit, newdata = x.train)

DT <- data.table('observed' = y.train, 'predicted' = preds)

ggplot(DT, aes(x = observed, y = predicted)) + geom_point() + coord_cartesian(xlim = c(0,15), ylim = c(0, 15)) + geom_abline(color = 'red')

# kaggle submission
submissionDT <- fread(sampleSubmission, colClasses = c("character", "double"))

for(theSegId in submissionDT$seg_id){
  
  testDT <- fread(paste0('../input/test/', theSegId, '.csv'))
  
  x.test <- testDT[, list('avg' = mean(acoustic_data, na.rm = TRUE)
                          , 'min' = min(acoustic_data, na.rm = TRUE)
                          , 'max' = max(acoustic_data, na.rm = TRUE)
                          , 'sd' = sd(acoustic_data, na.rm = TRUE)
                          , 'kurtosis' = e1071::kurtosis(acoustic_data, na.rm = TRUE)
  )
  ]
  preds <- predict(xgb_fit, newdata = x.test)
  #print(preds)
  submissionDT[seg_id == theSegId, time_to_failure := preds]
}

fwrite(submissionDT, 'submission.csv')