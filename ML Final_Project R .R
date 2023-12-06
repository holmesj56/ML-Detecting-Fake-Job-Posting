rm(list=ls())
setwd("/Users/apoorvauplap/apo/ASU/PhD/Courses/Fall23/EGR 598/Project/data")

#install.packages("magrittr")
#install.packages("tidyverse")
#install.packages("car")
#install.packages("caret")
#install.packages("OptimalCutpoints")
#install.packages("generalhoslem")
#install.packages("dplyr") 
#install.packages("tidyr") 
#install.packages("nnet") 
#install.packages('ipred')
#install.packages('DescTools')
#install.packages('glmtoolbox')

library(car)
library(generalhoslem)
library(caret)
library(OptimalCutpoints)
library(magrittr)
library(tidyverse)
library("party")
library("RCurl")
library(tidyr)
library(ipred)
library(dplyr)
library(tidyverse)
library(dslabs)
library(dplyr)
library(nnet)
library(DescTools)
library(glmtoolbox)
library(gridExtra)
library(grid)
library(tinytex)
library(caret)
library(lubridate)
library(tictoc)
library(e1071)
library(data.table)
library(tidytext)
library(stopwords)
library(readr)
library(tm)
library(SnowballC)
library(wordcloud)
library(randomForest)

# Get data
job_data <- read.csv("fake_job_postings.csv")
head(job_data)

str(job_data)
summary(job_data)

# Create the corpus vector
data_vec <- Corpus(VectorSource(job_data$description))
# Remove punctuation
data_vec <- tm_map(data_vec, removePunctuation)
# Remove stop words
data_vec <- tm_map(data_vec, removeWords, stopwords(kind = "en"))
# Perform stemming
data_vec <- tm_map(data_vec, stemDocument)

# High frequency words
frequencies <- DocumentTermMatrix(data_vec)
# Remove sparse data
sparse_data <- removeSparseTerms(frequencies, 0.995)
# Converting to dataframe for further analysis
sparse_df <- as.data.frame(as.matrix(sparse_data))
# Assigning column names
colnames(sparse_df) <- make.names(colnames(sparse_df))
# Adding the dependent variable
sparse_df$fraudulent <- job_data$fraudulent 
# Removing duplicate column names
colnames(sparse_df) <- make.unique(colnames(sparse_df), sep = "_")

sparse_df <- sparse_data_df
summary(sparse_df)

set.seed(192, sample.kind = "Rounding")
test_index <- createDataPartition(y = sparse_df$fraudulent, times = 1, p = 0.1, list= FALSE)
train_set <- sparse_df[-test_index, ]
validation <- sparse_df[test_index, ]
train_set$fraudulent = as.factor(train_set$fraudulent)
validation$fraudulent = as.factor(validation$fraudulent)

# SVM

ctrl <- trainControl(method = "cv", verboseIter = TRUE, number = 5)
grid_svm <- expand.grid(C = c(0.01))
svm_fit <- train(fraudulent ~ .,data = train_set, 
                 method = "svmLinear", preProcess = c("center","scale"),
                 tuneGrid = grid_svm, trControl = ctrl)

#Prediction
svm_pred <- predict(svm_fit, newdata = validation)
#Confusion Matrix
CM <- confusionMatrix(svm_pred, validation$fraudulent)

CM
