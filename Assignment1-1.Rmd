---
title: "Assignment 2"
author: "Chew Yoong, Chang"
date: "December 27, 2015"
output: html_document
---

### Data

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

### Citation

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har]. 

### Goal

The goal is to find a model that can predicht the classes below based on the sensor data of an activity.

- exactly according to the specification (Class A)
- throwing the elbows to the front (Class B)
- lifting the dumbbell only halfway (Class C)
- lowering the dumbbell only halfway (Class D)
- throwing the hips to the front (Class E)

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes." [1]
Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. All other available variables after cleaning will be used for prediction.
Two models will be tested using decision tree and random forest algorithms. The model with the highest accuracy will be chosen as our final model.

####Cross-validation

Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: subTraining data (75% of the original Training data set) and subTesting data (25%). Our models will be fitted on the subTraining data set, and tested on the subTesting data. Once the most accurate model is choosen, it will be tested on the original Testing data set.

####Expected out-of-sample error

The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion of correct classified observation over the total sample in the subTesting data set. Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations/total observations in the Test data set, which is the quantity: 1-accuracy found from the cross-validation data set.

Reasons for my choices

Our outcome variable "classe" is an unordered factor variable. Thus, we can choose our error type as 1-accuracy. We have a large sample size with N= 19622 in the Training data set. This allow us to divide our Training sample into subTraining and subTesting to allow cross-validation. Features with all missing values will be discarded as well as features that are irrelevant. All other features will be kept as relevant variables.
Decision tree and random forest algorithms are known for their ability of detecting the features that are important for classification [2]. Feature selection is inherent, so it is not so necessary at the data preparation phase. Thus, there won't be any feature selection section in this report.

## Loading data

Below the code for loading the data (which was already downloaded to my harddrive).

```{r echo=FALSE, warning=FALSE, message=FALSE}
library("knitr")
opts_chunk$set(cache=FALSE)
options(scipen=999)
```
```{r echo=TRUE, warning=FALSE, message=FALSE}
library("dplyr")
library("caret")
library("tidyr")
library("rpart.plot")
library("randomForest")

set.seed(54356)

pml.training <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!", ""), dec = ".")
```

## Cleaning data

The data needs to be cleaned before it can be used for modelling. I tried several different ways of cleanng the data  before I came up with the following steps:

1. Remove new_window == yes observations because these seem to be aggragates of other column.
2. Remove the first columns (id, timestamps, subject name) because they are not usefull in predicting.
3. Remove all columns with NA values.

```{r}
x <- pml.training %>% filter(new_window == "no")
x <- x[8:length(x)]
x <- x[ , ! apply(x ,2 ,function(x) any(is.na(x)) ) ]
```

## Creating training and testset for cross validation

The assignment provides a training and testset, however, the testset is not really a testset, but more a submission set. To be able to validate the model the provided trainingset will be split in a training and testset for the modelling.

```{r}
inTrain <- createDataPartition(y=x$classe,
                               p=0.6, list=FALSE)
trainingset <- subset(x[inTrain,])
testset <- subset(x[-inTrain,])

```

## Cross validation

The default resampling scheme for the caret train function is bootstrap. I have used custom settings instead by setting the below `trainControl`.

The out of sample error should be higher than the in sample error because the the model is based on the training set and will therefor most likely have a slightly worst performance on the testset. This will be shown further in the project.

## Selecting variables

First I made a model on a small part of the training set (for speed). Then I selected the 20 most important variables with `varImp` and run the model again. I repeated this, meanwhile balancing the accurancy and number of variables. With 10 variables I still got very good accuracy and speed on the model with the full training set.

###A look at the Data

The variable "classe" contains 5 levels: A, B, C, D and E. A plot of the outcome variable will allow us to see the frequency of each levels in the subTraining data set and compare one another.

```{r}
dim (trainingset)
head(trainingset)
plot(trainingset$classe, col="blue", main="Bar Plot of levels of the variable classe within the subTraining data set", xlab="classe levels", ylab="Frequency")
```


# First prediction model :- Decision Tree Model

```{r}
model1 <- rpart(classe ~ ., data=trainingset, method="class")

# Predicting:
prediction1 <- predict(model1, testset, type = "class")

# Plot of the Decision Tree
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0)
```

## Test results on our subTesting data set:

```{r}
confusionMatrix(prediction1, testset$classe)
```


#Second prediction model :- Using Random Forest

```{r}
model2 <- randomForest(classe ~. , data=trainingset, method="class")

# Predicting:
prediction2 <- predict(model2, testset, type = "class")

# Test results on subTesting data set:
confusionMatrix(prediction2, testset$classe)
```

##Decision

As expected, Random Forest algorithm performed better than Decision Trees.
Accuracy for Random Forest model was 0.995 (95% CI: (0.993, 0.997)) compared to 0.739 (95% CI: (0.727, 0.752)) for Decision Tree model. The random Forest model is choosen. The accuracy of the model is 0.995. The expected out-of-sample error is estimated at 0.005, or 0.5%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. Our Test data set comprises 20 cases. With an accuracy above 99% on our cross-validation data, we can expect that very few, or none, of the test samples will be missclassified.






