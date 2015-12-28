# Practical-Machine-Learning-Writeup
My Practical Machine Learning Writeup

*NOTE: My apologise for the minuts of delay in my submit. I have a confusion with the timezone, sorry about that.* 

## Load Libraries
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(e1071)
```

## Read the data sources
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
training = read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), na.strings=c("NA","#DIV/0!",""))
testing  = read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), na.strings=c("NA","#DIV/0!",""))

```

## Prepare data
Unifie variables names of training and testing data sources 
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
names(testing)=c(names(training)[-length(names(training))],"probolem_id")
```

Create a 30% partition of training data for use it leater to do a test before the 20 predictions proposed in the activity.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
set.seed(100490)
inTrain = createDataPartition(y=training$classe, p=0.7, list=FALSE)

training = training[inTrain,]
pretest = training[-inTrain,]
dim(training)
dim(pretest)
```

Explore the data (always the training date, never the test data) to delete de variables without information, with a large number of NA's or "zero covariates":
* Exploring the training data, we observe that the first six variables are only index or references. Then it have to be removed.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
str(training,list.len=15)
table(training$classe)

training = training[, 7:160]
pretest = pretest[, 7:160]
testing = testing[, 7:160]
```
* We remove the variables with more than 50% of observations with NA. We do this because we have a lot of X's variables. If we did not have a lot of X's variables, we could use the imput functions to imput NA's; but it is not necessary in this case. And as always, we do the same with test data.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
training = training[,colSums(is.na(training))< (nrow(training)/2) ]

pretest = pretest[,names(training)]
testing = testing[,c(names(training)[-length(names(training))],"probolem_id")]

```

* Identify the "zero covariates". In this case, there are not.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
nzv_cols = nearZeroVar(training)
nzv_cols
```

## Developing the model
We first adjust the random forest model to find the most "important" covariants to reduce the nomber of covariants in the model to make easier the model interpretability.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
modFit = randomForest(classe~., data=training, importance=T, ntree=100)
varImpPlot(modFit)
```

<img class=center src=http://i65.tinypic.com/v2xrav.jp height=450>

Using the Accuracy and Gini graphs above, we select the top 10 variables that we will use for model building.

The 10 covariates selected are: yaw_belt, roll_belt, num_window, pitch_belt, magnet_dumbbell_y, magnet_dumbbell_z, pitch_forearm, accel_dumbbell_y, roll_arm, and roll_forearm.

We procede to analize the correlate between the 10 selected varibales.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
corr = cor(training[,c("yaw_belt","roll_belt","num_window","pitch_belt","magnet_dumbbell_z","magnet_dumbbell_y","pitch_forearm","accel_dumbbell_y","roll_arm","roll_forearm")])
diag(corr) = 0
which(abs(corr)>0.75, arr.ind=T)
cor(training$roll_belt,training$yaw_belt)
```
```
          row col
roll_belt   2   1
yaw_belt    1   2
```

These two covariants have a 0.814 correlation, then we should choose only one of them to enter in the model. To selecte one, we use the classification model to see the importance of this two covaraints and choose the most important:
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
modFit2 = rpart(classe~., data=training, method="class")
prp(modFit2)
```
<img class=center src=http://i66.tinypic.com/2f0dgk3.png height=450>

We observe that the most important one is "roll_belt" covariant, as a resault the 9 covariants that we use are these:
*roll_belt
*num_window
*pitch_belt
*magnet_dumbbell_y
*magnet_dumbbell_z
*pitch_forearm
*accel_dumbbell_y
*roll_arm
*roll_forearm

We procede to develop the model with these covaraints:
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
modFit3 <- train(classe~roll_belt+num_window+pitch_belt+magnet_dumbbell_y+magnet_dumbbell_z+pitch_forearm+accel_dumbbell_y+roll_arm+roll_forearm,
                  data=training,
                  method="rf",
                  trControl=trainControl(method="cv",number=2),
                  prox=T,
                  verbose=T,
                  allowParallel=T)
```

To evaluate the model before do the 20 proposed predictions, we use de "pretest" data reserved from the beginning.
```{r setup, cache = F, echo = F, message = F, warning = F, tidy = F}
pred <- predict(modFit3, newdata=pretest)
confusionMat <- confusionMatrix(pred, pretest$classe)
confusionMat
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1150    0    0    0    0
         B    0  815    0    0    0
         C    0    0  721    0    0
         D    0    0    0  708    0
         E    0    0    0    0  727

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9991, 1)
    No Information Rate : 0.2791     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000    1.000   1.0000   1.0000
Specificity            1.0000   1.0000    1.000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000    1.000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000    1.000   1.0000   1.0000
Prevalence             0.2791   0.1978    0.175   0.1718   0.1764
Detection Rate         0.2791   0.1978    0.175   0.1718   0.1764
Detection Prevalence   0.2791   0.1978    0.175   0.1718   0.1764
Balanced Accuracy      1.0000   1.0000    1.000   1.0000   1.0000
```

The results show us that the accurancy of the model are very nice, in this case we have achieved 1 of accurancy, a perfect result! :)


## Predictions
We do the predictions for the testing data and we use the provided code to prepare the 20 documents to evaluation.
```
predictions = predict(modFit, newdata=testing)
testing$classe = predictions

answers = testing$classe
write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_",i,".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
write_files(answers)
```
