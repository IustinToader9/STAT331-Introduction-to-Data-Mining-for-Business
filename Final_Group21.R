#-----------------------------------Preliminary---------------------------
rm(list=ls())
setwd("/Users/iustintoader/Desktop/Drexel 20:21/Winter 20:21/STAT331/FinalProject_STAT331")

#Load required packages
library(DescTools)
library(caret)
library(NeuralNetTools)
library(randomForest)

#Import dataset
ChurnDF <- read.csv("CustomerChurn.csv", stringsAsFactors = FALSE)


#-----------------------------------------------General data exploration and preparation----------------------------------------
#First we set the data types for the dataset
#Nominal variables
facs <- c("customerID", "gender", "SeniorCitizen", "Partner", "Dependents", 
          "PhoneService", "InternetService", "PaperlessBilling", "PaymentMethod", "Churn")

#The only ordinal variable is Contract

#Numerical variables
nums <- c("tenure", "MonthlyCharges", "TotalCharges")

ChurnDF[,facs] <- lapply(X=ChurnDF[, facs], FUN=factor)
ChurnDF$Contract <- factor(x=ChurnDF$Contract, levels=c("Month-to-month", "One year", "Two year"), 
                           ordered=TRUE)

str(ChurnDF)
Desc(x=ChurnDF, plotit=FALSE)
PlotMiss(x=ChurnDF)
#We can see that there are 2 NA values in the TotalCharges column. Intuitively,
#this should only occur if the tenure of the customer is 0.
ChurnDF[ChurnDF$tenure == 0, ]
#Our assumption is correct. We will impute the  NAs with the median value of TotalCharges
pp <- preProcess(x = ChurnDF,
                 method = "medianImpute")
ChurnDF <- predict(object = pp, 
                 newdata = ChurnDF)
#We check for duplicated customerIDs
ChurnDF[duplicated(ChurnDF$customerID)]
#Our target variable Y is Churn, which we will take a closer look at
Desc(ChurnDF$Churn)
#We can see that there is a class imbalance which we will address later on

#We will now also check for outliers in the dataset
cen_sc <- preProcess(x = ChurnDF,
                     method = c("center", "scale"))
churn_sc=predict(object = cen_sc, newdata = ChurnDF)
churn_sc[abs(churn_sc$TotalCharges) > 3, ]
#No outliers detected


#-------------------------------------Artificial Neural Network Analysis--------------------------------------
#Our first analysis method will be a supervised model for classification. We will use an Artificial
#Neural Network in order to predict if any given customer will leave the company or not.

#ANN data preparation
#1. Missing values:
#We have already accounted for them in the general data preparation
#2. Binarization:
#We need to binarize all categorical variables
ChurnDF_bin <- ChurnDF
#We create a convenience vector for categorical variables with only 2 class levels
cat_2lv <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling")
ChurnDF_bin[, cat_2lv] <- lapply(X=ChurnDF[,cat_2lv], FUN=class2ind, drop2nd=TRUE)
#For the only ordinal variable we have, we need to convert it to unordered before binarizing it
ChurnDF_bin$Contract <- factor(x=ChurnDF_bin$Contract, ordered = FALSE)

#We create and append the dummy variables for InternetService
cats <- dummyVars(formula = ~ InternetService, data=ChurnDF_bin)
cat_dums <- predict(object = cats, newdata=ChurnDF_bin)
ChurnDF_bin <- cbind(ChurnDF_bin, cat_dums)

#We create and append the dummy variables for Contract
cats <- dummyVars(formula = ~ Contract, data=ChurnDF_bin)
cat_dums <- predict(object = cats, newdata=ChurnDF_bin)
ChurnDF_bin <- cbind(ChurnDF_bin, cat_dums)

#We create and append the dummy variables for PaymentMethod
cats <- dummyVars(formula = ~ PaymentMethod, data=ChurnDF_bin)
cat_dums <- predict(object = cats, newdata=ChurnDF_bin)
ChurnDF_bin <- cbind(ChurnDF_bin, cat_dums)


#We drop the original variables from the dataframe, plus customerID which is not relevant to our analysis
ChurnDF_bin <- ChurnDF_bin[,-c(1, 8:9, 11)]


#3. Rescale numeric variables:
#We will deal with rescaling during hyperparameter tuning


########### Base Model Training and Testing
sub_ann <- createDataPartition(y=ChurnDF_bin$Churn, p=0.8, list=FALSE)

train_ann <- ChurnDF_bin[sub_ann,]
test_ann <- ChurnDF_bin[-sub_ann,]

#We will proceed to hyperparameter tuning to find the optimal number of hidden nodes and weight decay
#We will use a grid search and a 10-fold cross validation repeated 10 times.
grids <- expand.grid(size = seq(from = 1, # min value
                                          to = 7, # max value
                                          by = 1), # count by
                               decay = seq(from = 0, # min value
                                           to = 0.1, # max value
                                           by = 0.01)) # count by
grids
ctrl <- trainControl(method = "repeatedcv",
                     number = 10, # 10 folds
                     repeats = 10, # 10 repeats
                     search = "grid") # grid search

set.seed(105828)

annMod <- train(form = Churn ~., # use all other variables to predict target
                data = train_ann, # training data
                preProcess = "range", # apply min-max normalization
                method = "nnet", # use nnet()
                trControl = ctrl, 
                tuneGrid = grids, # search over the created grid
                trace = FALSE) # suppress output

annMod
#Tuned ANN model visualization
plotnet(mod_in = annMod$finalModel, # nnet object
        pos_col = "darkgreen", # positive weights are shown in green
        neg_col = "darkred", # negative weights are shown in red
        bias = FALSE, # do not plot bias
        circle_cex = 4, # reduce circle size (default is 5)
        cex_val = 0.6) # reduce text label size (default is 1)

#Training performance
tune.tr.preds_ann <- predict(object = annMod, # tuned model
                         newdata = train_ann) # training data
tune_tr_conf_ann <- confusionMatrix(data = tune.tr.preds_ann, # predictions
                                reference = train_ann$Churn, # actual
                                positive = "Yes",
                                mode = "everything")

#Testing performance
tune.te.preds_ann <- predict(object = annMod, # tuned model
                         newdata = test_ann) # testing data
tune_te_conf_ann <- confusionMatrix(data = tune.te.preds_ann, # predictions
                                reference = test_ann$Churn, # actual
                                positive = "Yes",
                                mode = "everything")
tune_te_conf_ann

### Performance and Goodness of Fit
# Overall
cbind(Training = tune_tr_conf_ann$overall,
      Testing = tune_te_conf_ann$overall)
# By Class
cbind(Training = tune_tr_conf_ann$byClass, 
      Testing = tune_te_conf_ann$byClass)


########### Data resampling for Class Imbalance
#Now we will try to resample the data 
#We will first create a convenience vector for the predictor variables
preds <- names(train_ann)[!names(train_ann) %in% "Churn"]

#####Random Undersampling
set.seed(105828)
train_ann_ds <- downSample(x=train_ann[,preds],
                       y=train_ann$Churn,
                       yname="Churn")
par(mfrow = c(1,2)) # split plot window into 2 columns (1 row)
plot(train_ann$Churn, main = "Original")
plot(train_ann_ds$Churn, main = "RUS")
#Training undersampled model
annMod_RUS <- train(form = Churn ~., # use all other variables to predict target
                data = train_ann_ds, # training data
                preProcess = "range", # apply min-max normalization
                method = "nnet", # use nnet()
                trControl = ctrl, 
                tuneGrid = grids, # search over the created grid
                trace = FALSE) # suppress output
annMod_RUS

# Training Performance
tune.tr.preds_ann_RUS <- predict(object = annMod_RUS, # tuned model
                         newdata = train_ann) # training data
tune_tr_conf_ann_RUS <- confusionMatrix(data = tune.tr.preds_ann_RUS, # predictions
                                reference = train_ann$Churn, # actual
                                positive = "Yes",
                                mode = "everything")

#Testing performance
tune.te.preds_ann_RUS <- predict(object = annMod_RUS, # tuned model
                         newdata = test_ann) # testing data
tune_te_conf_ann_RUS <- confusionMatrix(data = tune.te.preds_ann_RUS, # predictions
                                reference = test_ann$Churn, # actual
                                positive = "Yes",
                                mode = "everything")
tune_te_conf_ann_RUS

### Performance and Goodness of Fit
# Overall
cbind(Training = tune_tr_conf_ann_RUS$overall,
      Testing = tune_te_conf_ann_RUS$overall)
# By Class
cbind(Training = tune_tr_conf_ann_RUS$byClass, 
      Testing = tune_te_conf_ann_RUS$byClass)

#####Random Oversampling
set.seed(105828)
train_ann_us <- upSample(x=train_ann[,preds],
                       y=train_ann$Churn,
                       yname="Churn")

plot(train_ann_us$Churn, main = "ROS")

#Training oversampled model
annMod_ROS <- train(form = Churn ~., # use all other variables to predict target
                    data = train_ann_us, # training data
                    preProcess = "range", # apply min-max normalization
                    method = "nnet", # use nnet()
                    trControl = ctrl, 
                    tuneGrid = grids, # search over the created grid
                    trace = FALSE) # suppress output
annMod_ROS

# Training Performance
tune.tr.preds_ann_ROS <- predict(object = annMod_ROS, # tuned model
                             newdata = train_ann) # training data
tune_tr_conf_ann_ROS <- confusionMatrix(data = tune.tr.preds_ann_ROS, # predictions
                                    reference = train_ann$Churn, # actual
                                    positive = "Yes",
                                    mode = "everything")

#Testing performance
tune.te.preds_ann_ROS <- predict(object = annMod_ROS, # tuned model
                             newdata = test_ann) # testing data
tune_te_conf_ann_ROS <- confusionMatrix(data = tune.te.preds_ann_ROS, # predictions
                                    reference = test_ann$Churn, # actual
                                    positive = "Yes",
                                    mode = "everything")
tune_te_conf_ann_ROS

### Performance and Goodness of Fit
# Overall
cbind(Training = tune_tr_conf_ann_ROS$overall,
      Testing = tune_te_conf_ann_ROS$overall)
# By Class
cbind(Training = tune_tr_conf_ann_ROS$byClass, 
      Testing = tune_te_conf_ann_ROS$byClass)


#Comparing CI Models
#Training Performance
cbind(Base = tune_tr_conf_ann$overall, # base model
      Under = tune_tr_conf_ann_RUS$overall, # undersampled model
      Over = tune_tr_conf_ann_ROS$overall) # oversampled model
# By Class
cbind(Base = tune_tr_conf_ann$byClass, # base model
      Under = tune_tr_conf_ann_RUS$byClass, # undersampled model
      Over = tune_tr_conf_ann_ROS$byClass) # oversampled model

# Testing Performance
# Overall
cbind(Base = tune_te_conf_ann$overall, # base model
      Under = tune_te_conf_ann_RUS$overall, # undersampled model
      Over = tune_te_conf_ann_ROS$overall) # oversampled model
# By Class
cbind(Base = tune_te_conf_ann$byClass, # base model
      Under = tune_te_conf_ann_RUS$byClass, # undersampled model
      Over = tune_te_conf_ann_ROS$byClass) # oversampled model



#--------------------------------Ensemble Methods - Random Forest Analysis---------------------------

#Since we are using ensemble methods of Decision Trees, the same considerations apply. Our missing values have already been
#imputed in the original dataset and no rescaling/standardization is necessary. As such, we can use the original ChurnDF dataframe.
#We will still make a copy of it in order to keep a reference to the customerID variable which will be dropped for redundance purposes.
ChurnDF_rf <- ChurnDF[ ,!names(ChurnDF) %in% "customerID"]

#We first split the data into training and testing subsets following the 85/15 rule
set.seed(105828)
sub_rf <- createDataPartition(y=ChurnDF_rf$Churn,
                              p=0.85,
                              list=FALSE)
train_rf <- ChurnDF_rf[sub_rf,]
test_rf <- ChurnDF_rf[-sub_rf,]

#####Base Untuned Model
#We will first apply the untuned model with the default m (mtry) value, which is equal to the square root of our
#number of predictors.
set.seed(831) # initialize random seed

rf_mod <- randomForest(formula = Churn ~. , # use all other variables to predict Churn
                       data = train_rf, # training data
                       importance = TRUE, # obtain variable importance 
                       ntree = 500) # number of trees in forest

# We can view basic output from the model
rf_mod


# Variable Importance Plot
#We can view the most important variables 
#in the Random Forest model
varImpPlot(x = rf_mod, # randomForest object
           main = "Variable Importance Plot") # title

## Training Performance
base.RFpreds <- predict(object = rf_mod, # RF model
                        type = "class") # class predictions

RF_btrain_conf <- confusionMatrix(data = base.RFpreds, # predictions
                                  reference = train_rf$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
RF_btrain_conf


## Testing Performance
base.teRFpreds <- predict(object = rf_mod, # RF model
                          newdata = test_rf, # testing data
                          type = "class")
RF_btest_conf <- confusionMatrix(data = base.teRFpreds, # predictions
                                 reference = test_rf$Churn, # actual
                                 positive = "Yes",
                                 mode = "everything")
RF_btest_conf

## Hyperparameter Tuning

# We will tune the number of variables to 
# randomly sample as potential variables to split on 
# (m, the mtry argument).

set.seed(831) # initialize random seed

tuneR <- tuneRF(x = train_rf[, -13],
                y = train_rf$Churn, 
                ntreeTry = 500,
                doBest = TRUE) 

# View basic model information
tuneR

# View variable importance for the tuned 
# model
varImpPlot(tuneR)

## Training Performance
tune.trRFpreds <- predict(object = tuneR, # tuned RF model
                          type = "class") # class predictions

RF_ttrain_conf <- confusionMatrix(data = tune.trRFpreds, # predictions
                                  reference = train_rf$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
RF_ttrain_conf


## Testing Performance
tune.teRFpreds <- predict(object = tuneR, # tuned RF model
                          newdata = test_rf, # testing data
                          type = "class")

RF_ttest_conf <- confusionMatrix(data = tune.teRFpreds, # predictions
                                 reference = test_rf$Churn, # actual
                                 positive = "Yes",
                                 mode = "everything")
RF_ttest_conf

### Performance and Goodness of Fit
# Overall
cbind(Training = RF_ttrain_conf$overall,
      Testing = RF_ttest_conf$overall)
# By Class
cbind(Training = RF_ttrain_conf$byClass, 
      Testing = RF_ttest_conf$byClass)


########### Data resampling for Class Imbalance
#Now we will try to resample the data 
#We will move straight to Hyperparameter tuning

####Random Undersampling
set.seed(105828)
train_rf_ds <- downSample(x=train_rf[, -13],
                       y=train_rf$Churn,
                       yname="Churn")

#Training undersampled model
set.seed(831) # initialize random seed

tuneR_ds <- tuneRF(x = train_rf_ds[, -13],
                y = train_rf_ds$Churn, 
                ntreeTry = 500,
                doBest = TRUE) 

# View basic model information
tuneR_ds

# View variable importance for the tuned 
# model
varImpPlot(tuneR_ds)

## Training Performance
tune.trRFpreds_ds <- predict(object = tuneR_ds, # tuned RF model
                          type = "class") # class predictions

RF_ttrain_conf_ds <- confusionMatrix(data = tune.trRFpreds_ds, # predictions
                                  reference = train_rf_ds$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
RF_ttrain_conf_ds


## Testing Performance
tune.teRFpreds_ds <- predict(object = tuneR_ds, # tuned RF model
                          newdata = test_rf, # testing data
                          type = "class")

RF_ttest_conf_ds <- confusionMatrix(data = tune.teRFpreds_ds, # predictions
                                 reference = test_rf$Churn, # actual
                                 positive = "Yes",
                                 mode = "everything")
RF_ttest_conf_ds

### Performance and Goodness of Fit
# Overall
cbind(Training = RF_ttrain_conf_ds$overall,
      Testing = RF_ttest_conf_ds$overall)
# By Class
cbind(Training = RF_ttrain_conf_ds$byClass, 
      Testing = RF_ttest_conf_ds$byClass)


#####Random Oversampling
set.seed(105828)
train_rf_us <- upSample(x=train_rf[, -13],
                          y=train_rf$Churn,
                          yname="Churn")

#Training oversampled model
set.seed(831) # initialize random seed

tuneR_us <- tuneRF(x = train_rf_us[, -13],
                   y = train_rf_us$Churn, 
                   ntreeTry = 500,
                   doBest = TRUE) 

# View basic model information
tuneR_us

# View variable importance for the tuned 
# model
varImpPlot(tuneR_us)

## Training Performance

tune.trRFpreds_us <- predict(object = tuneR_us, # tuned RF model
                             type = "class") # class predictions

RF_ttrain_conf_us <- confusionMatrix(data = tune.trRFpreds_us, # predictions
                                     reference = train_rf_us$Churn, # actual
                                     positive = "Yes",
                                     mode = "everything")
RF_ttrain_conf_us


## Testing Performance
tune.teRFpreds_us <- predict(object = tuneR_us, # tuned RF model
                             newdata = test_rf, # testing data
                             type = "class")

RF_ttest_conf_us <- confusionMatrix(data = tune.teRFpreds_us, # predictions
                                    reference = test_rf$Churn, # actual
                                    positive = "Yes",
                                    mode = "everything")
RF_ttest_conf_us

### Performance and Goodness of Fit
# Overall
cbind(Training = RF_ttrain_conf_us$overall,
      Testing = RF_ttest_conf_us$overall)
# By Class
cbind(Training = RF_ttrain_conf_us$byClass, 
      Testing = RF_ttest_conf_us$byClass)


#Comparing CI Models
#Training Performance
cbind(Base = RF_ttrain_conf$overall, # base model
      Under = RF_ttrain_conf_ds$overall, # undersampled model
      Over = RF_ttrain_conf_us$overall) # oversampled model
# By Class
cbind(Base = RF_ttrain_conf$byClass, # base model
      Under = RF_ttrain_conf_ds$byClass, # undersampled model
      Over = RF_ttrain_conf_us$byClass) # oversampled model

## Testing Performance
# Overall
cbind(Base = RF_ttest_conf$overall, # base model
      Under = RF_ttest_conf_ds$overall, # undersampled model
      Over = RF_ttest_conf_us$overall) # oversampled model
# By Class
cbind(Base = RF_ttest_conf$byClass, # base model
      Under = RF_ttest_conf_ds$byClass, # undersampled model
      Over = RF_ttest_conf_us$byClass) # oversampled model


#-------------------------------Comparison of ANN and Random Forest Models--------------------------

##### Base Models

### Training Performance
# Overall
cbind(ANN = tune_tr_conf_ann$overall,
      RF = RF_ttrain_conf$overall)
# By Class
cbind(ANN = tune_tr_conf_ann$byClass,
      RF = RF_ttrain_conf$byClass)


### Testing Performance
# Overall
cbind(ANN = tune_te_conf_ann$overall,
      RF = RF_ttest_conf$overall)
# By Class
cbind(ANN = tune_te_conf_ann$byClass,
      RF = RF_ttest_conf$byClass)


##### Undersampled Models

### Training Performance
# Overall
cbind(ANN_US = tune_tr_conf_ann_RUS$overall,
      RF_US = RF_ttrain_conf_ds$overall)
# By Class
cbind(ANN_US = tune_tr_conf_ann_RUS$byClass,
      RF_US = RF_ttrain_conf_ds$byClass)

### Testing Performance
# Overall
cbind(ANN_US = tune_te_conf_ann_RUS$overall,
      RF_US = RF_ttest_conf_ds$overall)
# By Class
cbind(ANN_US = tune_te_conf_ann_RUS$byClass,
      RF_US = RF_ttest_conf_ds$byClass)


##### Oversampled Models

### Training Performance
# Overall
cbind(ANN_OS = tune_tr_conf_ann_ROS$overall,
      RF_OS = RF_ttrain_conf_us$overall)
# By Class
cbind(ANN_OS = tune_tr_conf_ann_ROS$byClass,
      RF_OS = RF_ttrain_conf_us$byClass)

### Testing Performance
# Overall
cbind(ANN_OS = tune_te_conf_ann_ROS$overall,
      RF_OS = RF_ttest_conf_us$overall)
# By Class
cbind(ANN_OS = tune_te_conf_ann_ROS$byClass,
      RF_OS = RF_ttest_conf_us$byClass)

RF_ttest_conf_ds$table
tune_te_conf_ann_RUS$table












