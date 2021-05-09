library(Rfast)
library(caret)
library(VIM)
library(stringi)
library(stringr)
library(FSelector)
library(FSelectorRcpp)
library(imbalance)
library(e1071)
library(caret)
library(randomForest)
library(C50)
library(klaR)
library(kernlab)

##functions
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

############################Cleaning Training data################################

data <- read.csv("train.csv")

table(data$Is_sanctioned) <- as.factor(data$Is_sanctioned)

##changing columns to numerical

data$Gender[data$Gender == 'Male'] = "1"
data$Gender[data$Gender == 'Female'] = "2"
data$Gender <- as.factor(data$Gender)

data$Id_City[data$Id_City == 'A'] = "1"
data$Id_City[data$Id_City == 'B'] = "2"
data$Id_City[data$Id_City == 'C'] = "3"
data$Id_City <- as.factor(data$Id_City)

##calculating age from Cust_dob and Lead_Created_on(Age as on Lead_created)

data$Cust_dob<- as.POSIXct(data$Cust_dob,tz=Sys.timezone())
data$Lead_Created_On <- as.POSIXct(data$Lead_Created_On,tz=Sys.timezone())
data$age <- round(((data$Lead_Created_On - data$Cust_dob)/365), digits = 0)
data <- data[,c(1:5,23,6:22)]
data$age <- as.numeric(data$age)
##removing  Cust_dob and ead_Created_On
data$Cust_dob <- NULL
data$Lead_Created_On <- NULL

##Replacing blanks with NA 
data[data == ""] <- NA
##calculate mode for all the necessary columns
data$Loc_id[is.na(data$Loc_id)] <- getmode(data$Loc_id)
data$Id_City[is.na(data$Id_City)] <- getmode(data$Id_City)
##EMPLOYER ID
data$Employer_Id[data$Employer_Id == ""] <- NA
data$Employer_Id <- as.factor(data$Employer_Id)
data <- kNN(data, variable = c("Employer_Id"))
data$Employer_Id <- substr(data$Employer_Id,4,10)
data$Employer_Id <- as.numeric(data$Employer_Id)

###EMPLOYER_CT1
data$Employer_Ct1[is.na(data$Employer_Ct1)] <- getmode(data$Employer_Ct1)
data$Employer_Ct1[data$Employer_Ct1 == 'A'] = "1"
data$Employer_Ct1[data$Employer_Ct1 == 'B'] = "2"
data$Employer_Ct1[data$Employer_Ct1 == 'C'] = "3"
data$Employer_Ct1 <- as.factor(data$Employer_Ct1)

###EMPLOYER_CT2 - KNN
data$Employer_Ct2[is.na(data$Employer_Ct2)] <- getmode(data$Employer_Ct2)
data$Employer_Ct2 <- as.factor(data$Employer_Ct2)

##CUSTBANkCODE -KNN
data$CustBankCode <- stri_replace_all_charclass(data$CustBankCode, "[^[:alnum:]]", "")
data$CustBankCode[data$CustBankCode == "  "] <- NA
data$CustBankCode<- as.factor(data$CustBankCode)
data <- kNN(data, variable = c("CustBankCode"))
data$CustBankCode<- substr(data$CustBankCode,2,4)
data$CustBankCode <- as.numeric(data$CustBankCode)

#TypeBank - MODE

data$Type_Bank[is.na(data$Type_Bank)] <- getmode(data$Type_Bank)
data$Type_Bank[data$Type_Bank == 'P'] = "1"
data$Type_Bank[data$Type_Bank == 'G'] = "2"
data$Type_Bank <- as.factor(data$Type_Bank)

#ISCONTACTED 
data$IsContacted[data$IsContacted == 'Y'] = "1"
data$IsContacted[data$IsContacted == 'N'] = "2"
data$IsContacted <- as.factor(data$IsContacted)

#ORIGIN CATEGORY

data$Origin_Category[data$Origin_Category == 'A'] = "1"
data$Origin_Category[data$Origin_Category == 'B'] = "2"
data$Origin_Category[data$Origin_Category == 'C'] = "3"
data$Origin_Category[data$Origin_Category == 'D'] = "4"
data$Origin_Category[data$Origin_Category == 'E'] = "5"
data$Origin_Category[data$Origin_Category == 'F'] = "6"
data$Origin_Category[data$Origin_Category == 'G'] = "7"
data$Origin_Category <- as.factor(data$Origin_Category)

#Required amount - MEDIAN

data$Req_Amount[is.na(data$Req_Amount)] <- 30000


#Tenure

data$Tenure[is.na(data$Tenure)] <- getmode(data$Tenure)
data$Tenure <- as.factor(data$Tenure)
data <- kNN(data, variable = c("Tenure"))
data$Tenure <- as.integer(data$Tenure)

#Rate -Median

data$Rate[is.na(data$Rate)] <- 18


#Prev EMI

data$Prev_EMI[is.na(data$Prev_EMI)] <- 0

#curr EMI
data$Curr_EMI <- (data$Req_Amount)*(data$Rate/1200)*( (1+(data$Rate/1200))^(data$Tenure*12))/(((1+(data$Rate/1200))^(data$Tenure*12))-1)
data$Curr_EMI <- round(data$Curr_EMI,digits = 0)

##Lead origin
data$Lead_Origin<- substr(data$Lead_Origin,2,4)
data$Lead_Origin <- as.factor(data$Lead_Origin)

#
data$Is_sanctioned <- as.factor(data$Is_sanctioned)

data <- subset(data,select = S_No:Is_sanctioned)

#######################################################################################

data$Is_sanctioned <- as.factor(data$Is_sanctioned)

newdataset <- oversample(data, method = "ADASYN",classAttr = "Is_sanctioned")

information_gain(formula = Is_sanctioned ~ ., data = newdataset)



smp_floor <- floor(0.50*nrow(newdataset))
set.seed(123)
train_s <- sample(seq_len(nrow(newdataset)), size = smp_floor)
train<- newdataset[train_s,]
test <- newdataset[-train_s,]


control <- trainControl(method = "cv", summaryFunction = twoClassSummary, classProbs = T, savePredictions = T)
modelFit_RF <- train(Is_sanctioned~.,method ='rf', data = train, trcontrol = control)
pred_RF <- predict(modelFit_RF, test, type = 'raw')
CM_RF <- table(pred_RF, test$Is_sanctioned, dnn = c("Predicted","Actual"))
confusionMatrix(CM_RF)


##predicting values for the train dataset
##model_test.csv is the cleaned data set

model_test <- read.csv("Model_test.csv")

RF_test <- predict(modelFit_RF,model_test, type = 'raw')

write.csv(RF_test, "output_result.csv")




