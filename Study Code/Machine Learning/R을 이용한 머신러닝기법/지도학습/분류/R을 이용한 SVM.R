library(ggplot2)
data("diamonds")
data1 <- diamonds
View(data1)

# TEST , TRAIN SET 만드는 과정 
set.seed(7054)
r <- sample(nrow(data1),nrow(data1)/3)
test_set <- data1[r,]
train_set <- data1[-r,]
nrow(test_set)
nrow(test_set) + nrow(train_set)
train_set$cut_status <- NA
train_set$cut_status[train_set$cut=='Fair'|train_set$cut=='Good'|train_set$cut=='Very Good'] = 0 
train_set$cut_status[train_set$cut=='Premium'|train_set$cut=='Ideal'] = 1
train_set$cut_status <- as.factor(train_set$cut_status)
class(train_set$cut_status)
table(train_set$cut_status)

test_set$cut_status <- NA
test_set$cut_status[test_set$cut=='Fair'|test_set$cut=='Good'|test_set$cut=='Very Good'] = 0 
test_set$cut_status[test_set$cut=='Premium'|test_set$cut=='Ideal'] = 1
test_set$cut_status <- as.factor(test_set$cut_status)
class(test_set$cut_status)
table(test_set$cut_status)

# 모델 제작 
library(e1071)
svm.model <- svm(cut_status~carat+depth+table+x+y+z,data=train_set)
svm.model

install.packages("Epi")
library(Epi)
tr_result <- predict(svm.model,train_set)
table(tr_result,train_set$cut_status)
te_result <- predict(svm.model,test_set)
table(te_result,test_set$cut_Status)
par(oma=c(1,1,1,1),mar=c(1,3,3,1))

# roc auc 곡선 그림 확인
ROC(test=tr_result,stat=train_set$cut_status,plot='ROC',AUC=T,main='Predict cut status (Train Set)')
ROC(test=te_result,stat=test_set$cut_status,plot='ROC',AUC=T,main='Predict cut status (Test Set)')

# 컴퓨터 사양의 문제에 따라서 test, train set의 비중을 줄이는 방법
set.seed(7054)
r <- sample(nrow(train_set),nrow(train_set)/3)
train_set <- train_set[r,]
set.seed(7054)
r <- sample(nrow(test_set),nrow(test_set)/3)
test_set <- test_set[r,]

# 최적화 과정( gamma , cost 값을 조정하는 과정 )
best_svm.model <- tune.svm(cut_status~carat+depth+table+x+y+z,data=train_set,gamma=c(0.5,0.1,0.01),cost=c(0.1,2.4))
best_svm.model
tr_result <- predict(best_svm.model$best.model,train_set)
table(tr_result,train_set$cut_status)
te_result <- predict(best_svm.model$best.model,test_set)
table(te_result,test_set$cut_status)
ROC(test=tr_result,stat=train_set$cut_status,plot='ROC',AUC=T,main='Predict cut status (Train Set)')
ROC(test=te_result,stat=test_set$cut_status,plot='ROC',AUC=T,main='Predict cut status (Test Set)')

