library(e1071)
library(ggplot2)
data("diamonds")
data1 <- diamonds
View(data1)

set.seed(7054)
r <- sample(nrow(data1),nrow(data1)/3)
test_set <- data1[r,]
train_set <- data1[-r,]

svr.model <- svm(price~carat+depth+table+x+y+z,data=train_set)
svr.model
# parameter는 cost , gamma , epsilon 

# 모델 적용
tr_result <- predict(object = svr.model , newdata = train_set)
te_result <- predict(object = svr.model , newdata = test_set)
tr_result[1:10]
te_result[1:10]

# 예측이 잘 됬는지 확인하기 위해서 MSE 값을 이용
cor(tr_result,train_set$price)
cor(te_result,test_set$price)
mean((tr_result-train_set$price)^2)  
mean((te_result-test_set$price)^2) 

# 최적화 방법 
best_svr.model <- tune.svm(price~carat+depth+table+x+y+z,data=train_set,gamma=c(0.1,0.5),cost=c(2,4),epsilon = c(0.1,1))
tr_result <- predict(object=best_svr.model$best.model,newdata=train_set)
te_result <- predict(object=best_svr.model$best.model,newdata=test_set)
cor(tr_result,train_set$price)
mean((tr_result-train_set$price)^2) # train set MSE
mean((te_result-test_set$price)^2) # test set MSE
