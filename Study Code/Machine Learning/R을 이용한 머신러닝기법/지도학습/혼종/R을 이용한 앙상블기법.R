## 1. Bagging : Bootstrap을 이용한 복원추출 / 데이터셋을 분할하고 반복 복원 추출하여 결과를 합치는 과정 자체 Bagging / 병렬학습
# Bagging의 장단점 :  학습데이터의 noise에 대해서 영향을 덜 받게되는 장점이 있다. 하지만, 동시에 Tree의 직관적인 이해력을 헤치게 되어서, 모형 해석에 어려움이 발생하게 된다.  

install.packages("adabag")
library(adabag)
data(iris)

iris.bagging <- bagging(Species~., data=iris, mfinal=10) # 10번 반복복원추출
iris.bagging$importance # 변수의 중요도
# 도식화 
plot(iris.bagging$trees[[10]])
text(iris.bagging$trees[[10]])

# 예측값
pred <- predict(iris.bagging, newdata=iris)

# 정오분류표
table(pred$class, iris[,5])

## 2. Boosting : boosting은 오분류된 데이터에 집중해 더 많은 가중치를 주는 ensemble 기법이다. 
# Boosting의 장단점 : 높은 정확도를 가지고 있지만 이상치에 취약한 모습을 보여준다. 
# AdaBoost :  각 라운드에서 안좋은 분류기에 대해서 가중치 부여가 이루어지는 것이다. 
# gradientboost : 목적함수의 오차를 최소화시키는 방향으로 진행하는 기법이다.
# XGboost : Gradient Boosting과는 달리 목적식에 Regularization term이 추가된 형태 --> gradient boost보다 과대적합이 해소된 모습을 볼 수 있다.
# AdaBoost 도식화 
boo.adabag <- boosting(Species~., data=iris, boos=TRUE, mfinal=10)
boo.adabag$importance
plot(boo.adabag$trees[[10]])
text(boo.adabag$trees[[10]])

# 예측값
pred <- predict(boo.adabag, newdata = iris)

# 정오분류표
tb <- table(pred$class,iris[,5])
tb

# Adaboost library
install.packages("ada")
library(ada)

# setosa 제외
data(iris)
iris[iris$Species != "setosa", ] -> iris
n <- dim(iris)[1]

traind <- sample(1:n, floor(.6*n), FALSE)
testd <- setdiff(1:n, traind)
iris[,5] <- as.factor((levels(iris[,5])[2:3])[as.numeric(iris[,5])-1])

# Ada boost 훈련용 데이터로 모형구축 
# test데이터에 대한 오분류율 0%
gdis <- ada(Species~., data=iris[traind,],iter=20, nu=1,type="discrete")
gdis <- addtest(gdis, iris[testd, -5], iris[testd, 5])
gdis

## XGBOOST : 숫자형 벡터에 대해서만 작동하기 때문에 ONE HOT ENCODING이 필요로 하다.
# One hot encoding : 범주형 변수를 숫자형 벡터로 변환하는 방법, 모든 가능한 값에 플래그를 사용하여 희소 행렬을 만드는 과정

# XGboost를 사용한 모델 구축
install.packages("xgboost")
library(xgboost)
data(bank,package="lightgbm")

# 변수 확인 및 데이터 확인
dim(bank)
head(bank)

# target변수(종속변수) 를 0과 1로 변환하기.
install.packages("car")
library(car)
dt_xgboost <- bank
dt_xgboost$y <- recode(dt_xgboost$y,"'no'=0;'yes'=1") # target변수를 0과 1로 변환 
str(dt_xgboost$y)
table(dt_xgboost$y)

install.packages("Matrix")
library(Matrix)

# sparse matrix 생성 
dt_xgb_sparse_matrix <- sparse.model.matrix(y~.-1,data=dt_xgboost)

# train data set sampling index 정의 
train_index <- sample(1:nrow(dt_xgb_sparse_matrix),2500)

# train 및 test data set 및 label data 생성 
train_x <- dt_xgb_sparse_matrix[train_index,]
test_x <-  dt_xgb_sparse_matrix[-train_index,]
train_y <- dt_xgboost[train_index,'y']
test_y <- dt_xgboost[-rain_index,'y']

dim(train)
dim(test)
dtrain <- xgb.DMatrix(data=train_x,label=as.matrix(train_y))
dtest <- xgb.DMatrix(data=test_x,label=as.matrix(test_y))

param <- list(max_depth=3, # 나무 깊이
              eta=0.1,
              verbose=0,
              nthread=2,
              objective="binary:logistic",
              eval_metric="auc")

xgb <- xgb.train(params=param,
                 data=dtrain,
                 nrounds=10,
                 subsample=0.5,
                 colsample_bytree=0.5,
                 nuym_class=1)

train_y_pred <- predict(xgb,dtrain)
test_y_pred <- predict(xgb,dtest)

install.packages("MLmetrics")
library(MLmetrics)

KS_Stat(train_y_pred,train_y)
KS_Stat(test_y_pred,test_y)

model <- xgb.dump(xgb,with_stats=T)
names <- dimnames(dtrain[[2]])
importance_matrix <- xgb.importance(names,model=xgb)
xgb.plot.importance(importance_matrix[1:10,])


