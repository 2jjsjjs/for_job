# CART 알고리즘(classification and regression Tree)
library(rpart)
rpartTree <- rpart(Species~., data=iris)
rpartTree
plot(rpartTree,margin=1)
text(rpartTree,cex=1)
predict(rpartTree,newdata=iris,type="class")
predicted <- predict(rpartTree,newdata=iris,type="class")
sum(predicted == iris$Species) / NROW(predicted)
real <- iris$Species
table(real,predicted)
# 문제점1 : 과적합 문제 ( 특정데이터에 대해서는 정확하지만 다른 데이터는 정확도가 떨어지는 현상 )
# 문제점2 : 다양한 값으로 분할 가능한 변수가 다른 변수에 비하여 선호되는 현상

# 과적합문제를 해결하기 위해서는 가지치기의 단계가 필요하다 (다른방법으로 하는)
install.packages("tree")
library(caret)
library(tree)
set.seed(1000)
intrain <- createDataPartition(y=iris$Species,p=0.7,list=FALSE)
train <- iris[intrain,]
test <- iris[-intrain,]
treemod <- tree(Species~.,data=iris)
plot(treemod)
text(treemod)
cv.trees <- cv.tree(treemod, FUN=prune.misclass)
plot(cv.trees)
prune.trees <- prune.misclass(treemod,best=6)
plot(prune.trees)
text(prune.trees,pretty=0)
library(e1071)
treepred <- predict(prune.trees, test, type='class')
confusionMatrix(treepred, test$Species)

# 조건부 추론 나무
install.packages("party")
library(party)
str(iris)
set.seed(1567)
num <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[num==1,]
head(trainData)
testData <- iris[num==2,]
head(testData)
myF <-Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width
ctreeResult <- ctree(myF,data=trainData)
table(predict(ctreeResult),trainData$Species)
forcasted <- predict(ctreeResult,data=testData)
table(forcasted,testData$Species)
plot(ctreeResult)

# 교차검증
data_cv <- cv.tree(treemod,K=10)
data_cv
