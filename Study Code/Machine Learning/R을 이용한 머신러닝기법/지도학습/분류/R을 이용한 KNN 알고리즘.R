## KNN 알고리즘 
# KNN 수행 단계
# 1. 데이터 탐색과 수집 2. 데이터 정규화 3. 트레이닝 데이터 세트와 테스트 데이터 세트의 생성 4. 모델 트레이닝 5. 모델평가

# 1. 데이터 탐색과 수집 
iris
summary(iris)
install.packages("ggvis")
library(ggvis)
iris %>% ggvis(~Petal.Length,~Petal.Width, fill = ~factor(Species)) %>% layer_points()

# 2. 데이터 정규화 
min_max_normalizer <-  function(x){
  num <-  x - min(x)
  denom <- max(x) - min(x)
  return(num/denom)
}
normalized_iris <-  as.data.frame(lapply(iris[1:4], min_max_normalizer))
summary(normalized_iris)

table(iris$Species)
set.seed(1234)
random_samples <- sample(2,nrow(iris),replace=TRUE,prob=c(0.67,0.33))

# 3. 트레이닝 데이터 세트와 테스트 데이터 세트의 생성 
iris.train <- iris[random_samples == 1 , 1:4]
iris.trainLabels <- iris[random_samples == 1 , 5]
iris.test <-  iris[random_samples == 2 , 1:4]
iris.testLabels <- iris[random_samples == 2 , 5]
iris.testLabels

# 라이브러리 세팅
library(class)

# 4. 모델 트레이닝
iris_model <-  knn(train=iris.training,test=iris.test,cl=iris.trainLabels,k=3)
iris_model

# 5. 모델 평가 
install.packages("gmodels")
library(gmodels)

# 비교를 위한 교차표 준비 
CrossTable(x=iris.testLabels,y=iris_model,prop.chisq=FALSE)
