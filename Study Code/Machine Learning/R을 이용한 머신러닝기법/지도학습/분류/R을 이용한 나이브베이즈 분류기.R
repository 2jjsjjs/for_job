# 나이브 베이즈 분류기는 텍스트 분류를 위해 전통적으로 사용되는 분류기

colnames(iris)
levels(iris$Species) # 위의 독립 변수 4개를 사용하여 아이리스 종류를 예측하는것이 최종목표이다.

# 나이브 베이즈 모델 학습
install.packages("klaR")
library(klaR)
train <- sample(1:500,100) # 무작위로 100개 추출
nb <- NaiveBayes(Species~.,data=iris,subset=train) # 그룹 = train / data = iris
predict(nb,iris[-train,])$class # 예측분류

# 정분류 오분류 계산 
tt <- table(iris$Species[-train], predict(nb,iris[-train,])$class)
sum(tt[row(tt)==col(tt)])/sum(tt) # 정분류율
1 - sum(tt[row(tt)==col(tt)])/sum(tt) # 오오분류율

# 정오분류표 그래프화 (분류의 성능 판단)
library(ggplot2)
test <- iris[-train,]
test$pred <- predict(nb,iris[-train,])$class
ggplot(test,aes(Species,pred,color=Species))+
  geom_jitter(width=0.2,height=0.1,size=2)+
  labs(title="confusion matrix", subtitle="predicted vs observed from iris dataset",
       y = "predicted", x = "Truth")
