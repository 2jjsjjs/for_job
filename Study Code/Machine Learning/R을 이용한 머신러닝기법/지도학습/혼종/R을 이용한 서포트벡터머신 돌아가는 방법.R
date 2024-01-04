## SVM 모델 적용 1

# 데이터 전처리 
iris.subset <- subset(iris,select=c("Sepal.Length","Sepal.Width","Species"), Species %in% c("setosa","virginica"))
iris.subset
plot(x=iris.subset$Sepal.Length,y=iris.subset$Sepal.Width,col=iris.subset$Species ,pch = 19)

# svm 모델 적용
svm.model = svm(Species~.,data=iris.subset , kernel = "linear", cost=1 ,scale=F)
svm.model

# 서포트 벡터가 되는 지점 확인 
points(iris.subset[svm.model$index,c(1,2)],col="blue",cex=2)
points(svm.model$SV,col="blue",cex=3)

# 최대 초평면 마진 계산하기 
w = t(svm.model$coefs) %*% svm.model$SV
b = -svm.model$rho
abline(a=-b/w[1,2], b=-w[1,1]/w[1,2], col="red",lty=5)

## SVM 모델 적용 2 ( cost값을 올리니까 서포트 벡터머신의 값이 3개로 줄어듬 )
svm.model = svm(Species~.,data=iris.subset,type="C-classification",kernel='linear',cost=10000,scale=F)
svm.model
plot(x=iris.subset$Sepal.Length,y=iris.subset$Sepal.Width,col=iris.subset$Species,pch=19)
points(svm.model$SV,col="blue",cex=3)

# 최대 초평면 마진 계산하기 
w = t(svm.model$coefs) %*% svm.model$SV
b = -svm.model$rho
abline(a=-b/w[1,2], b=-w[1,1]/w[1,2], col="red",lty=5)

## Iris 전체 데이터에 svm 적용하기 
model.iris <- svm(Species~.,iris,scale=T)
model.iris
plot(model.iris,iris,Petal.Width ~ Petal.Length,slice=list(Sepal.Width=3,Sepal.Length=4))
