# 로지스틱 회귀분석 ( 반응변수가 범주형 데이터의 경우에 사용되는 회귀분석법 )
# 종속변수 Y는 성공 및 실패의 두가지값을 가지고있음 
# 지도학습으로 분류되며 특정 결과의 분류 및 예측을 위해 사용됨

# 실습 대장암데이터 
install.packages("survival")
library(survival)
str(colon)

# 로지스틱 회귀분석 수행 
# status ( 사망 혹은 재발인경우 1 )
colon1 <- na.omit(colon) # 결측치 처리
result <- glm(status~sex+age+obstruct+adhere+nodes+differ+extent+surg,family = binomial , data=colon1)
summary(result)

# 유의미한 변수 선택 ( 후진 선택법에 따른 변수선택법 )
reduced.model <- step(result,direction="backward")
summary(reduced.model)

# 예측인자들의 odds ratio 구하기 
# digits = 소수점 / suppressmessages = 경고메시지를 출력하지 않게 하는 방법
ORtable <- function(x,digits=2){
  suppressMessages(a <- confint(x))
  result <- data.frame(exp(coef(x)),exp(a))
  result <- cbind(result,round(summary(x)$coefficient[,4],3))
  colnames(result) <- c("OR","2.5%","97.5","P")
  result
}
ORtable(reduced.model)

# odds ratio 시각화 
install.packages("moonBook")
library(moonBook)
odds_ratio <- ORtable(reduced.model)
odds_ratio <- odds_ratio[2:nrow(odds_ratio),]
HRplot(odds_ratio,type=2,show.CI=TRUE,cex=2)

# oddsratio > 1 인 경우 x가 증가하는 방향으로 목표변수에 영향을 미친다 
# oddsratio < 1 인 경우 x가 감소하는 방향으로 목표변수에 영향을 미친다 
# 목표변수의 범주가 3개 이상인 경우에는 로지스틱회귀모델을 이항 로지스틱모델의 확장
# 목표변수가 순서를 가질때는 프로빗연결함수 사용
