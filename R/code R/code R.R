# Multiple Linear Regression

library(DAAG)
library(ggplot2)
library(dplyr)
library(car)
library(forecast)

c(1.45,1.93,0.81,0.61,1.55,0.95,0.45,1.14,0.74,0.98,1.41,0.81,0.89,0.68,1.39,1.53,0.91,1.49,1.38,1.73,1.11 , 1.68,0.66,0.69,1.98) -> y
c(0.58,0.86,0.29,0.20,0.56,0.28,0.08,0.41,0.22,0.35,0.59,0.22,0.26,0.12,0.65,0.70,0.30,0.70,0.39,0.72,0.45,0.81,0.04,0.20,0.95) -> x1
c(0.71,0.13,0.79,0.20,0.56,0.92,0.01,0.60,0.70,0.73,0.13,0.96,0.27,0.21,0.88,0.30,0.15,0.09,0.17,0.25,0.30,0.32,0.82,0.98,0.00) -> x2

data <- data.frame(y, x1, x2)
data

fit1 <- lm(data = data , y~.)
fit1

summary(fit1)

correlation_matrix <- cor(data)
correlation_matrix
cat('correlation x1 & x2 : ',correlation_matrix[2,3])

confint(fit1)

fit_value10 <- 0.433547 + (1.652993*data$x1) + (0.003945*data$x2)
fit_value11 <- fit1$fitted.values
fit1_value <- data.frame(fit_value10 , fit_value11)
fit1_value

avPlots(fit1)

fit2 <- lm(data = data , y~x1)
fit2

summary(fit2)

confint(fit2)

fit_value20 <-  0.43609 + (1.65121*data$x1)
fit_value21 <- fit2$fitted.values
fit2_vlue <- data.frame(fit_value20 , fit_value21)
fit2_vlue

resi <- fit2$residuals
fit_value <- fit2$fitted.values

data2 <- data.frame(resi , fit_value)
data2

data2 %>% ggplot(aes(sample = resi)) + stat_qq(color = 'black') + stat_qq_line(color = 'blue') + theme_test()

ks.test(data2$resi , 'pnorm' , mean(data2$resi) , sd(data2$resi))
shapiro.test(data2$resi)

data2 %>% ggplot(aes(x = fit_value , y = resi)) + geom_point(color = 'black') +
  theme_test() + labs(x = 'y^' , y = 'residual') + geom_hline(yintercept = c(0) , color = 'green') + 
  geom_hline(yintercept = c(-0.3,0.3) , color = 'blue')

resid <- ts(data2$resi)
ggAcf(resid , color = 'black') + theme_test()

durbinWatsonTest(data2$resi)

