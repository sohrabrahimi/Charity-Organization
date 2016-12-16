charity <- read.csv("C:/Users/sur216/Box Sync/school stuff/Team A/charity.csv")
head(charity)
names(charity)
xmat = charity[,c(-0:-8,-22,-23,-24)]
pairs(xmat)

# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)
charity.t$incm <- log(charity.t$incm)
charity.t$inca <- log(charity$inca)

xmat2 = charity.t[,c(-0:-8,-22,-23,-24)]
pairs(xmat2) # the scatter plots look natural after these transformations

head(charity.t,10)
names(charity.t)
num_x = charity.t[!(charity.t$donr%in%c(0,NA)),]

head(num_x,10)
# add further transformations if desired
# for example, some statistical methods can struggle when predictors are highly skewed

# set up data for analysis

data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)



# Prediction modeling

# Least squares regression

model.ls1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.867523
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1696615

# drop wrat for illustrative purposes
model.ls2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

head(data.train.std.y)
head(data.valid.std.y)
head(data.test.std)

pred.valid.ls2 <- predict(model.ls2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls2)^2) # mean prediction error
# 1.867433
sd((y.valid - pred.valid.ls2)^2)/sqrt(n.valid.y) # std error
# 0.1696498

# Results

# MPE  Model
# 1.867523 LS1
# 1.867433 LS2

# select model.ls2 since it has minimum mean prediction error in the validation sample

yhat.test <- predict(model.ls2, newdata = data.test.std) # test predictions

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="~/Documents/teaching/psu/ip.csv", 
          row.names=FALSE) # use group member initials for file name

# submit the csv file in Angel for evaluation based on actual test donr and damt values



######### Variable selection ######## 
# Best Subsets
dim(data.train.std.y)
library(leaps)
regfit.best = regsubsets(damt ~ .+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data = data.train.std.y, nvmax = 27)
reg.summary = summary(regfit.full)
reg.summary$rss

test.mat = model.matrix(damt ~ .+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data = data.valid.std.y)
val.errors = rep(NA,27)
for (i in 1:27) {
  coefi = coef(regfit.best, id = i)
  pred = test.mat[, names(coefi)]%*%coefi
  val.errors[i] = mean((data.valid.std.y$damt- pred)^2)
}



plot(val.errors, ylab = "MSE", pch = 19, type = "b")
legend("topright", legend = c("Training"), col = c( "black"), 
       pch = 19)

val.errors
which.min(val.errors)# 19 is the best but too many. we select 11

train_val = rbind(data.train.std.y,data.valid.std.y) #combine the train and validation datasets
regfit.best.full = regsubsets(damt ~ .+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data = train_val, nvmax = 27)
coef(regfit.best.full,11) #optimom model
val.errors[11] # error for optimal model



# K-fold best subset 
head (train_val)
k=10
set.seed (1)
folds=sample (1:k,nrow(train_val),replace =TRUE)
cv.errors =matrix (NA ,k,27, dimnames =list(NULL , paste (1:27) ))

predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

for(j in 1:k){
   best.fit =regsubsets(damt~.+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif,data=train_val[folds !=j,],nvmax =27)
   for(i in 1:27) {
     pred=predict (best.fit ,train_val [folds ==j,], id=i)
     cv.errors [j,i]=mean( (train_val$damt[folds ==j]-pred)^2)
     }
}

mean.cv.errors =apply(cv.errors ,2, mean)
mean.cv.errors
par(mfrow =c(1,1))
plot(mean.cv.errors ,type="b") # still 11 is the best

regfit_pred = lm(damt~reg3+reg4+home+chld+hinc+incm+plow+npro+rgif+agif+inca:plow, data = train_val)

yhat_subset = predict(regfit_pred,data.test.std) #predicted vales for test
head(yhat_subset)

######### model shrinkage ######## 

# LASSO Regression
library(glmnet)
names(data.valid.std.y)
x_valid= model.matrix (damt~.*.,data.valid.std.y)
y_valid= data.valid.std.y$damt
x_train = model.matrix (damt~.*.,data.train.std.y)
y_train = data.train.std.y$damt
x_full = model.matrix (damt~.*.,train_val)
y_full = train_val$damt
x_test = model.matrix(~.*.,data.test.std )


grid =10^seq (10,-3, length =1000)
lasso.mod =glmnet (x_train,y_train,alpha =1, lambda =grid, standardize=FALSE, thresh =1e-12)

set.seed (1)
cv.out =cv.glmnet (x_train,y_train,alpha =1)
plot(cv.out)
bestlam =cv.out$lambda.min #best lambda
bestlam

lasso.pred=predict(lasso.mod ,s=bestlam ,newx=x_valid)
mean((lasso.pred-y_valid)^2)
lasso_full = glmnet(x_full,y_full,alpha =1)

plot(lasso_full)
yhat_lasso = predict (lasso_full ,s=bestlam, type = "response", newx = x_test) #yhats predecited with lasso
lasso_coef = predict (lasso_full ,s=bestlam, type = "coefficient", newx = x_test)
lasso_coef[lasso_coef!=0] # non-zero coefficients


# PCR Regression

library (pls)
set.seed (1)
pcr_fit=pcr(damt~.+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data=data.train.std.y ,scale=FALSE,validation ="CV")
summary (pcr_fit )
validationplot(pcr_fit,val.type="MSEP")

pcr_valid=predict (pcr_fit ,as.data.frame(x_valid) , ncomp =25)
mean((pcr_valid-as.data.frame(y_valid))^2)


pcr_full=pcr(damt~.+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data=train_val,scale=FALSE,validation ="CV")
yhat_pcr=predict (pcr_full ,data.test.std , ncomp =14)

#PLS Regression

set.seed (1)
plsr_fit=plsr(damt~.+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data=data.train.std.y ,scale=FALSE,validation ="CV")
summary (plsr_fit )
validationplot(plsr_fit,val.type="MSEP")

plsr_valid=predict (plsr_fit ,as.data.frame(x_valid) , ncomp =5)
mean((plsr_valid-as.data.frame(y_valid))^2)


plsr_full=plsr(damt~.+avhv:incm+avhv:inca+avhv:plow+incm:inca+incm:plow+inca:plow+lgif:tgif, data=train_val,scale=FALSE,validation ="CV")
yhat_plsr=predict (plsr_full ,data.test.std , ncomp =5)

######### non-linear models ######## 
# GAM
# we will use all the variables resulted from the shrinked model
coef(regfit_pred)
library (splines )
library(gam)
par(mfrow=c(2,3),oma = c(0, 0, 3, 0))
plot(data.train.std.y$rgif,data.train.std.y$damt, ylab="damt", xlab = "rgif")
plot(data.train.std.y$agif,data.train.std.y$damt, ylab="damt", xlab ="agif")
plot(data.train.std.y$plow,data.train.std.y$damt, ylab="damt", xlab ="plow")
plot(data.train.std.y$npro,data.train.std.y$damt, ylab="damt", xlab ="npro")
plot(data.train.std.y$inca,data.train.std.y$damt, ylab="damt", xlab ="inca")
plot(data.train.std.y$incm,data.train.std.y$damt, ylab="damt", xlab ="incm")
mtext("Scatter plots for training set", outer = TRUE, cex = 1.5)


gam0 = glm(damt~reg3+reg4+home+chld+incm+npro+rgif+agif+plow:inca+hinc+plow,data=data.train.std.y)
gam1=glm(damt~reg3+reg4+home+chld+incm+npro+rgif+agif+plow:inca+ns(hinc ,3)+ns(plow ,5),data=data.train.std.y)
gam2=glm(damt~reg3+reg4+home+chld+incm+npro+rgif+agif+plow:inca+s(hinc ,3)+s(plow ,5),data=data.train.std.y)
gam3=glm(damt~reg3+reg4+home+chld+ns(incm,4)+npro+rgif+ns(agif,3)+plow:inca+ns(hinc ,3)+ns(plow ,5),data=data.train.std.y)

gam0_pred<- predict(gam0, newdata = data.valid.std.y)
gam1_pred<- predict(gam1, newdata = data.valid.std.y) # validation predictions
gam2_pred<- predict(gam2, newdata = data.valid.std.y) # validation predictions
gam3_pred<- predict(gam3, newdata = data.valid.std.y)
anova(gam0,gam1,gam2,gam3, test = "F") # there is enough evidence that gam3 is the full model (gam3) is the superior model

mean((y.valid - gam0_pred)^2) # mean prediction error 0
mean((y.valid - gam1_pred)^2) # mean prediction error 1
mean((y.valid - gam2_pred)^2) # mean prediction error 2
mean((y.valid - gam3_pred)^2) # mean prediction error 3
par(mfrow=c(2,2))
plot(gam3, se=TRUE ,col ="blue ")

yhat_gam3 = predict(gam3,newdata =data.test.std)


######### regression tree ######## 
library(tree)
tree_fit = tree(damt~. , data = data.train.std.y)
plot(tree_fit)
text(tree_fit,pretty = 0)
summary(tree_fit)
tree_cv = cv.tree(tree_fit)
plot(tree_cv$size ,tree_cv$dev ,type="b")
prune_tree=prune.tree(tree_fit,best =7)
plot(prune_tree)
text(prune_tree,pretty = 0)

yhat_tree=predict (tree_fit,newdata =data.valid.std.y)
tree_valid=data.valid.std.y[,21]
plot(yhat_tree,tree_valid, pch = 19)
abline (0,1)
mean((yhat_tree -tree_valid)^2)

yhat_tree_final=predict (tree_fit,newdata =data.test.std)





