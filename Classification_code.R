############################################################################
###############     Applied Data Mining, Fall 2016##########################
###########Team Project: Sohrab Rahimi and Miriam Kerstein##################
############################################################################

#Libraries
library(MASS)
library(class)
library(ISLR)
library(tree)
library(e1071)

# Read the data
charity = read.csv("C:/Users/mirik/Desktop/R Final Project/charity.csv")

# Predictor transformations
charity.t = charity
charity.t$avhv <- log(charity.t$avhv)
charity.t$incm <- log(charity.t$incm)
charity.t$inca <- log(charity.t$inca)

# Set up data for analysis
data.train = charity.t[charity$part=="train",]
x.train = data.train[,2:21]
c.train = data.train[,22] # donr
n.train.= length(c.train) # 3984
y.train = data.train[c.train==1,23] # damt for observations with donr=1
n.train.y = length(y.train) # 1995

data.valid = charity.t[charity$part=="valid",]
x.valid = data.valid[,2:21]
c.valid = data.valid[,22] # donr
n.valid.c = length(c.valid) # 2018
y.valid = data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y = length(y.valid) # 999

data.test = charity.t[charity$part=="test",]
n.test = dim(data.test)[1] # 2007
x.test = data.test[,2:21]

x.train.mean = apply(x.train, 2, mean)
x.train.sd = apply(x.train, 2, sd)
x.train.std = t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c = data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y = data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std = t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c = data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y = data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std = t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std = data.frame(x.test.std)

train = data.train.std.c
valid = data.valid.std.c

####################################################################################################
#####                                SOME USEFUL UTILITIES                                  ########
####################################################################################################

#Given a list of posterior probabilites, returns the cutoff by maximizing profit
cutoff = function(probs, c.valid){
  profit = cumsum(14.5*c.valid[order(probs, decreasing=T)]-2)
  number.mailings = which.max(profit) # number of mailings that maximizes profits
  cutoff.prob = sort(probs, decreasing=T)[number.mailings+1]
  return (cutoff.prob)
}

#Given a list of posterior probabilites, applies the classification using the cutoff and returns the profit derived
profit = function(probs, c.valid){
  cut = cutoff(probs, c.valid)
  chat = ifelse(probs>cut, 1, 0) # mail to everyone above the cutoff
  table = table(chat, c.valid) # classification table
  total.mailings = table["1","0"]+table["1","1"]
  correct.mailings = table["1","1"]
  income = 14.5*correct.mailings
  cost = 2*total.mailings
  prof = income-cost
  return(prof)
}

#Given a table, return the profit
table.profit = function(table){
  total.mailings = table["1","0"]+table["1","1"]
  correct.mailings = table["1","1"]
  income = 14.5*correct.mailings
  cost = 2*total.mailings
  prof = income-cost
  return(prof)
}




####################################################################################################
#####                         LoGISTIC REGRESSION CLASSIFICATION                            ########
####################################################################################################

#First, fit an logistic regression model using all the predictors, and no interaction terms 
model.log = glm(donr~., data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
# 11402.5

#Now, inspect all p values and see if any predictors can be eliminated
summary(model.log)
#The p-values suggest eliminating reg3, reg4, genf, inca, plow, lgif, rgif, and agif

#Fit another logistic regression model, using only significant predictors
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + wrat + avhv + incm + npro + tgif + tdon + tlag, data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11410.5

#Now, try adding a quadratic term for hinc 
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + wrat + avhv + incm + npro + tgif + tdon + tlag, data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11651.5


#That improved profits significantly! 

#Try some interaction terms
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + wrat + avhv + incm + npro + tgif + tdon + tlag + avhv:incm + avhv:inca + avhv:plow + incm:inca +inca:plow, data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11635.5 
#Not an improvement!

#Now, try adding a quadratic term for wrat
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + I(wrat^2)+ wrat + avhv + incm + npro + tgif + tdon + tlag, data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11672

#Now, try adding a quadratic term for chld
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + I(wrat^2)+ wrat + avhv + incm +I(chld^2) + npro + tgif + tdon + tlag, data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11687

#Now, try adding a quadratic term for other predictors

#######BEST MODEL FROM ALL MODELS ATTEMPTED#############################
model.log = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + I(wrat^2)+ wrat + avhv +I(avhv^2)+ incm +I(chld^2) + npro +I(npro^2)+ tgif +I(tgif^2)+ tdon +I(tdon^2)+ tlag +I(tlag^2), data = train, family = "binomial")
probs.log = predict(model.log, valid, type = "response")
profit.log = profit(probs.log, c.valid)
profit.log
#11772.5 - IMPORTANT QUESTION TO CONSIDER: IS THIS OVERFITTING? Note that it is achieving higher profits on the validation data, not just on training.

model.best = glm(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + I(wrat^2)+ wrat + avhv +I(avhv^2)+ incm +I(chld^2) + npro +I(npro^2)+ tgif +I(tgif^2)+ tdon +I(tdon^2)+ tlag +I(tlag^2), data = train, family = "binomial")

####################################################################################################
#####                                   LDA CLASSIFICATION                                  ########
####################################################################################################

#Significant predictors, with quadratic term
model.lda = lda(donr~ reg1 + reg2 +home + chld+ hinc + I(hinc^2) + wrat + avhv + incm + npro + tgif + tdon + tlag, data = train)
probs.lda = predict(model.lda, valid)$posterior[,2]
profit.lda = profit(probs.lda, c.valid)
profit.lda
#11615

#All predictors, with quadratic term
model.lda = lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data=train)
probs.lda = predict(model.lda, valid)$posterior[,2]
profit.lda = profit(probs.lda, c.valid)
profit.lda
#11627.5

#All predictors, without quadratic term
model.lda = lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data=train)
probs.lda = predict(model.lda, valid)$posterior[,2]
profit.lda = profit(probs.lda, c.valid)
profit.lda
#11350.5

#Logistic regression performs better than LDA

####################################################################################################
#####                                   QDA CLASSIFICATION                                  ########
####################################################################################################

#With all predictors, and no quadratic term
model.qda = qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data=train)
probs.qda = predict(model.qda, valid)$posterior[,2]
profit.qda = profit(probs.qda, c.valid)
profit.qda
#11266

#With all predictors, and quadratic term
model.qda = qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data=train)
probs.qda = predict(model.qda, valid)$posterior[,2]
profit.qda = profit(probs.qda, c.valid)
profit.qda
#11225.5

#Note that with QDA, performance degenerates when you include the quadratic term.
#This is likely because QDA inherently assumes the quadratic form, and so the addition of the term obscures the relationship

#Performance is noticeably poorer than LDA and Logistic regression. This suggests that the linear form
#assumed by logistic regression and LDA capture the true relationship more accurately than the 
#quadratic form assumed by QDA

####################################################################################################
#####                                   KNN CLASSIFICATION                                  ########
####################################################################################################

#First determine which terms to include, using k=3. Afterwards, we will find optimal value for k

#All predictors 
train.x = train[,-21]
train.y = train$donr
test.x = valid[,-21]
test.y = valid$donr


knn.fit = knn(train=train.x, test=test.x, cl=train.y, k=3, prob=T)
knn.pred.class = c(knn.fit)-1
knn.pred.prob = attr(knn.fit, "prob") #This is the probability of the winning class
knn.prob = knn.pred.class*knn.pred.prob+(1-knn.pred.class)*(1-knn.pred.prob) # n.train post probs of Y=1
profit.knn = profit(knn.prob, test.y)
profit.knn
#11172

#Only significant predictors
train.x = cbind(train$reg1, train$reg2,  train$home, train$chld, train$hinc,  train$wrat, train$avhv, train$incm,  train$npro, train$tgif,  train$tdon, train$tlag)
test.x = cbind(valid$reg1, valid$reg2,  valid$home, valid$chld, valid$hinc,  valid$wrat, valid$avhv, valid$incm,  valid$npro, valid$tgif,  valid$tdon, valid$tlag)
knn.fit = knn(train=train.x, test=test.x, cl=train.y, k=3, prob=T)
knn.pred.class = c(knn.fit)-1
knn.pred.prob = attr(knn.fit, "prob") #This is the probability of the winning class
knn.prob = knn.pred.class*knn.pred.prob+(1-knn.pred.class)*(1-knn.pred.prob) # n.train post probs of Y=1
profit.knn = profit(knn.prob, test.y)
profit.knn
#11045.5

#Using only significant predictors does not improve results. So, let's find optimal k using all predictors
train.x = train[,-21]
test.x = valid[,-21]
#Loop to find best value...
mer = rep(NA, 30)
print(mer)
set.seed(2014)
for (i in 1:30) mer[i] = sum((train.y-(c(knn.cv(train=train.x, cl=train.y, k=i))-1))^2)/3985
plot(mer)
which.min(mer)
#Optimal k is 12
knn.fit = knn(train=train.x, test=test.x, cl=train.y, k=12, prob=T)
knn.pred.class = c(knn.fit)-1
knn.pred.prob = attr(knn.fit, "prob") #This is the probability of the winning class
knn.prob = knn.pred.class*knn.pred.prob+(1-knn.pred.class)*(1-knn.pred.prob) # n.train post probs of Y=1
profit.knn = profit(knn.prob, test.y)
profit.knn
#11292

#We see that even the best performing KNN classifier, still performs more poorly than our logistic regresison model


####################################################################################################
#####                               CLASSIFICATION TREES                                    ########
####################################################################################################

#Unpruned tree
bin.donor = ifelse(train$donr>=1, "Yes", "No")
train.tree = cbind(train, bin.donor)
tree.donors = tree(bin.donor~.-donr, train.tree)
summary(tree.donors)
plot(tree.donors)
text(tree.donors, pretty=0)
tree.donors
tree.pred = predict(tree.donors, valid, type = "class")
table(tree.pred, c.valid)
          #c.valid
#tree.pred   0   1
#       No  783  70
#       Yes 236 929
totalMailed = 236 + 929
trueDonors = 929
cost = totalMailed * 2
income = trueDonors * 14.5
profit = income-cost
profit


#Pruned tree
set.seed(3)
cv.donors = cv.tree(tree.donors, FUN=prune.misclass)
cv.donors
#Lowest error is with size of one and two!
prune.donor = prune.misclass(tree.donors, best = 2)
summary(prune.donor)
plot(prune.donor)
text(prune.donor, pretty=0)
prune.donor
tree.pred = predict(prune.donor, valid, type = "class")
table(tree.pred, c.valid)
#         c.valid
#tree.pred   0   1
#       No  922 403
#       Yes  97 596
totalMailed = 97 + 596
trueDonors = 596
cost = totalMailed * 2
income = trueDonors * 14.5
profit = income-cost
profit

#The pruned tree does really poorly as compared to our other classification methods!
#Even the unpruned tree, though, does not perform as well as the other methods.

####################################################################################################
#####                               SUPPORT VECTOR MACHINES                                 ########
####################################################################################################
train$donr = as.factor(train$donr)

#Linear Kernel
svm.linear = svm(donr~., data = train, kernel = "linear", cost = 0.01, scale = FALSE)
summary(svm.linear)
ypred=predict(svm.linear,valid)
table = table(predict = ypred, truth = c.valid)
prof = table.profit(table)
prof
#10437.5


#Radial Kernel
svm.radial = svm(donr~., data = train, kernel = "radial", cost = 0.01, scale = FALSE)
summary(svm.radial)
ypred=predict(svm.radial,valid)
table = table(predict = ypred, truth = c.valid)
prof = table.profit(table)
prof
#9246

#Polynomial Kernel
svm.poly = svm(donr~., data = train, kernel = "polynomial", cost = 0.01, scale = FALSE)
summary(svm.poly)
ypred=predict(svm.poly,valid)
table = table(predict = ypred, truth = c.valid)
prof = table.profit(table)
prof
#11159.5

#The SVM does not perform as well as the Logistic Regression



####################################################################################################
#####                           FINAL CHAT AND WRITE TO FILE                                ########
####################################################################################################
post.test <- predict(model.best, data.test.std, type="response") # post probs for test data
# Oversampling adjustment for calculating number of mailings for test set
post.valid.log1 <- predict(model.best, data.valid.std.c, type="response") # n.valid post probs
profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log1)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set
cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#chat.test
#0    1 
#1680  327
#Based on this, we will mail to 327 of the test observations

length(chat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s

cht <- data.frame(chat=chat.test)    # chat.test is your classification predictions (2007 0 and 1 values)
write.csv(cht, file="C:/Users/mirik/Desktop/R Final Project/chat.csv", row.names=FALSE) 

