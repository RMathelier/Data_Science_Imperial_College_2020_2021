
##### Coursework final Data Science #####

setwd("C:/Users/robin/Dropbox/Applications/Overleaf/Coursework DS 6")

library(ggplot2)
library(corrplot)
library(dbplyr)
library(tidyverse)
library(randomForest)
library(xtable)
library(gridExtra)

my_theme =
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 15),
        legend.text = element_text(size = 12),
        strip.text = element_text(size = 14))

### load and process data ###

data_raw = read.csv("evals.csv")
for (col_name in colnames(data_raw)){
  if (class(data_raw[,col_name]) == "character") {
    data_raw[,col_name] = as.factor(data_raw[,col_name])
  }
}

p = dim(data_raw)[2]
n = dim(data_raw)[1]

### Delete variables bias ###

features_bias = c("age",
                  "gender",
                  "ethnicity",
                  "language",
                  "bty_f1lower", "bty_f1upper", "bty_f2upper", "bty_m1lower", "bty_m1upper", "bty_m2upper", "bty_avg",
                  "pic_outfit", "pic_color")

features_bias_other = c("age",
                        "gender",
                        "ethnicity",
                        "language",
                        "pic_outfit", "pic_color")

data = data_raw[, ! names(data_raw) %in% features_bias, drop = F]
data_bty = data_raw[, ! names(data_raw) %in% features_bias_other_bty, drop = F]
sum(is.na(data))  # No NAs

## split the examples into training and test

set.seed(42)
TEST_SIZE = 0.2

test_indices = sample(1:nrow(data), size=as.integer(TEST_SIZE*nrow(data)), replace=FALSE)
data_train = data[-test_indices,]
data_test = data[test_indices,]
data_train_bty = data_bty[-test_indices,]
data_test_bty = data_bty[test_indices,]
data_train_comp = data_raw[-test_indices,]
data_test_comp = data_raw[test_indices,]

x_train = data_train[,2:dim(data)[2]]
y_train = data_train[,1]
x_test = data_test[,2:dim(data)[2]]
y_test = data_test[,1]

x_train_bty = data_train_bty[,2:dim(data_bty)[2]]
y_train_bty = data_train_bty[,1]
x_test_bty = data_test_bty[,2:dim(data_bty)[2]]
y_test_bty = data_test_bty[,1]

summary(data_test_comp)

##### Data vizualisation #####

# histogram score

pdf("hist_score.pdf")
ggplot(data_train) +
  geom_bar(aes(x = score), fill = 'navyblue', alpha = 0.7) + 
  scale_x_continuous(name = "Average professor evaluation score", limits=c(1, 5)) +
  scale_y_continuous(name = "Number of observations") +
  my_theme
dev.off()

# Rank 

pdf("rank.pdf")
ggplot(data, aes(x = score, fill = rank)) +
  geom_density(adjust = 1, alpha = 0.5) +
  facet_grid(rank ~ .) +
  labs(fill = "rank",
       y = "Density")  +
  my_theme
dev.off()

######### Random forests ##########

mae = function(y_true, y_obs) mean(abs(y_true-y_obs)) 

set.seed(42)
data_forest_test = data

# grid over which we will perform the hyperparameter search:
hparam_grid = as.data.frame(expand.grid(mtry=seq(1, 7, by=1), maxnodes=seq(10, 100, by=10)))

# to store the OOB estimates of the MSE
oob_maes = rep(0.0, nrow(hparam_grid))

# perform the gridsearch
for(hparam_idx in 1:nrow(hparam_grid)) {
  # train candidate model
  this_mtry = hparam_grid[hparam_idx, 1]
  this_maxnodes = hparam_grid[hparam_idx, 2]
  rf = randomForest(x_train, 
                    y_train, 
                    mtry=this_mtry, 
                    maxnodes=this_maxnodes)
  
  # calculate OOB MAE
  oob_maes[hparam_idx] <- mae(y_train, predict(rf))
}

# select the best model (which has the minimum OOB MSE)
best_hparam_set <- hparam_grid[which.min(oob_maes),]

# train a model on the whole training set with the selected hyperparameters
rf_final <- randomForest(x_train, y_train,
                         mtry=best_hparam_set$mtry,
                         maxnodes=best_hparam_set$maxnodes,
                         importance=TRUE,
                         ntree = 1000)

# the test performance of the final model
yhat_test <- predict(rf_final, newdata=x_test)


# MAE
mae(yhat_test, y_test)

data_forest_test = data_raw[test_indices,]
data_forest_test$pred = yhat_test

## features importance ##

data.frame(rf_final$importance) %>% arrange(desc(IncNodePurity))

######### Biases random forest ##########

##### Gender #####

length(which(data_forest_test$gender=="male"))     # nb male in test set
length(which(data_forest_test$gender=="female"))   # nb female in test set

g_gender = ggplot(data_forest_test, aes(x = pred, fill = gender)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "gender",
       y = "Density")  +
  my_theme

mean(data_forest_test$pred[which(data_forest_test$gender=="male")])
mean(data_forest_test$pred[which(data_forest_test$gender=="female")])

ks.test(data_forest_test$pred[which(data_forest_test$gender=="male")],
        data_forest_test$pred[which(data_forest_test$gender=="female")])$p.value

##### Ethnicity #####

g_eth = ggplot(data_forest_test, aes(x = pred, fill = ethnicity)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "Ethnicity",
       y = "Density")  +
  my_theme

length(data_forest_test$pred[which(data_forest_test$ethnicity=="minority")])
length(data_forest_test$pred[which(data_forest_test$ethnicity=="not minority")])

ks.test(data_forest_test$pred[which(data_forest_test$ethnicity=="minority")],
        data_forest_test$pred[which(data_forest_test$ethnicity=="not minority")])$p.value


##### Language #####

length(which(data_forest_test$language=="english"))       # nb english in training set
length(which(data_forest_test$language=="non-english"))   # nb non english in training set

g_lang = ggplot(data_forest_test, aes(x = pred, fill = language)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "Language",
       y = "Density")  +
  my_theme

mean(data_forest_test$pred[which(data_forest_test$language=="english")])
mean(data_forest_test$pred[which(data_forest_test$language=="non-english")])

ks.test(data_forest_test$pred[which(data_forest_test$language=="english")],
        data_forest_test$pred[which(data_forest_test$language=="non-english")], exact = TRUE)$p.value

##### Picture ######

g_pic = ggplot(data_forest_test, aes(x = pred, fill = pic_outfit)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "pic_outfit",
       y = "Density")  +
  my_theme

##### plot #####

pdf("grid_bias.pdf", width = width/1.3, height = height/1.3)
grid.arrange(g_gender, g_eth, g_lang, g_pic, ncol = 2)
dev.off()

##### beauty #####

pdf("beauty_reg_RF.pdf")
ggplot(data_forest_test, aes(x = bty_avg, y = pred, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(formula = y ~ x, method = lm, se = FALSE) +
  labs(x = "Beauty average",
       y = "score",
       color = "Gender") +
  my_theme
dev.off()

data_rf_with_bty = data_forest_test[,c("bty_avg","pred", "gender")]
head(data_rf_with_bty)

mylm = lm(pred~bty_avg, data_forest_test)
summary(mylm)

data_bty_male = data_forest_test[which(data_forest_test$gender=="male"), c("bty_avg", "pred", "gender")]
mylm2 = lm(pred ~ bty_avg, data_bty_male)
summary(mylm2)

data_bty_female = data_forest_test[which(data_forest_test$gender=="female"),c("bty_avg","pred", "gender")]
mylm3 = lm(pred~bty_avg, data_bty_female)
summary(mylm3)


######## Random Forests with beauty features ##########

set.seed(42)

# grid over which we will perform the hyperparameter search:
hparam_grid = as.data.frame(expand.grid(mtry=seq(1, 13, by=1), maxnodes=seq(10, 200, by=10)))

# to store the OOB estimates of the MSE
oob_mses = rep(0.0, nrow(hparam_grid))

# perform the gridsearch
for(hparam_idx in 1:nrow(hparam_grid)) {
  # train candidate model
  this_mtry = hparam_grid[hparam_idx, 1]
  this_maxnodes = hparam_grid[hparam_idx, 2]
  rf_bty = randomForest(x_train_bty, 
                    y_train_bty, 
                    mtry=this_mtry, 
                    maxnodes=this_maxnodes)
  
  # calculate OOB MSE
  oob_mses[hparam_idx] <- mae(y_train_bty, predict(rf_bty))
}

# select the best model (which has the minimum OOB MSE)
best_hparam_set <- hparam_grid[which.min(oob_mses),]

# train a model on the whole training set with the selected hyperparameters
rf_final_bty <- randomForest(x_train_bty, y_train_bty,
                         mtry=best_hparam_set$mtry,
                         maxnodes=best_hparam_set$maxnodes,
                         importance=TRUE,
                         ntree = 2000)

# the test performance of the final model
yhat_test <- predict(rf_final_bty, newdata=x_test_bty)


# MAE
mae(yhat_test, y_test)

data_bty_forest_test = data_raw[test_indices,]
data_bty_forest_test$pred = yhat_test

## features importance ##

data.frame(rf_final_bty$importance) %>% arrange(desc(IncNodePurity))

## beauty ##

pdf("beauty_reg_RF_bty.pdf", width = 10, height = 5)
ggplot(data_bty_forest_test, aes(x = bty_avg, y = pred, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(formula = y ~ x, method = lm, se = FALSE) +
  labs(x = "Beauty average",
       y = "score",
       color = "Gender") +
  my_theme
dev.off()


data_rf_with_bty = data_bty_forest_test[,c("bty_avg","pred", "gender")]
head(data_rf_with_bty)

mylm = lm(pred~bty_avg,data_rf_with_bty)
summary(mylm)

data_bty_male = data_bty_forest_test[which(data_bty_forest_test$gender=="male"), c("bty_avg", "pred", "gender")]
mylm2 = lm(pred ~ bty_avg, data_bty_male)
summary(mylm2)

data_bty_female = data_bty_forest_test[which(data_bty_forest_test$gender=="female"),c("bty_avg","pred", "gender")]
mylm3 = lm(pred~bty_avg, data_bty_female)
summary(mylm3)


########### Biases study dataset #############


##### Gender #####

ggplot(data_train_comp, aes(x = score, fill = gender)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "gender",
       y = "Density")  +
  my_theme

ks.test(data_train_comp$score[which(data_train_comp$gender == "female")], 
        data_train_comp$score[which(data_train_comp$gender == "male")])$p.value

mean(data_train_comp$score[which(data_train_comp$gender == "female")])
mean(data_train_comp$score[which(data_train_comp$gender == "male")])

# Age vs Gender

ggplot(data_train_comp, aes(x = age, y = score, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = lm, se = FALSE) +
  labs(x = "Age",
       y = "Score",
       color = "Gender") +
  my_theme

# Gender vs cls_students

ggplot(data_train_comp, aes(x = cls_students, y = score, color = language)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = lm, se = FALSE) +
  labs(x = "Age",
       y = "Beauty",
       color = "Gender") +
  my_theme

##### Ethnicity #####

ggplot(data_train_comp, aes(x = score, fill = ethnicity)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "ethnicity",
       y = "Density")  +
  my_theme

ks.test(data_train_comp$score[which(data_train_comp$ethnicity == "minority")], 
        data_train_comp$score[which(data_train_comp$ethnicity == "not minority")])$p.value

##### Ethnicity + gender #####

data_train_comp$wom_min = "0"
data_train_comp$wom_min[which(data_train_comp$gender=="female" & data_raw$ethnicity == "minority")] = 1
data_train_comp$wom_min = as.factor(data_train_comp$wom_min)

ggplot(data_train_comp, aes(x = score, fill = wom_min)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "gender",
       y = "Density")  +
  my_theme

mean(data_forest_test$pred[which(data_forest_test$wom_min=="1")])
mean(data_forest_test$pred[which(data_forest_test$wom_min=="0")])

ks.test(data$score[which(data$ethnicity == "minority")], 
        data$score[which(data$ethnicity == "not minority")])$p.value

# Language

g = ggplot(data_train_comp, aes(x = score, fill = language)) +
  geom_density(adjust = 1, alpha = 0.5) +
  labs(fill = "language",
       y = "Density")  +
  my_theme
g

ks.test(data_train_comp$score[which(data_train_comp$language == "english")], 
        data_train_comp$score[which(data_train_comp$language == "non-english")])$p.value

# beauty

ggplot(data_train_comp, aes(x = bty_avg, y = score, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = lm, se = FALSE) +
  labs(x = "Beauty average",
       y = "score",
       color = "Gender") +
  my_theme

# beauty vs age #

pdf("beauty_age.pdf")
ggplot(data_train_comp, aes(x = age, y = bty_avg, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = lm, se = FALSE) +
  labs(x = "Age",
       y = "Beauty",
       color = "Gender") +
  my_theme
dev.off()

# Students agree on beauty ? #

# male vs female 

data_train_comp$bty_mavg = sapply(1:n, function(i) (1/3)*(data_train_comp$bty_m1lower[i] + data_train_comp$bty_m1upper[i] + data_train_comp$bty_m2upper[i]))
data_train_comp$bty_favg = sapply(1:n, function(i) (1/3)*(data_train_comp$bty_f1lower[i] + data_train_comp$bty_f1upper[i] + data_train_comp$bty_f2upper[i]))

pdf("marks_fem_mal.pdf")
ggplot(data_train_comp, aes(x = bty_mavg, y = bty_favg, color = gender)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = lm, se = FALSE) +
  labs(x = "beauty score from male students",
       y = "beauty score from female students",
       color = "Gender") +
  my_theme + geom_abline(intercept = 0, slope = 1, lty = 2)
dev.off()

# every student again each other

library(reshape2)

cormat = cor(data_raw[,13:18])
melted_cormat <- melt(cormat)
head(melted_cormat)

pdf("cormat.pdf")
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "white", high = "red", 
                       midpoint = 0.5, limit = c(0.5,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
dev.off()



##### Correlation between bias variables and predictors #####


keep_feat = c("cls_perc_eval", "age", "cls_students", "cls_did_eval",
              "cls_level, cls_profs, cls_credits", "rank",
              "gender", "language", "ethnicity","bty_avg",
              "score")

data_stud = data_raw[-test_indices, names(data_raw) %in% keep_feat, drop = F]

head(data_stud)









