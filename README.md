ST558, Project3
================
Nataliya Peshekhodko
2023-10-29

- <a href="#01-introduction" id="toc-01-introduction">0.1 Introduction</a>
- <a href="#02-packages" id="toc-02-packages">0.2 Packages</a>
- <a href="#03-data" id="toc-03-data">0.3 Data</a>
- <a href="#04-explanatory-data-analysiseda"
  id="toc-04-explanatory-data-analysiseda">0.4 Explanatory Data
  Analysis(EDA)</a>
- <a href="#05-modeling" id="toc-05-modeling">0.5 Modeling</a>
  - <a href="#051-log-loss" id="toc-051-log-loss">0.5.1 Log loss</a>
  - <a href="#052-logistic-regression"
    id="toc-052-logistic-regression">0.5.2 Logistic regression</a>

## 0.1 Introduction

## 0.2 Packages

``` r
library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(caret)
library(Metrics)
```

## 0.3 Data

``` r
data = read_csv('./data/diabetes_binary_health_indicators_BRFSS2015.csv')
```

Checking for NA values

``` r
sum(is.na(data))
```

    ## [1] 0

``` r
head(data)
```

    ## # A tibble: 6 × 22
    ##   Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies
    ##             <dbl>  <dbl>    <dbl>     <dbl> <dbl>  <dbl>  <dbl>                <dbl>        <dbl>  <dbl>   <dbl>
    ## 1               0      1        1         1    40      1      0                    0            0      0       1
    ## 2               0      0        0         0    25      1      0                    0            1      0       0
    ## 3               0      1        1         1    28      0      0                    0            0      1       0
    ## 4               0      1        0         1    27      0      0                    0            1      1       1
    ## 5               0      1        1         1    24      0      0                    0            1      1       1
    ## 6               0      1        1         1    25      1      0                    0            1      1       1
    ## # ℹ 11 more variables: HvyAlcoholConsump <dbl>, AnyHealthcare <dbl>, NoDocbcCost <dbl>, GenHlth <dbl>, MentHlth <dbl>,
    ## #   PhysHlth <dbl>, DiffWalk <dbl>, Sex <dbl>, Age <dbl>, Education <dbl>, Income <dbl>

Combine Education levels `1` and `2` into one level `12`

``` r
transformed <- data %>%
  mutate (Education = if_else(Education == 1 | Education == 2, 12, Education))
```

Sub-setting data for the selected education level:

``` r
education_level=12

subset <- transformed %>%
  filter(Education == education_level)
```

Checking data structure:

``` r
str(subset)
```

    ## tibble [4,217 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : num [1:4217] 0 1 1 1 0 0 1 0 1 0 ...
    ##  $ HighBP              : num [1:4217] 1 1 0 1 1 1 1 1 1 0 ...
    ##  $ HighChol            : num [1:4217] 1 1 1 1 0 1 1 1 0 1 ...
    ##  $ CholCheck           : num [1:4217] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ BMI                 : num [1:4217] 38 28 32 25 35 45 25 37 30 36 ...
    ##  $ Smoker              : num [1:4217] 1 1 0 1 1 1 1 1 0 0 ...
    ##  $ Stroke              : num [1:4217] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ HeartDiseaseorAttack: num [1:4217] 0 1 1 1 0 1 0 0 0 0 ...
    ##  $ PhysActivity        : num [1:4217] 0 0 1 0 1 1 0 1 1 0 ...
    ##  $ Fruits              : num [1:4217] 1 0 0 1 1 1 0 1 0 0 ...
    ##  $ Veggies             : num [1:4217] 1 1 0 1 1 1 0 1 0 1 ...
    ##  $ HvyAlcoholConsump   : num [1:4217] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ AnyHealthcare       : num [1:4217] 1 1 1 1 1 1 1 1 1 0 ...
    ##  $ NoDocbcCost         : num [1:4217] 0 0 0 0 0 1 0 0 0 1 ...
    ##  $ GenHlth             : num [1:4217] 5 4 1 5 4 5 4 3 3 4 ...
    ##  $ MentHlth            : num [1:4217] 15 0 0 15 0 5 3 0 0 0 ...
    ##  $ PhysHlth            : num [1:4217] 30 0 0 30 1 10 6 0 7 30 ...
    ##  $ DiffWalk            : num [1:4217] 1 0 1 1 1 1 1 0 1 0 ...
    ##  $ Sex                 : num [1:4217] 0 1 0 0 1 0 1 0 0 1 ...
    ##  $ Age                 : num [1:4217] 13 12 13 9 9 7 12 11 10 5 ...
    ##  $ Education           : num [1:4217] 12 12 12 12 12 12 12 12 12 12 ...
    ##  $ Income              : num [1:4217] 3 4 2 3 5 2 2 2 3 3 ...

Variables in data set:

- **Diabetes_binary** - 0 = no diabetes, 1 = diabetes
- **HighBP** - 0 = no high blood pressure, 1 = high blood pressure
- **HighChol** - 0 = no high cholesterol, 1 = high cholesterol
- **CholCheck** - 0 = no cholesterol check in 5 years, 1 = yes
  cholesterol check in 5 years
- **BMI** - Body Mass Index
- **Smoker** - Have you smoked at least 100 cigarettes in your entire
  life? $$Note: 5 packs = 100 cigarettes$$ 0 = no, 1 = yes
- **Stroke** - (Ever told) you had a stroke. 0 = no, 1 = yes
- **HeartDiseaseorAttack** - Coronary heart disease (CHD) or myocardial
  infarction (MI), 0 = no, 1 = yes
- **PhysActivity** - Physical activity in past 30 days - not including
  job, 0 = no, 1 = yes
- **Fruits** - Consume Fruit 1 or more times per day, 0 = no, 1 = yes
- **Veggies** - Consume Vegetables 1 or more times per day, 0 = no 1 =
  yes
- **HvyAlcoholConsump** - Heavy drinkers (adult men having more than 14
  drinks per week and adult women having more than 7 drinks per week) 0
  = no
- **AnyHealthcare** - Have any kind of health care coverage, including
  health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes
- **NoDocbcCost** - Was there a time in the past 12 months when you
  needed to see a doctor but could not because of cost? 0 = no 1 = yes
- **GenHlth** - Would you say that in general your health is: scale 1-5
  1 = excellent 2 = very good 3 = good 4 = fair 5 = poor
- **MentHlth** - Now thinking about your mental health, which includes
  stress, depression, and problems with emotions, for how
- **PhysHlth** - Now thinking about your physical health, which includes
  physical illness and injury, for how many days during the past 30
- **DiffWalk** - Do you have serious difficulty walking or climbing
  stairs? 0 = no, 1 = yes
- **Sex** - 0 = female, 1 = male
- **Age** - 13-level age category, 1 = 18-24, 9 = 60-64, 13 = 80 or
  older
- **Education** - Education level scale 1-6, 1 = Never attended school
  or only kindergarten, 2 = Grades 1 through 8
- **Income** - Income scale scale 1-8, 1 = less than 10,000 dol, 5 =
  less than 35,000 dol, 8 = 75,000 dol or more

## 0.4 Explanatory Data Analysis(EDA)

``` r
table(subset$Diabetes_binary)
```

    ## 
    ##    0    1 
    ## 2987 1230

``` r
ggplot(data = subset, aes(x = Age)) +
  geom_histogram(color = "black", fill = 'brown') +
  labs(title = "Histogram of Age groups distribution", x = "Age group", y = "Frequency")
```

![](README_files/figure-gfm/unnamed-chunk-183-1.png)<!-- -->

``` r
table(subset$Diabetes_binary, subset$Age)
```

    ##    
    ##       1   2   3   4   5   6   7   8   9  10  11  12  13
    ##   0  25  48 104 157 200 203 308 273 291 320 315 285 458
    ##   1   1   4   9  14  21  64  85 128 163 203 193 161 184

``` r
table(factor (subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$Sex, labels = c("Female", "Male")))
```

    ##            
    ##             Female Male
    ##   No diabet   1576 1411
    ##   Diabet       707  523

``` r
corrplot(cor(as.matrix(subset %>% select(-Education))), 
         type="upper", 
         tl.pos = "lt")
```

![](README_files/figure-gfm/unnamed-chunk-186-1.png)<!-- -->

``` r
table(subset$Diabetes_binary, subset$GenHlth)
```

    ##    
    ##        1    2    3    4    5
    ##   0  243  399 1010  924  411
    ##   1   32   62  255  510  371

``` r
table(subset$Diabetes_binary, factor(subset$HighBP, labels = c("No high BP", "High BP")) )
```

    ##    
    ##     No high BP High BP
    ##   0       1472    1515
    ##   1        268     962

``` r
table(subset$Diabetes_binary, factor (subset$HighChol, labels = c("No high chol", "High chol")))
```

    ##    
    ##     No high chol High chol
    ##   0         1611      1376
    ##   1          354       876

``` r
ggplot(subset, aes(x = as_factor(Diabetes_binary), y = BMI, fill = as_factor(Diabetes_binary))) +
  geom_boxplot() +
  labs(title = "BMI distribution for patients with and without diabetes", x = "Diabetes", y = "BMI")
```

![](README_files/figure-gfm/unnamed-chunk-190-1.png)<!-- -->

## 0.5 Modeling

Converting some of the variables to factors.

``` r
names = c('HighBP' ,'HighChol', 
          'CholCheck', 'Smoker', 
          'Diabetes_binary', 'Stroke',
          'HeartDiseaseorAttack', 'PhysActivity',
          'Fruits', 'Veggies', 
          'HvyAlcoholConsump', 'Sex',
          'Age','Income')
subset[,names] = lapply(subset[,names] , factor)
str(subset)
```

    ## tibble [4,217 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : Factor w/ 2 levels "0","1": 1 2 2 2 1 1 2 1 2 1 ...
    ##  $ HighBP              : Factor w/ 2 levels "0","1": 2 2 1 2 2 2 2 2 2 1 ...
    ##  $ HighChol            : Factor w/ 2 levels "0","1": 2 2 2 2 1 2 2 2 1 2 ...
    ##  $ CholCheck           : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ BMI                 : num [1:4217] 38 28 32 25 35 45 25 37 30 36 ...
    ##  $ Smoker              : Factor w/ 2 levels "0","1": 2 2 1 2 2 2 2 2 1 1 ...
    ##  $ Stroke              : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ HeartDiseaseorAttack: Factor w/ 2 levels "0","1": 1 2 2 2 1 2 1 1 1 1 ...
    ##  $ PhysActivity        : Factor w/ 2 levels "0","1": 1 1 2 1 2 2 1 2 2 1 ...
    ##  $ Fruits              : Factor w/ 2 levels "0","1": 2 1 1 2 2 2 1 2 1 1 ...
    ##  $ Veggies             : Factor w/ 2 levels "0","1": 2 2 1 2 2 2 1 2 1 2 ...
    ##  $ HvyAlcoholConsump   : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ AnyHealthcare       : num [1:4217] 1 1 1 1 1 1 1 1 1 0 ...
    ##  $ NoDocbcCost         : num [1:4217] 0 0 0 0 0 1 0 0 0 1 ...
    ##  $ GenHlth             : num [1:4217] 5 4 1 5 4 5 4 3 3 4 ...
    ##  $ MentHlth            : num [1:4217] 15 0 0 15 0 5 3 0 0 0 ...
    ##  $ PhysHlth            : num [1:4217] 30 0 0 30 1 10 6 0 7 30 ...
    ##  $ DiffWalk            : num [1:4217] 1 0 1 1 1 1 1 0 1 0 ...
    ##  $ Sex                 : Factor w/ 2 levels "0","1": 1 2 1 1 2 1 2 1 1 2 ...
    ##  $ Age                 : Factor w/ 13 levels "1","2","3","4",..: 13 12 13 9 9 7 12 11 10 5 ...
    ##  $ Education           : num [1:4217] 12 12 12 12 12 12 12 12 12 12 ...
    ##  $ Income              : Factor w/ 8 levels "1","2","3","4",..: 3 4 2 3 5 2 2 2 3 3 ...

Spiting up data training and validation data sets.

``` r
set.seed(5)
trainIndex <- createDataPartition(subset$Diabetes_binary, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train_data = subset[trainIndex, ]
val_data = subset[-trainIndex, ]
```

### 0.5.1 Log loss

**Log loss**, also known as **logarithmic loss** or **cross-entropy
loss**, is a common evaluation metric for binary classification models.
It measures the performance of a model by quantifying the difference
between predicted probabilities and actual values. Log-loss is
indicative of how close the prediction probability is to the
corresponding actual/true value, penalizing inaccurate predictions with
higher values. Lower log-loss indicates better model performance.

Mathematical interpretation: Log Loss is the negative average of the log
of corrected predicted probabilities for each instance.

$$log \ loss = -\frac{1}{N} \sum_{i=1}^N y_i log(p(y_i)) + (1-y_i)log(1-p(y_i))$$

$p(y_i)$ is the probability of $1$. $1-p(y_i)$ is the probability of 0.

### 0.5.2 Logistic regression

Logistic regression is a statistical and machine learning model used for
binary classification tasks. It’s a type of regression analysis that’s
well-suited for predicting the probability of an observation belonging
to one of two classes or categories.

- Logistic regression is used when the response variable is binary,
  meaning it has two possible outcomes or classes.
- Logistic regression uses the `sigmoid` function to model the
  relationship between the features and the probability of the binary
  outcome. The logistic function has an S-shaped curve and maps any
  real-valued number to a value between 0 and 1.
  $p(x)=\frac{1}{1+e^{-(\beta_0+\beta_1x)}}$. ($p(x)$ is the probability
  of the dependent variable being 1)
- The goal of logistic regression is to find the best-fitting model by
  estimating the coefficients $\beta_0$, $\beta_1$. This is typically
  done using a process called maximum likelihood estimation. The
  coefficients are adjusted to maximize the likelihood of the observed
  data given the model.

#### 0.5.2.1 Logistic regression model 1

``` r
train_data$Diabetes_binary_transformed = train_data$Diabetes_binary
val_data$Diabetes_binary_transformed = val_data$Diabetes_binary

levels(train_data$Diabetes_binary_transformed) = make.names(levels(train_data$Diabetes_binary_transformed))
levels(val_data$Diabetes_binary_transformed) = make.names(levels(val_data$Diabetes_binary_transformed))
```

``` r
train.control = trainControl(method = "repeatedcv", 
                              number = 5, 
                              repeats = 10,
                              summaryFunction=mnLogLoss,
                              classProbs = TRUE)

set.seed(83)
lr_model_1 = train(Diabetes_binary_transformed ~ 
                                   HighChol+
                                   BMI + 
                                   GenHlth, 
                                 data = train_data,
                                 method = "glm", 
                                 family="binomial",
                                 metric="logLoss",
                                 trControl = train.control
                                )
summary(lr_model_1)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -4.592156   0.247084 -18.585  < 2e-16 ***
    ## HighChol1    0.781034   0.091961   8.493  < 2e-16 ***
    ## BMI          0.042900   0.005958   7.200 6.03e-13 ***
    ## GenHlth      0.536778   0.044825  11.975  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 3179.4  on 2948  degrees of freedom
    ## AIC: 3187.4
    ## 
    ## Number of Fisher Scoring iterations: 4

Create custom fucntion for log loss calculation.

``` r
calculateLogLoss <- function(predicted_probabilities, true_labels) {
  # Avoid log(0) or log(1) which can lead to NaN or Inf values
  predicted_probabilities = pmax(pmin(predicted_probabilities, 1 - 1e-15), 
                                 1e-15)

  log_loss <- -mean(true_labels * log(predicted_probabilities) + 
                      (1 - true_labels) * log(1 - predicted_probabilities))
  return(log_loss)
}
```

Calculate log loss for train data set

``` r
train_predictions = predict(lr_model_1, 
                             newdata = train_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.numeric(train_data$Diabetes_binary)

log_loss_train_lr_model_1 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_1))
```

    ## [1] "Log Loss: 0.213560995554274"

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_1, 
                             newdata = val_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.numeric(val_data$Diabetes_binary)

log_loss_val_lr_model_1 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_1))
```

    ## [1] "Log Loss: 0.164892302343813"

#### 0.5.2.2 Logistic regression model 2

``` r
train.control = trainControl(method = "repeatedcv", 
                              number = 5, 
                              repeats = 10,
                              summaryFunction=mnLogLoss,
                              classProbs = TRUE)

set.seed(8)
lr_model_2 = train(Diabetes_binary_transformed ~ 
                                   poly(BMI, 2) + 
                                   HighChol + HeartDiseaseorAttack+
                                   HighChol:HeartDiseaseorAttack,
                                 data = train_data,
                                 method = "glm", 
                                 family="binomial",
                                 metric="logLoss",
                                 trControl = train.control
                                )
summary(lr_model_2)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                                    Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                        -1.70440    0.08161 -20.885  < 2e-16 ***
    ## `poly(BMI, 2)1`                    19.41529    2.31775   8.377  < 2e-16 ***
    ## `poly(BMI, 2)2`                   -11.25094    2.53156  -4.444 8.82e-06 ***
    ## HighChol1                           1.00723    0.10228   9.848  < 2e-16 ***
    ## HeartDiseaseorAttack1               1.19581    0.18591   6.432 1.26e-10 ***
    ## `HighChol1:HeartDiseaseorAttack1`  -0.56475    0.22056  -2.561   0.0105 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 3252.1  on 2946  degrees of freedom
    ## AIC: 3264.1
    ## 
    ## Number of Fisher Scoring iterations: 4

Calculate log loss for train data set

``` r
train_predictions = predict(lr_model_2, 
                             newdata = train_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.numeric(train_data$Diabetes_binary)

log_loss_train_lr_model_2 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_2))
```

    ## [1] "Log Loss: 0.186836031977423"

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_2, 
                             newdata = val_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.numeric(val_data$Diabetes_binary)

log_loss_val_lr_model_2 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_2))
```

    ## [1] "Log Loss: 0.182165322539743"

#### 0.5.2.3 Logistic regression model 3

``` r
train.control = trainControl(method = "repeatedcv", 
                              number = 5, 
                              repeats = 10,
                              summaryFunction=mnLogLoss,
                              classProbs = TRUE)

set.seed(10)
lr_model_3 = train(Diabetes_binary_transformed ~ Income+
                     Age+GenHlth+
                     HighBP+
                     HeartDiseaseorAttack+
                     poly(BMI, 2),
                   data = train_data,
                   method = "glm", 
                   family="binomial",
                   metric="logLoss",
                   trControl = train.control
                   )
summary(lr_model_3)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)            -4.55153    1.06940  -4.256 2.08e-05 ***
    ## Income2                -0.09801    0.13317  -0.736   0.4617    
    ## Income3                -0.26546    0.13646  -1.945   0.0517 .  
    ## Income4                -0.32754    0.15383  -2.129   0.0332 *  
    ## Income5                -0.03149    0.15968  -0.197   0.8437    
    ## Income6                -0.31032    0.20075  -1.546   0.1222    
    ## Income7                -0.44303    0.30508  -1.452   0.1465    
    ## Income8                -0.44358    0.29623  -1.497   0.1343    
    ## Age2                    0.94088    1.18711   0.793   0.4280    
    ## Age3                    0.37080    1.13855   0.326   0.7447    
    ## Age4                    0.34767    1.10713   0.314   0.7535    
    ## Age5                    0.53675    1.08749   0.494   0.6216    
    ## Age6                    1.25698    1.06991   1.175   0.2401    
    ## Age7                    0.95707    1.06483   0.899   0.3688    
    ## Age8                    1.50201    1.06113   1.415   0.1569    
    ## Age9                    1.50045    1.06053   1.415   0.1571    
    ## Age10                   1.84630    1.06003   1.742   0.0816 .  
    ## Age11                   1.83384    1.06027   1.730   0.0837 .  
    ## Age12                   1.71433    1.06152   1.615   0.1063    
    ## Age13                   1.74607    1.05939   1.648   0.0993 .  
    ## GenHlth                 0.47562    0.04789   9.932  < 2e-16 ***
    ## HighBP1                 0.70305    0.10423   6.745 1.53e-11 ***
    ## HeartDiseaseorAttack1   0.44419    0.10717   4.145 3.40e-05 ***
    ## `poly(BMI, 2)1`        19.58091    2.47392   7.915 2.47e-15 ***
    ## `poly(BMI, 2)2`       -12.12649    2.68387  -4.518 6.23e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 3009.7  on 2927  degrees of freedom
    ## AIC: 3059.7
    ## 
    ## Number of Fisher Scoring iterations: 5

Calculate log loss for train data set

``` r
train_predictions = predict(lr_model_3, 
                             newdata = train_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.numeric(train_data$Diabetes_binary)

log_loss_train_lr_model_3 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_3))
```

    ## [1] "Log Loss: 0.269088363800483"

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_3, 
                             newdata = val_data %>% select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.numeric(val_data$Diabetes_binary)

log_loss_val_lr_model_3 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_3))
```

    ## [1] "Log Loss: 0.246097189832437"
