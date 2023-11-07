ST558, Project3
================
Jacob Press, Nataliya Peshekhodko
2023-11-07

- <a href="#1-introduction" id="toc-1-introduction">1 Introduction</a>
- <a href="#2-packages" id="toc-2-packages">2 Packages</a>
- <a href="#3-data" id="toc-3-data">3 Data</a>
- <a href="#4-explanatory-data-analysiseda"
  id="toc-4-explanatory-data-analysiseda">4 Explanatory Data
  Analysis(EDA)</a>
- <a href="#5-modeling" id="toc-5-modeling">5 Modeling</a>
  - <a href="#51-log-loss" id="toc-51-log-loss">5.1 Log loss</a>
  - <a href="#52-logistic-regression" id="toc-52-logistic-regression">5.2
    Logistic regression</a>
    - <a href="#521-fit-logistic-regression-model-1"
      id="toc-521-fit-logistic-regression-model-1">5.2.1 Fit Logistic
      regression model 1</a>
    - <a href="#522-fit-logistic-regression-model-2"
      id="toc-522-fit-logistic-regression-model-2">5.2.2 Fit Logistic
      regression model 2</a>
    - <a href="#523-fit-logistic-regression-model-3"
      id="toc-523-fit-logistic-regression-model-3">5.2.3 Fit Logistic
      regression model 3</a>
  - <a href="#53-lasso-logistic-regression"
    id="toc-53-lasso-logistic-regression">5.3 LASSO logistic regression</a>
    - <a href="#531-fit-and-validate-lasso-logistic-regression"
      id="toc-531-fit-and-validate-lasso-logistic-regression">5.3.1 Fit and
      validate LASSO logistic regression</a>
  - <a href="#54-classification-tree-model"
    id="toc-54-classification-tree-model">5.4 Classification tree model</a>
  - <a href="#55-random-forest-model" id="toc-55-random-forest-model">5.5
    Random forest model</a>
  - <a href="#56-new-model---support-vector-machine"
    id="toc-56-new-model---support-vector-machine">5.6 New model - Support
    Vector Machine</a>
  - <a href="#57-new-model---naive-bayes"
    id="toc-57-new-model---naive-bayes">5.7 New model - Naive Bayes</a>
- <a href="#6-summary" id="toc-6-summary">6 Summary</a>

# 1 Introduction

In this project we will read and analyze dataset [Diabetes health
indicator
dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/)
for each education level as specified by
[EDUCA](https://www.icpsr.umich.edu/web/NAHDAP/studies/34085/datasets/0001/variables/EDUCA?archive=NAHDAP).
**Level 1** (Never attended school or only kindergarten) and **Level 2**
(Grades 1 - 8) will be combined. We will create different classification
models for predicting the `Diabetes_binary` variable. The best model
will be chosen based on `log loss` function.

Levels of educations based on EDUCA

``` r
education_levels=list()
education_levels[['Never attended school or only kindergarten or Grades 1-8']] = '12'
education_levels[['Grades 9-11 (Some high school)']] = '3'
education_levels[['Grade 12 or GED (High school graduate)']] = '4'
education_levels[['College 1 year to 3 years (Some college or technical school)']] = '5'
education_levels[['College 4 years or more (College graduate)']] = '6'
```

This report is built for education level **College 1 year to 3 years
(Some college or technical school)**.

# 2 Packages

In order to achieve our goals, we will be using the following `R`
packages.

``` r
library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(klaR)
```

- `tidyverse` - is a collection of R packages, required for data
  transformation and manipulation
- `caret` - required for training and evaluating machine learning models
- `ggplot2` - required for for creating data visualizations and graphics
- `corrplot` - required for correlation matrix visualizing

# 3 Data

Reading data from `diabetes_binary_health_indicators_BRFSS2015.csv`
file.

``` r
data = read_csv('./data/diabetes_binary_health_indicators_BRFSS2015.csv')
```

Checking for NA values

``` r
sum(is.na(data))
```

    ## [1] 0

There are no missing values in the data set.

Let’s look at the head of data.

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
education_level = params$education_level

subset <- transformed %>%
  filter(Education == education_level)
```

Checking data structure:

``` r
str(subset)
```

    ## tibble [69,910 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : num [1:69910] 0 1 0 0 1 0 1 1 0 0 ...
    ##  $ HighBP              : num [1:69910] 1 1 1 0 0 0 1 1 0 0 ...
    ##  $ HighChol            : num [1:69910] 1 1 1 0 0 0 0 1 0 0 ...
    ##  $ CholCheck           : num [1:69910] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ BMI                 : num [1:69910] 24 30 34 26 23 28 27 34 22 26 ...
    ##  $ Smoker              : num [1:69910] 0 1 1 1 1 1 0 1 0 1 ...
    ##  $ Stroke              : num [1:69910] 0 0 0 0 0 0 0 1 0 0 ...
    ##  $ HeartDiseaseorAttack: num [1:69910] 0 1 0 0 0 0 0 0 0 0 ...
    ##  $ PhysActivity        : num [1:69910] 1 0 0 0 1 0 1 1 1 1 ...
    ##  $ Fruits              : num [1:69910] 1 1 1 0 0 0 1 0 1 1 ...
    ##  $ Veggies             : num [1:69910] 1 1 1 1 0 1 1 0 1 1 ...
    ##  $ HvyAlcoholConsump   : num [1:69910] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ AnyHealthcare       : num [1:69910] 1 1 1 1 1 1 1 1 1 0 ...
    ##  $ NoDocbcCost         : num [1:69910] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ GenHlth             : num [1:69910] 2 5 3 3 2 3 1 4 2 1 ...
    ##  $ MentHlth            : num [1:69910] 3 30 0 0 0 0 0 0 0 0 ...
    ##  $ PhysHlth            : num [1:69910] 0 30 30 15 0 7 0 7 10 1 ...
    ##  $ DiffWalk            : num [1:69910] 0 1 1 0 0 0 0 1 0 0 ...
    ##  $ Sex                 : num [1:69910] 0 0 0 0 1 1 0 0 0 1 ...
    ##  $ Age                 : num [1:69910] 11 9 10 7 7 5 13 9 12 4 ...
    ##  $ Education           : num [1:69910] 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ Income              : num [1:69910] 4 1 1 7 6 5 4 4 7 3 ...

Variables in the data set:

- **Diabetes_binary** - 0 = no diabetes, 1 = diabetes
- **HighBP** - 0 = no high blood pressure, 1 = high blood pressure
- **HighChol** - 0 = no high cholesterol, 1 = high cholesterol
- **CholCheck** - 0 = no cholesterol check in 5 years, 1 = yes
  cholesterol check in 5 years
- **BMI** - Body Mass Index
- **Smoker** - Have you smoked at least 100 cigarettes in your entire
  life? 0 = no, 1 = yes
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

# 4 Explanatory Data Analysis(EDA)

First, let’s look at number of thr records with Diabetes and without
Diabetes for the selected education level:

``` r
table (factor (subset$Diabetes_binary, labels = c("No diabet", "Diabet")) )
```

    ## 
    ## No diabet    Diabet 
    ##     59556     10354

Let’s look at `Age` distribution for the selected education level:

``` r
ggplot(data = subset, aes(x = Age)) +
  geom_histogram(color = "black", fill = 'brown') +
  labs(title = "Histogram of Age groups distribution", 
       x = "Age group", 
       y = "Frequency")
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Let’s look at number of cases with Diabetes and without Diabetes for
each age group for the selected education level

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$Age, labels = c("Age 18 - 24", "Age 25 to 29", "Age 30 to 34", 
                                    "Age 35 to 39", "Age 40 to 44",
                                    "Age 45 to 49", "Age 50 to 54",
                                    "Age 55 to 59", "Age 60 to 64", 
                                    "Age 65 to 69", "Age 70 to 74",
                                    "Age 75 to 79", "Age 80 or older")) )
```

    ##            
    ##             Age 18 - 24 Age 25 to 29 Age 30 to 34 Age 35 to 39 Age 40 to 44 Age 45 to 49 Age 50 to 54 Age 55 to 59
    ##   No diabet        2381         2154         2919         3514         3917         4603         6477         7466
    ##   Diabet             25           50           93          220          344          515          927         1316
    ##            
    ##             Age 60 to 64 Age 65 to 69 Age 70 to 74 Age 75 to 79 Age 80 or older
    ##   No diabet         7660         6990         4731         3261            3483
    ##   Diabet            1805         1986         1394          889             790

Number of cases with Diabetes and without Diabetes for males and
females.

``` r
table(factor (subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$Sex, labels = c("Female", "Male")))
```

    ##            
    ##             Female  Male
    ##   No diabet  35539 24017
    ##   Diabet      5683  4671

Linear correlation between numeric variables.

``` r
corrplot(cor(as.matrix(subset %>% dplyr::select(-Education))), 
         type="upper", 
         tl.pos = "lt")
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Number of cases with Diabetes and without Diabetes for each general
health level.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$GenHlth, labels = c("Excellent", "Very good", "Good", "Fair", "Poor")) )
```

    ##            
    ##             Excellent Very good  Good  Fair  Poor
    ##   No diabet     10290     22605 18136  6282  2243
    ##   Diabet          314      1869  4070  2817  1284

Number of cases with Diabetes and without Diabetes for high blood
pressure and normal blood pressure patients.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$HighBP, labels = c("No high BP", "High BP")) )
```

    ##            
    ##             No high BP High BP
    ##   No diabet      36499   23057
    ##   Diabet          2520    7834

Number of cases with Diabetes and without Diabetes for high cholesterol
and normal cholesterol patients.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor (subset$HighChol, labels = c("No high chol", "High chol")))
```

    ##            
    ##             No high chol High chol
    ##   No diabet        36855     22701
    ##   Diabet            3368      6986

BMI distribution for patients with Diabetes and without Diabetes for the
selected education level.

``` r
ggplot(subset, aes(x = as_factor(Diabetes_binary), 
                   y = BMI, 
                   fill = as_factor(Diabetes_binary))) +
  geom_boxplot() +
  labs(title = "BMI distribution for patients with and without diabetes", 
       x = "Diabetes", 
       y = "BMI") +
  scale_fill_manual(values = c("0" = "grey", "1" = "red"),
                    labels = c("0" = "Without Diabetes", "1" = "With Diabetes")) +
  labs(fill = "Diabetes Status")
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
ggplot(subset, aes(x = as.factor(GenHlth), fill = as.factor(Diabetes_binary), group = Diabetes_binary)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Number of Cases with Diabetes and Without Diabetes by GenHlth Level",
    x = "GenHlth Level",
    y = "Number of Cases"
  ) +
  scale_fill_manual(
    values = c("0" = "grey", "1" = "red"),
    labels = c("0" = "Without Diabetes", "1" = "With Diabetes")
  ) + 
  labs(fill = "Diabetes Status")
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

# 5 Modeling

Converting some of the variables to factors.

``` r
names = c('HighBP' ,'HighChol', 
          'CholCheck', 'Smoker', 
          'Diabetes_binary', 'Stroke',
          'HeartDiseaseorAttack', 'PhysActivity',
          'Fruits', 'Veggies', 
          'HvyAlcoholConsump', 'Sex',
          'Age','Income', 'GenHlth', 
          'MentHlth', 'PhysHlth', 'DiffWalk',
          'AnyHealthcare', 'NoDocbcCost')
subset[,names] = lapply(subset[,names] , factor)
str(subset)
```

    ## tibble [69,910 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : Factor w/ 2 levels "0","1": 1 2 1 1 2 1 2 2 1 1 ...
    ##  $ HighBP              : Factor w/ 2 levels "0","1": 2 2 2 1 1 1 2 2 1 1 ...
    ##  $ HighChol            : Factor w/ 2 levels "0","1": 2 2 2 1 1 1 1 2 1 1 ...
    ##  $ CholCheck           : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ BMI                 : num [1:69910] 24 30 34 26 23 28 27 34 22 26 ...
    ##  $ Smoker              : Factor w/ 2 levels "0","1": 1 2 2 2 2 2 1 2 1 2 ...
    ##  $ Stroke              : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 2 1 1 ...
    ##  $ HeartDiseaseorAttack: Factor w/ 2 levels "0","1": 1 2 1 1 1 1 1 1 1 1 ...
    ##  $ PhysActivity        : Factor w/ 2 levels "0","1": 2 1 1 1 2 1 2 2 2 2 ...
    ##  $ Fruits              : Factor w/ 2 levels "0","1": 2 2 2 1 1 1 2 1 2 2 ...
    ##  $ Veggies             : Factor w/ 2 levels "0","1": 2 2 2 2 1 2 2 1 2 2 ...
    ##  $ HvyAlcoholConsump   : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ AnyHealthcare       : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 1 ...
    ##  $ NoDocbcCost         : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ GenHlth             : Factor w/ 5 levels "1","2","3","4",..: 2 5 3 3 2 3 1 4 2 1 ...
    ##  $ MentHlth            : Factor w/ 31 levels "0","1","2","3",..: 4 31 1 1 1 1 1 1 1 1 ...
    ##  $ PhysHlth            : Factor w/ 31 levels "0","1","2","3",..: 1 31 31 16 1 8 1 8 11 2 ...
    ##  $ DiffWalk            : Factor w/ 2 levels "0","1": 1 2 2 1 1 1 1 2 1 1 ...
    ##  $ Sex                 : Factor w/ 2 levels "0","1": 1 1 1 1 2 2 1 1 1 2 ...
    ##  $ Age                 : Factor w/ 13 levels "1","2","3","4",..: 11 9 10 7 7 5 13 9 12 4 ...
    ##  $ Education           : num [1:69910] 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ Income              : Factor w/ 8 levels "1","2","3","4",..: 4 1 1 7 6 5 4 4 7 3 ...

Spiting up data training and validation data sets.

``` r
set.seed(5)
trainIndex <- createDataPartition(subset$Diabetes_binary, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train_data = subset[trainIndex, ]
val_data = subset[-trainIndex, ]


#Taking sample TEMPORARY. 
#Will be removed
#Having performance issues and not able to render for original sizes. 
#TODO: remove the following lines
train_data <- train_data[sample(nrow(train_data), size = 500), ]
val_data <- val_data[sample(nrow(val_data), size = 200), ]
```

## 5.1 Log loss

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

$p(y_i)$ is the probability of $1$.

$1-p(y_i)$ is the probability of 0.

## 5.2 Logistic regression

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

Creating lists to store model performance on train and validations data
sets.

``` r
models_performace_train = list()
models_performace_val = list()
```

### 5.2.1 Fit Logistic regression model 1

``` r
train_data$Diabetes_binary_transformed = train_data$Diabetes_binary
val_data$Diabetes_binary_transformed = val_data$Diabetes_binary

levels(train_data$Diabetes_binary_transformed) = make.names(levels(train_data$Diabetes_binary_transformed))
levels(val_data$Diabetes_binary_transformed) = make.names(levels(val_data$Diabetes_binary_transformed))
```

``` r
train.control = trainControl(method = "cv", 
                              number = 5, 
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
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -5.41963    0.81384  -6.659 2.75e-11 ***
    ## HighChol1    1.31522    0.27723   4.744 2.09e-06 ***
    ## BMI          0.06037    0.01929   3.130  0.00175 ** 
    ## GenHlth2     0.90160    0.65075   1.385  0.16591    
    ## GenHlth3     1.44067    0.63830   2.257  0.02401 *  
    ## GenHlth4     1.76049    0.67208   2.619  0.00881 ** 
    ## GenHlth5     2.33565    0.75703   3.085  0.00203 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 436.34  on 499  degrees of freedom
    ## Residual deviance: 367.78  on 493  degrees of freedom
    ## AIC: 381.78
    ## 
    ## Number of Fisher Scoring iterations: 6

Create custom function for log loss calculation.

``` r
calculateLogLoss <- function(predicted_probabilities, true_labels) {
  predicted_probabilities = pmax(pmin(predicted_probabilities, 1 - 1e-15), 
                                 1e-15)

  log_loss <- -mean(true_labels * log(predicted_probabilities) + 
                      (1 - true_labels) * log(1 - predicted_probabilities))
  return(log_loss)
}
```

Calculate log loss for train data set for logistic regression model \#1.

``` r
train_predictions = predict(lr_model_1, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_lr_model_1 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_1))
```

    ## [1] "Log Loss: 2.06823946399948"

``` r
models_performace_train[["logistic_regression_model_1"]] <- log_loss_train_lr_model_1
```

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_1, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_lr_model_1 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_1))
```

    ## [1] "Log Loss: 2.00691668005207"

``` r
models_performace_val[["logistic_regression_model_1"]] = log_loss_val_lr_model_1
```

### 5.2.2 Fit Logistic regression model 2

``` r
train.control = trainControl(method = "cv", 
                              number = 5, 
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
    ##                                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                        -2.7058     0.2447 -11.056  < 2e-16 ***
    ## `poly(BMI, 2)1`                    14.8809     3.0247   4.920 8.67e-07 ***
    ## `poly(BMI, 2)2`                    -6.4593     2.7555  -2.344   0.0191 *  
    ## HighChol1                           1.3233     0.2962   4.467 7.92e-06 ***
    ## HeartDiseaseorAttack1               1.6137     0.7410   2.178   0.0294 *  
    ## `HighChol1:HeartDiseaseorAttack1`  -0.3101     0.8660  -0.358   0.7202    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 436.34  on 499  degrees of freedom
    ## Residual deviance: 364.88  on 494  degrees of freedom
    ## AIC: 376.88
    ## 
    ## Number of Fisher Scoring iterations: 5

Calculate log loss for train data set

``` r
train_predictions = predict(lr_model_2, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_lr_model_2 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_2))
```

    ## [1] "Log Loss: 2.08378166100139"

``` r
models_performace_train[["logistic_regression_model_2"]] = log_loss_train_lr_model_2
```

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_2, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_lr_model_2 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_2))
```

    ## [1] "Log Loss: 2.00151083244908"

``` r
models_performace_val[["logistic_regression_model_2"]] = log_loss_val_lr_model_2
```

### 5.2.3 Fit Logistic regression model 3

``` r
train.control = trainControl(method = "cv", 
                              number = 5, 
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
    ##                         Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)            -18.81127 1284.64905  -0.015 0.988317    
    ## Income2                  0.32736    0.88831   0.369 0.712486    
    ## Income3                  0.31925    0.97326   0.328 0.742893    
    ## Income4                 -0.28857    0.89708  -0.322 0.747694    
    ## Income5                 -0.87913    0.89322  -0.984 0.325003    
    ## Income6                 -0.53967    0.82874  -0.651 0.514922    
    ## Income7                 -0.88491    0.85250  -1.038 0.299261    
    ## Income8                  0.03232    0.81229   0.040 0.968257    
    ## Age2                    -0.14701 1909.28665   0.000 0.999939    
    ## Age3                    -0.66997 1691.45642   0.000 0.999684    
    ## Age4                    14.83061 1284.64889   0.012 0.990789    
    ## Age5                    13.66514 1284.64917   0.011 0.991513    
    ## Age6                    15.24343 1284.64878   0.012 0.990533    
    ## Age7                    14.51288 1284.64881   0.011 0.990986    
    ## Age8                    15.66917 1284.64876   0.012 0.990268    
    ## Age9                    15.68211 1284.64874   0.012 0.990260    
    ## Age10                   15.35998 1284.64876   0.012 0.990460    
    ## Age11                   15.69848 1284.64877   0.012 0.990250    
    ## Age12                   15.88321 1284.64879   0.012 0.990135    
    ## Age13                   16.61620 1284.64880   0.013 0.989680    
    ## GenHlth2                 0.64022    0.72477   0.883 0.377051    
    ## GenHlth3                 1.09699    0.71723   1.529 0.126145    
    ## GenHlth4                 1.49157    0.75381   1.979 0.047848 *  
    ## GenHlth5                 1.61479    0.84816   1.904 0.056927 .  
    ## HighBP1                  1.46402    0.34329   4.265    2e-05 ***
    ## HeartDiseaseorAttack1    0.64869    0.44301   1.464 0.143117    
    ## `poly(BMI, 2)1`         13.43511    3.49955   3.839 0.000123 ***
    ## `poly(BMI, 2)2`         -9.38649    3.77513  -2.486 0.012904 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 436.34  on 499  degrees of freedom
    ## Residual deviance: 309.78  on 472  degrees of freedom
    ## AIC: 365.78
    ## 
    ## Number of Fisher Scoring iterations: 17

Calculate log loss for train data set

``` r
train_predictions = predict(lr_model_3, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_lr_model_3 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lr_model_3))
```

    ## [1] "Log Loss: 4.6605997060963"

``` r
models_performace_train[["logistic_regression_model_3"]] = log_loss_train_lr_model_3
```

Calculate log loss for validation data set

``` r
val_predictions = predict(lr_model_3, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_lr_model_3 = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lr_model_3))
```

    ## [1] "Log Loss: 4.21478373525451"

``` r
models_performace_val[["logistic_regression_model_3"]] = log_loss_val_lr_model_3
```

## 5.3 LASSO logistic regression

`LASSO (Least Absolute Shrinkage and Selection Operator) logistic regression`
is a statistical method that combines logistic regression with LASSO
regularization. It is used for binary classification problems where you
want to predict the probability of an event occurring, such as whether a
customer will buy a product (yes/no) based on various predictor
variables.

How it works:

- `LASSO logistic regression` models the probability of an event using
  the logistic function. It models the log-odds of the event as a linear
  combination of predictor variables. The logistic function is used to
  transform the linear combination into probabilities.
- `LASSO` adds a regularization term to the logistic regression model.
  The regularization term is a penalty based on the absolute values of
  the model coefficients (L1 regularization). This penalty encourages
  some of the coefficient values to become exactly zero, effectively
  performing feature selection.
- `LASSO` regularization promotes sparsity in the model. It can
  automatically select a subset of the most relevant predictor variables
  by setting the coefficients of irrelevant variables to zero. This
  helps to reduce overfitting and build more interpretable models.
- The degree of regularization is controlled by a hyper parameter
  denoted as $\lambda$.

### 5.3.1 Fit and validate LASSO logistic regression

``` r
train.control <- trainControl(method = "cv",
                              number = 5, 
                              summaryFunction=mnLogLoss,
                              classProbs = TRUE)


set.seed(2)

# Limiting number of features due to performance issues
lasso_log_reg<-train(#Diabetes_binary ~., 
                   Diabetes_binary ~ HighBP + HighChol + BMI + Smoker + AnyHealthcare + GenHlth + Age + Sex,
                   data = dplyr::select(train_data, -Diabetes_binary_transformed),
                   method = 'glmnet',
                   metric="logLoss",
                   tuneGrid = expand.grid(alpha = 1, 
                                          lambda=seq(0, 1, by = 0.25))
)
```

    ## Warning in train.default(x, y, weights = w, ...): The metric "logLoss" was not in the result set. Accuracy will be used
    ## instead.

``` r
lasso_log_reg$results
```

    ##   alpha lambda  Accuracy     Kappa AccuracySD   KappaSD
    ## 1     1   0.00 0.8156319 0.1673699 0.02414534 0.0665525
    ## 2     1   0.25 0.8361410 0.0000000 0.02186780 0.0000000
    ## 3     1   0.50 0.8361410 0.0000000 0.02186780 0.0000000
    ## 4     1   0.75 0.8361410 0.0000000 0.02186780 0.0000000
    ## 5     1   1.00 0.8361410 0.0000000 0.02186780 0.0000000

Obtained the best tuning parameter $\lambda$ value is

``` r
lasso_log_reg$bestTune$lambda
```

    ## [1] 1

Plot obtained accuracy for different $\lambda$ values.

``` r
plot(lasso_log_reg)
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

Calculate log loss for train data set

``` r
train_predictions = predict(lasso_log_reg, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_lasso = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_lasso))
```

    ## [1] "Log Loss: 1.58079701892314"

``` r
models_performace_train[["lasso"]] = log_loss_train_lasso
```

Calculate log loss for validation data set

``` r
val_predictions = predict(lasso_log_reg, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_lasso = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_lasso))
```

    ## [1] "Log Loss: 1.55235287424248"

``` r
models_performace_val[["lasso"]] = log_loss_val_lasso
```

## 5.4 Classification tree model

A Classification tree model is a supervised machine learning model used
to predict group membership. It has a hierarchical tree structure
consisting a root node, branches, internal nodes, and leaf nodes.
Classification trees are used when the target variable is categorical.
One benefit to classification trees is they are intuitive and usually
easy to explain.

Here is a break down of the tree structure:

- **Root Node** - The beginning node on the graph.
- **Branches** - The arrows connecting the nodes.
- **Internal Nodes** - A non-leaf node denoting a test on an attribute.
- **Leaf Nodes** - The terminal node displaying the classification.

``` r
train_control <- trainControl(method = "cv",
                              summaryFunction=mnLogLoss,
                              classProbs = TRUE,
                              number = 5)
set.seed(1122)
tree_model <- train(Diabetes_binary_transformed ~ ., 
                data = dplyr::select(train_data, -Diabetes_binary), 
                method = "rpart",
                trControl = train_control,
                metric="logLoss",
                tuneGrid = data.frame(cp=seq(0,.022, by = .001))
                )

tree_model$results
```

    ##       cp   logLoss  logLossSD
    ## 1  0.000 0.4042589 0.05191466
    ## 2  0.001 0.4042589 0.05191466
    ## 3  0.002 0.4042589 0.05191466
    ## 4  0.003 0.4042589 0.05191466
    ## 5  0.004 0.4042589 0.05191466
    ## 6  0.005 0.4042589 0.05191466
    ## 7  0.006 0.4042589 0.05191466
    ## 8  0.007 0.4042589 0.05191466
    ## 9  0.008 0.4017918 0.04921642
    ## 10 0.009 0.4017918 0.04921642
    ## 11 0.010 0.4017918 0.04921642
    ## 12 0.011 0.4017918 0.04921642
    ## 13 0.012 0.4017918 0.04921642
    ## 14 0.013 0.4017918 0.04921642
    ## 15 0.014 0.4017918 0.04921642
    ## 16 0.015 0.4017918 0.04921642
    ## 17 0.016 0.3989471 0.05362653
    ## 18 0.017 0.3989471 0.05362653
    ## 19 0.018 0.3989471 0.05362653
    ## 20 0.019 0.3989471 0.05362653
    ## 21 0.020 0.3989471 0.05362653
    ## 22 0.021 0.3989471 0.05362653
    ## 23 0.022 0.3989471 0.05362653

``` r
plot(tree_model)
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-39-1.png)<!-- -->
Calculate log loss for train data set

``` r
train_predictions = predict(tree_model, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_tree = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_tree))
```

    ## [1] "Log Loss: 1.58079701892314"

``` r
models_performace_train[["classification_tree"]] = log_loss_train_tree
```

Calculate log loss for validation data set

``` r
val_predictions = predict(tree_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_tree = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_tree))
```

    ## [1] "Log Loss: 1.55235287424248"

``` r
models_performace_val[["classification_tree"]] = log_loss_val_tree
```

## 5.5 Random forest model

A Random Forest classification model is a supervised machine learning
model used for classification tasks. It is an ensemble of multiple
decision trees, where each tree predicts the class label of an input
based on a set of features. The final prediction in a Random Forest is
determined through a combination of predictions from individual decision
trees, often using majority voting for classification tasks.

Random Forest might be chosen over a basic Classification Tree for
several reasons:

- **Generalization** - Random Forest typically offers better
  generalization to new, unseen data. It reduces the risk of
  overfitting, which is a common issue with basic Classification Trees.
- **Higher Accuracy** - Random Forest often provides higher accuracy
  because it combines multiple decision trees. The majority voting from
  these trees leads to a more reliable and accurate classification.
- **Robustness to Noise** - Basic Classification Trees are sensitive to
  noise in the data, which can lead to overfitting. Random Forest,
  through ensemble learning, is more robust to noise and outliers.
- **Reduced Variance** - A basic Classification Tree can vary
  significantly with small changes in the training data. Random Forest
  reduces this variance because the ensemble of trees accounts for
  different sources of variance.
- **Feature Selection** - Random Forest provides a measure of feature
  importance. It can help identify which features are most relevant for
  making predictions. This feature selection is especially valuable when
  dealing with high-dimensional data.

``` r
train_control <- trainControl(
  method = "cv",   
  number = 5,
  summaryFunction=mnLogLoss,
  classProbs = TRUE
)

# limiting number of the features due 
# to performance issues with random forest algorithm
set.seed(11)
rf_model = train(
  #Diabetes_binary_transformed ~ ., 
  Diabetes_binary_transformed ~ HighChol+
                                BMI + 
                                GenHlth+
                                HeartDiseaseorAttack,
  data = dplyr::select(train_data, -Diabetes_binary),
  method = "rf",
  metric="logLoss",
  tuneGrid = data.frame(mtry = c(1:3)), 
  trControl = train_control
)

rf_model$results
```

    ##   mtry   logLoss logLossSD
    ## 1    1 1.1608183 0.4972774
    ## 2    2 1.0110875 0.3786886
    ## 3    3 0.8726241 0.3064959

``` r
plot(rf_model)
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

Calculate log loss for train data set

``` r
train_predictions = predict(rf_model, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_rf = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_rf))
```

    ## [1] "Log Loss: 14.2387279811037"

``` r
models_performace_train[["random_forest"]] = log_loss_train_rf
```

Calculate log loss for validation data set

``` r
val_predictions = predict(rf_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_rf = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_rf))
```

    ## [1] "Log Loss: 11.4669515205831"

``` r
models_performace_val[["random_forest"]] = log_loss_val_rf
```

## 5.6 New model - Support Vector Machine

Support Vector Machine(SVM) is a supervised machine learning algorithm
that is used for both classification and regression tasks. It is a
powerful and versatile algorithm known for its ability to handle complex
decision boundaries and high-dimensional data. SVM works by finding the
*optimal hyperplane* that best separates data points into different
classes or predicts a continuous target variable.

Main components of SVM:

- **Hyperplane** - SVM’s core concept is to find the optimal hyperplane
  that maximizes the margin between two classes in a data set. The
  hyperplane is the decision boundary that separates data points into
  different classes. In two dimensions, it’s a line; in higher
  dimensions, it’s a hyperplane.
- **Support Vectors** - Support Vectors are the data points that are
  closest to the decision boundary, or hyperplane. These support vectors
  play a crucial role in determining the position and orientation of the
  hyperplane.
- **Margin** - The margin is the distance between the decision boundary
  (hyperplane) and the closest support vectors. SVM aims to maximize
  this margin, as it represents the separation between classes. The
  larger the margin, the better the model’s generalization.
- **Kernel Trick** - SVM can handle both linearly separable and
  non-linearly separable data. The kernel trick allows SVM to transform
  data into higher-dimensional space, making it possible to find linear
  separation in this transformed space. Common kernel functions include
  linear, polynomial, radial basis function (RBF), and sigmoid.
- **C Parameter** - SVM has a hyper parameter called C, which controls
  the trade-off between maximizing the margin and minimizing the
  classification error. Smaller C values lead to a larger margin but may
  allow some mis-classification, while larger C values lead to a smaller
  margin with fewer mis-classifications.

``` r
train_control = trainControl(
  method = "cv",
  number = 5,
  classProbs =  TRUE,
  summaryFunction=mnLogLoss
)

svm_grid = expand.grid(
  sigma = c(0.01, 0.1, 1),   # Range of sigma values for the RBF kernel
  C = c(0.1, 1, 10)          # Range of C values for regularization
)

# limiting number of features due to performance 
# issues of the algorithm 
svm_model = train(
  #Diabetes_binary_transformed ~ ., 
  Diabetes_binary_transformed ~ HighChol+
                                BMI + 
                                GenHlth,
  data = dplyr::select(train_data, -Diabetes_binary),
  method = "svmRadial",
  trControl = train_control,
  metric="logLoss",
  tuneGrid = svm_grid
)
```

    ## maximum number of iterations reached 0.0007706485 0.0007620639maximum number of iterations reached 0.001245194 0.001239285maximum number of iterations reached 0.0007402059 0.0007376888

``` r
svm_model
```

    ## Support Vector Machines with Radial Basis Function Kernel 
    ## 
    ## 500 samples
    ##   3 predictor
    ##   2 classes: 'X0', 'X1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 400, 400, 401, 400, 399 
    ## Resampling results across tuning parameters:
    ## 
    ##   sigma  C     logLoss  
    ##   0.01    0.1  0.4547899
    ##   0.01    1.0  0.4359948
    ##   0.01   10.0  0.4417403
    ##   0.10    0.1  0.4373695
    ##   0.10    1.0  0.4342837
    ##   0.10   10.0  0.4395919
    ##   1.00    0.1  0.4323844
    ##   1.00    1.0  0.4289065
    ##   1.00   10.0  0.4235962
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final values used for the model were sigma = 1 and C = 10.

``` r
plot(svm_model)
```

![](Education_level_5_report_files/figure-gfm/unnamed-chunk-47-1.png)<!-- -->

``` r
svm_model$bestTune
```

    ##   sigma  C
    ## 9     1 10

Calculate log loss for train data set

``` r
train_predictions = predict(svm_model, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_svm = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_svm))
```

    ## [1] "Log Loss: 1.70609806421099"

``` r
models_performace_train[["svm"]] = log_loss_train_svm
```

Calculate log loss for validation data set

``` r
val_predictions = predict(svm_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_svm = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_svm))
```

    ## [1] "Log Loss: 1.66550790905258"

``` r
models_performace_val[["svm"]] = log_loss_val_svm
```

## 5.7 New model - Naive Bayes

``` r
train_control = trainControl(
  method = "cv",
  number = 5,
  summaryFunction=mnLogLoss,
  classProbs =  TRUE
)

nb_grid <- expand.grid(
  usekernel = TRUE,
  fL = seq(0,1, by = 0.5),
  adjust = seq(0.5, 1.5, by = 0.5)
)

nb_model = train(
  Diabetes_binary_transformed ~ ., 
  data = dplyr::select(train_data, -MentHlth, -PhysHlth, -Diabetes_binary),
  method = "nb",
  trControl = train_control,
  tuneGrid = nb_grid,
  metric="logLoss"
)

nb_model$results
```

    ##   usekernel  fL adjust  logLoss logLossSD
    ## 1      TRUE 0.0    0.5 2.902079 0.2170535
    ## 2      TRUE 0.0    1.0 2.906638 0.2138678
    ## 3      TRUE 0.0    1.5 2.922908 0.1913977
    ## 4      TRUE 0.5    0.5 2.902079 0.2170535
    ## 5      TRUE 0.5    1.0 2.906638 0.2138678
    ## 6      TRUE 0.5    1.5 2.922908 0.1913977
    ## 7      TRUE 1.0    0.5 2.902079 0.2170535
    ## 8      TRUE 1.0    1.0 2.906638 0.2138678
    ## 9      TRUE 1.0    1.5 2.922908 0.1913977

``` r
nb_model
```

    ## Naive Bayes 
    ## 
    ## 500 samples
    ##  19 predictor
    ##   2 classes: 'X0', 'X1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 400, 400, 401, 399, 400 
    ## Resampling results across tuning parameters:
    ## 
    ##   fL   adjust  logLoss 
    ##   0.0  0.5     2.902079
    ##   0.0  1.0     2.906638
    ##   0.0  1.5     2.922908
    ##   0.5  0.5     2.902079
    ##   0.5  1.0     2.906638
    ##   0.5  1.5     2.922908
    ##   1.0  0.5     2.902079
    ##   1.0  1.0     2.906638
    ##   1.0  1.5     2.922908
    ## 
    ## Tuning parameter 'usekernel' was held constant at a value of TRUE
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final values used for the model were fL = 0, usekernel = TRUE and adjust = 0.5.

Calculate log loss for train data set

``` r
train_predictions = predict(nb_model, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_nb = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_train_nb))
```

    ## [1] "Log Loss: 17.859301807168"

``` r
models_performace_train[["Naive Bayes"]] = log_loss_train_nb
```

Calculate log loss for validation data set

``` r
val_predictions = predict(nb_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_nb = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss:", log_loss_val_nb))
```

    ## [1] "Log Loss: 17.2927931373262"

``` r
models_performace_val[["Naive Bayes"]] = log_loss_val_nb
```

``` r
models_performace_train
```

    ## $logistic_regression_model_1
    ## [1] 2.068239
    ## 
    ## $logistic_regression_model_2
    ## [1] 2.083782
    ## 
    ## $logistic_regression_model_3
    ## [1] 4.6606
    ## 
    ## $lasso
    ## [1] 1.580797
    ## 
    ## $classification_tree
    ## [1] 1.580797
    ## 
    ## $random_forest
    ## [1] 14.23873
    ## 
    ## $svm
    ## [1] 1.706098
    ## 
    ## $`Naive Bayes`
    ## [1] 17.8593

``` r
models_performace_val
```

    ## $logistic_regression_model_1
    ## [1] 2.006917
    ## 
    ## $logistic_regression_model_2
    ## [1] 2.001511
    ## 
    ## $logistic_regression_model_3
    ## [1] 4.214784
    ## 
    ## $lasso
    ## [1] 1.552353
    ## 
    ## $classification_tree
    ## [1] 1.552353
    ## 
    ## $random_forest
    ## [1] 11.46695
    ## 
    ## $svm
    ## [1] 1.665508
    ## 
    ## $`Naive Bayes`
    ## [1] 17.29279

The best performed model based on train data set is

``` r
print (names(models_performace_train)[which.min(unlist(models_performace_train))])
```

    ## [1] "classification_tree"

The best performed model based on validation data set is

``` r
print (names(models_performace_val)[which.min(unlist(models_performace_val))])
```

    ## [1] "classification_tree"

# 6 Summary
