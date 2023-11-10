ST558, Project3
================
Jacob Press, Nataliya Peshekhodko
2023-11-10

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
    - <a href="#524-the-best-performed-logistic-regression-model"
      id="toc-524-the-best-performed-logistic-regression-model">5.2.4 The best
      performed logistic regression model</a>
  - <a href="#53-lasso-logistic-regression"
    id="toc-53-lasso-logistic-regression">5.3 LASSO logistic regression</a>
    - <a href="#531-fit-and-validate-lasso-logistic-regression"
      id="toc-531-fit-and-validate-lasso-logistic-regression">5.3.1 Fit and
      validate LASSO logistic regression</a>
  - <a href="#54-classification-tree-model"
    id="toc-54-classification-tree-model">5.4 Classification tree model</a>
    - <a href="#541-fit-and-validate-classification-tree-model"
      id="toc-541-fit-and-validate-classification-tree-model">5.4.1 Fit and
      validate classification tree model</a>
  - <a href="#55-random-forest-model" id="toc-55-random-forest-model">5.5
    Random forest model</a>
    - <a href="#551-fit-and-validate-random-forest-model"
      id="toc-551-fit-and-validate-random-forest-model">5.5.1 Fit and validate
      random forest model</a>
  - <a href="#56-new-model---support-vector-machine"
    id="toc-56-new-model---support-vector-machine">5.6 New model - Support
    Vector Machine</a>
    - <a href="#561-fit-and-validate-support-vector-machine-model"
      id="toc-561-fit-and-validate-support-vector-machine-model">5.6.1 Fit and
      validate support vector machine model</a>
  - <a href="#57-new-model---naive-bayes"
    id="toc-57-new-model---naive-bayes">5.7 New model - Naive Bayes</a>
    - <a href="#571-fit-and-validate-naive-bayes-model"
      id="toc-571-fit-and-validate-naive-bayes-model">5.7.1 Fit and validate
      Naive Bayes model</a>
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

This report is built for education level **College 4 years or more
(College graduate)**.

# 2 Packages

In order to achieve our goals, we will be using the following `R`
packages.

``` r
library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
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
    ##   Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies HvyAlcoholConsump
    ##             <dbl>  <dbl>    <dbl>     <dbl> <dbl>  <dbl>  <dbl>                <dbl>        <dbl>  <dbl>   <dbl>             <dbl>
    ## 1               0      1        1         1    40      1      0                    0            0      0       1                 0
    ## 2               0      0        0         0    25      1      0                    0            1      0       0                 0
    ## 3               0      1        1         1    28      0      0                    0            0      1       0                 0
    ## 4               0      1        0         1    27      0      0                    0            1      1       1                 0
    ## 5               0      1        1         1    24      0      0                    0            1      1       1                 0
    ## 6               0      1        1         1    25      1      0                    0            1      1       1                 0
    ## # ℹ 10 more variables: AnyHealthcare <dbl>, NoDocbcCost <dbl>, GenHlth <dbl>, MentHlth <dbl>, PhysHlth <dbl>, DiffWalk <dbl>,
    ## #   Sex <dbl>, Age <dbl>, Education <dbl>, Income <dbl>

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

    ## tibble [107,325 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : num [1:107325] 0 0 0 1 0 0 0 0 0 1 ...
    ##  $ HighBP              : num [1:107325] 0 1 1 0 0 1 0 0 0 1 ...
    ##  $ HighChol            : num [1:107325] 0 1 0 0 1 0 0 1 0 1 ...
    ##  $ CholCheck           : num [1:107325] 0 1 1 1 1 1 0 1 1 1 ...
    ##  $ BMI                 : num [1:107325] 25 25 30 25 33 33 23 28 32 37 ...
    ##  $ Smoker              : num [1:107325] 1 1 1 1 1 0 0 0 0 1 ...
    ##  $ Stroke              : num [1:107325] 0 0 0 0 1 0 0 0 0 1 ...
    ##  $ HeartDiseaseorAttack: num [1:107325] 0 0 0 0 0 0 0 0 0 1 ...
    ##  $ PhysActivity        : num [1:107325] 1 1 0 1 1 1 0 0 1 0 ...
    ##  $ Fruits              : num [1:107325] 0 1 0 1 0 0 0 0 1 0 ...
    ##  $ Veggies             : num [1:107325] 0 1 0 1 1 0 1 0 1 1 ...
    ##  $ HvyAlcoholConsump   : num [1:107325] 0 0 0 0 0 0 0 1 0 0 ...
    ##  $ AnyHealthcare       : num [1:107325] 0 1 1 1 1 1 1 1 1 1 ...
    ##  $ NoDocbcCost         : num [1:107325] 1 0 0 0 1 0 0 0 0 0 ...
    ##  $ GenHlth             : num [1:107325] 3 2 3 3 4 2 2 2 2 5 ...
    ##  $ MentHlth            : num [1:107325] 0 0 0 0 30 5 15 10 0 0 ...
    ##  $ PhysHlth            : num [1:107325] 0 2 14 0 28 0 0 0 0 0 ...
    ##  $ DiffWalk            : num [1:107325] 0 0 0 0 0 0 0 0 0 1 ...
    ##  $ Sex                 : num [1:107325] 0 1 0 1 0 0 0 1 0 1 ...
    ##  $ Age                 : num [1:107325] 7 10 9 13 4 6 2 4 5 10 ...
    ##  $ Education           : num [1:107325] 6 6 6 6 6 6 6 6 6 6 ...
    ##  $ Income              : num [1:107325] 1 8 7 8 2 8 7 8 8 5 ...

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

First, let’s look at number of the records with Diabetes and without
Diabetes for the selected education level:

``` r
table (factor (subset$Diabetes_binary, labels = c("No diabet", "Diabet")) )
```

    ## 
    ## No diabet    Diabet 
    ##     96925     10400

Let’s look at `Age` distribution for the selected education level and
check if all age groups are presented equally in the subset of data.

``` r
ggplot(data = subset, aes(x = Age)) +
  geom_histogram(color = "black", fill = 'brown') +
  labs(title = "Histogram of Age groups distribution", 
       x = "Age group", 
       y = "Frequency")
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Let’s look at number of cases with Diabetes and without Diabetes for
each age group for the selected education level.

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
    ##             Age 18 - 24 Age 25 to 29 Age 30 to 34 Age 35 to 39 Age 40 to 44 Age 45 to 49 Age 50 to 54 Age 55 to 59 Age 60 to 64
    ##   No diabet        1371         3766         5504         6863         7905         9029        10100        11171        12433
    ##   Diabet             15           40          103          179          337          486          882         1120         1742
    ##            
    ##             Age 65 to 69 Age 70 to 74 Age 75 to 79 Age 80 or older
    ##   No diabet        11850         7648         4516            4769
    ##   Diabet            2163         1587          905             841

Let’s check if number of cases with Diabetes and without Diabetes are
equal for males and females for the selected subset of data.

``` r
table(factor (subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$Sex, labels = c("Female", "Male")))
```

    ##            
    ##             Female  Male
    ##   No diabet  53299 43626
    ##   Diabet      4538  5862

Linear correlation between numeric variables allows to check which
variables are correlated with target variable `Diabetes_binary` and
could be used as predictors in the models.

``` r
corrplot(cor(as.matrix(subset %>% dplyr::select(-Education))), 
         type="upper", 
         tl.pos = "lt")
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Number of cases with Diabetes and without Diabetes for each general
health level.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$GenHlth, labels = c("Excellent", "Very good", 
                                        "Good", "Fair", "Poor")) )
```

    ##            
    ##             Excellent Very good  Good  Fair  Poor
    ##   No diabet     26023     41480 22751  5274  1397
    ##   Diabet          463      2626  4343  2146   822

Number of cases with Diabetes and without Diabetes for high blood
pressure and normal blood pressure patients.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor(subset$HighBP, labels = c("No high BP", "High BP")) )
```

    ##            
    ##             No high BP High BP
    ##   No diabet      66157   30768
    ##   Diabet          2930    7470

Number of cases with Diabetes and without Diabetes for high cholesterol
and normal cholesterol patients.

``` r
table(factor(subset$Diabetes_binary, labels = c("No diabet", "Diabet")), 
      factor (subset$HighChol, labels = c("No high chol", "High chol")))
```

    ##            
    ##             No high chol High chol
    ##   No diabet        61767     35158
    ##   Diabet            3604      6796

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

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

Let’s plot number of cases with Diabetes and Without Diabetes for each
General Health level for the selected sunset of data.

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

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

Visualization of the proportion with and without diabetes grouped by Age
for the given education level. It appears the proportion of with
diabetes increases with age, which is not surprising.

``` r
ggplot(subset, aes(x = as.factor(Age), y = 1, fill = as.factor(Diabetes_binary), group = Diabetes_binary)) +
  geom_bar(position = "fill", stat = "identity") +
  labs(
    title = "Number of Cases with Diabetes and Without Diabetes by Age",
    x = "Age",
    y = "Number of Cases"
  ) +
  scale_fill_manual(
    values = c("0" = "grey", "1" = "red"),
    labels = c("0" = "Without Diabetes", "1" = "With Diabetes")
  ) +
  labs(fill = "Diabetes Status")
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

Visualization showing the distribution of `Diabetes_binary` by `BMI`.

``` r
ggplot(subset, aes(x = as.factor(Diabetes_binary), y = BMI, group = Diabetes_binary, fill = as.factor(Diabetes_binary))) +
 labs(
   title = "Violin Plot of Diabetes Status by BMI",
   x = "Diabetes Status",
   y = "BMI Scale"
 ) +
 geom_violin(trim = FALSE) +
 scale_fill_manual(
   values = c("0" = "grey", "1" = "red"),
   labels = c("0" = "Without Diabetes", "1" = "With Diabetes")
   ) +
 labs(fill = "Diabetes Status")
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

Visualization showing the counts of `Diabetes_binary` by `Income` for
the given education level.

``` r
ggplot(subset, aes(x = as.factor(Income), fill = as.factor(Diabetes_binary), group = Diabetes_binary)) +
  geom_bar(position = "stack") +
  labs(
    title = "Number of Cases with Diabetes and Without Diabetes by Income",
    x = "Income",
    y = "Number of Cases"
  ) +
  scale_fill_manual(
    values = c("0" = "grey", "1" = "red"),
    labels = c("0" = "Without Diabetes", "1" = "With Diabetes")
  ) +
  labs(fill = "Diabetes Status")
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

# 5 Modeling

Converting some of the variables to factors and checking dataset
structure.

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

    ## tibble [107,325 × 22] (S3: tbl_df/tbl/data.frame)
    ##  $ Diabetes_binary     : Factor w/ 2 levels "0","1": 1 1 1 2 1 1 1 1 1 2 ...
    ##  $ HighBP              : Factor w/ 2 levels "0","1": 1 2 2 1 1 2 1 1 1 2 ...
    ##  $ HighChol            : Factor w/ 2 levels "0","1": 1 2 1 1 2 1 1 2 1 2 ...
    ##  $ CholCheck           : Factor w/ 2 levels "0","1": 1 2 2 2 2 2 1 2 2 2 ...
    ##  $ BMI                 : num [1:107325] 25 25 30 25 33 33 23 28 32 37 ...
    ##  $ Smoker              : Factor w/ 2 levels "0","1": 2 2 2 2 2 1 1 1 1 2 ...
    ##  $ Stroke              : Factor w/ 2 levels "0","1": 1 1 1 1 2 1 1 1 1 2 ...
    ##  $ HeartDiseaseorAttack: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 2 ...
    ##  $ PhysActivity        : Factor w/ 2 levels "0","1": 2 2 1 2 2 2 1 1 2 1 ...
    ##  $ Fruits              : Factor w/ 2 levels "0","1": 1 2 1 2 1 1 1 1 2 1 ...
    ##  $ Veggies             : Factor w/ 2 levels "0","1": 1 2 1 2 2 1 2 1 2 2 ...
    ##  $ HvyAlcoholConsump   : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 2 1 1 ...
    ##  $ AnyHealthcare       : Factor w/ 2 levels "0","1": 1 2 2 2 2 2 2 2 2 2 ...
    ##  $ NoDocbcCost         : Factor w/ 2 levels "0","1": 2 1 1 1 2 1 1 1 1 1 ...
    ##  $ GenHlth             : Factor w/ 5 levels "1","2","3","4",..: 3 2 3 3 4 2 2 2 2 5 ...
    ##  $ MentHlth            : Factor w/ 31 levels "0","1","2","3",..: 1 1 1 1 31 6 16 11 1 1 ...
    ##  $ PhysHlth            : Factor w/ 31 levels "0","1","2","3",..: 1 3 15 1 29 1 1 1 1 1 ...
    ##  $ DiffWalk            : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 2 ...
    ##  $ Sex                 : Factor w/ 2 levels "0","1": 1 2 1 2 1 1 1 2 1 2 ...
    ##  $ Age                 : Factor w/ 13 levels "1","2","3","4",..: 7 10 9 13 4 6 2 4 5 10 ...
    ##  $ Education           : num [1:107325] 6 6 6 6 6 6 6 6 6 6 ...
    ##  $ Income              : Factor w/ 8 levels "1","2","3","4",..: 1 8 7 8 2 8 7 8 8 5 ...

Splitting up data into training and validation datasets.

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
#train_data <- train_data[sample(nrow(train_data), size = 500), ]
#val_data <- val_data[sample(nrow(val_data), size = 200), ]
```

## 5.1 Log loss

**Log loss**, also known as **logarithmic loss** or **cross-entropy
loss**, is a common evaluation metric for binary classification models.
It measures the performance of a model by quantifying the difference
between predicted probabilities and actual values. Log-loss is
indicative of how close the prediction probability is to the
corresponding actual/true value, penalizing inaccurate predictions with
higher values. **Lower log-loss** indicates **better** model
performance.

Mathematical interpretation: Log Loss is the negative average of the log
of corrected predicted probabilities for each instance.

$$log \ loss = -\frac{1}{N} \sum_{i=1}^N y_i log(p(y_i)) + (1-y_i)log(1-p(y_i))$$

$p(y_i)$ is the probability of $1$

$1-p(y_i)$ is the probability of 0

$y_i$ is the true binary outcome

We may prefer `log loss` to things like `accuracy` for several reasons:

- **Probabilistic Evaluation** - Log loss considers probabilities, while
  accuracy only looks at final decisions
- **Handles Imbalanced Data** - Log loss shows poor performance in
  imbalanced datasets
- **Fair Model Comparison** - Log loss enables fair model comparisons
  and makes it easier to evaluate which model is performing better

## 5.2 Logistic regression

**Logistic regression** is a statistical and machine learning model used
for binary classification tasks. It’s a type of regression analysis
that’s well-suited for predicting the probability of an observation
belonging to one of two classes or categories.

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

Creating lists to store model performances on train and validations data
sets.

``` r
models_performace_train = list()
models_performace_val = list()
logistic_regression_train = list()
```

### 5.2.1 Fit Logistic regression model 1

Before we can fit logistic regression models, we need to transform
response variable to the format `train` function expects using function
`make.names`.

``` r
train_data$Diabetes_binary_transformed = train_data$Diabetes_binary
val_data$Diabetes_binary_transformed = val_data$Diabetes_binary

levels(train_data$Diabetes_binary_transformed) = make.names(levels(train_data$Diabetes_binary_transformed))
levels(val_data$Diabetes_binary_transformed) = make.names(levels(val_data$Diabetes_binary_transformed))
```

Fit logistic regression model with `HighChol`, `BMI` and `GenHlth` as
predictors.

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
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -5.800363   0.074673  -77.68   <2e-16 ***
    ## HighChol1    0.931606   0.027334   34.08   <2e-16 ***
    ## BMI          0.055514   0.001722   32.24   <2e-16 ***
    ## GenHlth2     1.024228   0.060825   16.84   <2e-16 ***
    ## GenHlth3     1.950840   0.059608   32.73   <2e-16 ***
    ## GenHlth4     2.611159   0.064570   40.44   <2e-16 ***
    ## GenHlth5     3.059123   0.078687   38.88   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 47815  on 75127  degrees of freedom
    ## Residual deviance: 39920  on 75121  degrees of freedom
    ## AIC: 39934
    ## 
    ## Number of Fisher Scoring iterations: 6

Obtain `log loss` for train data set for logistic regression model \#1.

``` r
logistic_regression_train[['logistic_regression_model_1']] = lr_model_1$results$logLoss
print(paste("Obtained Log loss for for logistic regression model #1 on train dataset", 
            logistic_regression_train[['logistic_regression_model_1']]))
```

    ## [1] "Obtained Log loss for for logistic regression model #1 on train dataset 0.265779618703035"

### 5.2.2 Fit Logistic regression model 2

Fit logistic regression with second order `BMI`, `HighChol`,
`HeartDiseaseorAttack` and interaction between `HighChol`and
`HeartDiseaseorAttack` as predictors.

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
    ##                                    Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept)                        -3.10693    0.02349 -132.284  < 2e-16 ***
    ## `poly(BMI, 2)1`                   163.07823    3.32843   48.995  < 2e-16 ***
    ## `poly(BMI, 2)2`                   -96.20752    4.33402  -22.198  < 2e-16 ***
    ## HighChol1                           1.04706    0.02895   36.172  < 2e-16 ***
    ## HeartDiseaseorAttack1               1.62634    0.06912   23.530  < 2e-16 ***
    ## `HighChol1:HeartDiseaseorAttack1`  -0.65086    0.08106   -8.029 9.79e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 47815  on 75127  degrees of freedom
    ## Residual deviance: 41578  on 75122  degrees of freedom
    ## AIC: 41590
    ## 
    ## Number of Fisher Scoring iterations: 6

Obtain `log loss` for train data set for logistic regression model \#2.

``` r
logistic_regression_train[['logistic_regression_model_2']] = lr_model_2$results$logLoss
print(paste("Obtained Log loss for for logistic regression model #2 on train dataset", 
            logistic_regression_train[['logistic_regression_model_2']]))
```

    ## [1] "Obtained Log loss for for logistic regression model #2 on train dataset 0.276794113326796"

### 5.2.3 Fit Logistic regression model 3

Fit logistic regression with `Age`, `GenHlth`, `HighBP`,
`HeartDiseaseorAttack` and second order `BMI` as predictors.

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
    ## (Intercept)           -5.265e+00  2.941e-01 -17.902  < 2e-16 ***
    ## Income2                4.564e-02  1.315e-01   0.347  0.72854    
    ## Income3               -9.102e-04  1.230e-01  -0.007  0.99410    
    ## Income4               -8.991e-02  1.169e-01  -0.769  0.44188    
    ## Income5               -6.770e-02  1.104e-01  -0.613  0.53967    
    ## Income6               -1.544e-01  1.067e-01  -1.447  0.14803    
    ## Income7               -1.697e-01  1.050e-01  -1.616  0.10620    
    ## Income8               -2.901e-01  1.028e-01  -2.823  0.00475 ** 
    ## Age2                  -4.758e-01  3.360e-01  -1.416  0.15682    
    ## Age3                   6.355e-02  2.982e-01   0.213  0.83123    
    ## Age4                   2.012e-01  2.899e-01   0.694  0.48776    
    ## Age5                   7.800e-01  2.818e-01   2.768  0.00565 ** 
    ## Age6                   8.492e-01  2.796e-01   3.037  0.00239 ** 
    ## Age7                   1.290e+00  2.772e-01   4.654 3.26e-06 ***
    ## Age8                   1.299e+00  2.766e-01   4.696 2.65e-06 ***
    ## Age9                   1.584e+00  2.756e-01   5.747 9.10e-09 ***
    ## Age10                  1.786e+00  2.754e-01   6.485 8.90e-11 ***
    ## Age11                  1.914e+00  2.761e-01   6.931 4.19e-12 ***
    ## Age12                  1.879e+00  2.777e-01   6.764 1.35e-11 ***
    ## Age13                  1.720e+00  2.781e-01   6.185 6.21e-10 ***
    ## GenHlth2               8.519e-01  6.136e-02  13.885  < 2e-16 ***
    ## GenHlth3               1.601e+00  6.081e-02  26.322  < 2e-16 ***
    ## GenHlth4               2.106e+00  6.700e-02  31.429  < 2e-16 ***
    ## GenHlth5               2.495e+00  8.320e-02  29.982  < 2e-16 ***
    ## HighBP1                7.999e-01  3.048e-02  26.245  < 2e-16 ***
    ## HeartDiseaseorAttack1  4.259e-01  3.942e-02  10.806  < 2e-16 ***
    ## `poly(BMI, 2)1`        1.224e+02  3.693e+00  33.156  < 2e-16 ***
    ## `poly(BMI, 2)2`       -7.405e+01  4.807e+00 -15.404  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 47815  on 75127  degrees of freedom
    ## Residual deviance: 37277  on 75100  degrees of freedom
    ## AIC: 37333
    ## 
    ## Number of Fisher Scoring iterations: 7

Obtain `log loss` for train data set for logistic regression model \#3.

``` r
logistic_regression_train[['logistic_regression_model_3']] = lr_model_3$results$logLoss
print(paste("Obtained Log loss for for logistic regression model #3 on train dataset", 
            logistic_regression_train[['logistic_regression_model_3']]))
```

    ## [1] "Obtained Log loss for for logistic regression model #3 on train dataset 0.248587454361548"

### 5.2.4 The best performed logistic regression model

Now we can choose the best performed model based on train dataset
performance among logistic regression models.

``` r
best_lr_model = names(logistic_regression_train)[which.min(unlist(logistic_regression_train))]
```

The best performed logistic regression model is model
**logistic_regression_model_3**.

Save the logistic regression model performance on training dataset.

``` r
if (best_lr_model == "logistic_regression_model_1") {
  lr_model = lr_model_1
} else if (best_lr_model == "logistic_regression_model_2") {
  lr_model = lr_model_2
} else {
  lr_model = lr_model_3
}

models_performace_train[['logistic_regression_model']] = lr_model$results$logLoss

print(paste("Log Loss for logistic regression model for training dataset:", 
            models_performace_train[['logistic_regression_model']]))
```

    ## [1] "Log Loss for logistic regression model for training dataset: 0.248587454361548"

In order to obtain `log loss` metric for validation dataset, let’s
create custom function for log loss calculation.

``` r
calculateLogLoss <- function(predicted_probabilities, true_labels) {
  predicted_probabilities = pmax(pmin(predicted_probabilities, 1 - 1e-15), 
                                 1e-15)

  log_loss <- -mean(true_labels * log(predicted_probabilities) + 
                      (1 - true_labels) * log(1 - predicted_probabilities))
  return(log_loss)
}
```

Calculate and save logistic regression model performance on validation
dataset.

``` r
val_predictions = predict(lr_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_lr_model = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss for logistic regression model for validation dataset:", 
            log_loss_val_lr_model))
```

    ## [1] "Log Loss for logistic regression model for validation dataset: 2.99131012812533"

``` r
models_performace_val[["logistic_regression_model"]] = log_loss_val_lr_model
```

## 5.3 LASSO logistic regression

`LASSO (Least Absolute Shrinkage and Selection Operator) logistic regression`
is a statistical method that combines logistic regression with LASSO
regularization. It is used for binary classification problems where you
want to predict the probability of an event occurring based on various
predictor variables.

$$\sum_{i=1}^n(y_i-\sum_{j}x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

- `LASSO logistic regression` models the probability of an event using
  the logistic function. It models the log-odds of the event as a linear
  combination of predictor variables. The logistic function is used to
  transform the linear combination into probabilities.
- `LASSO` adds a regularization term to the logistic regression model.
  The regularization term is a penalty based on the absolute values of
  the model coefficients (L1 regularization). This penalty encourages
  some of the coefficient values to become **exactly zero**, effectively
  performing feature selection.
- `LASSO` regularization promotes sparsity in the model. It can
  automatically select a subset of the most relevant predictor variables
  by setting the coefficients of irrelevant variables to zero. This
  helps to reduce overfitting and build more interpretable models.
- The degree of regularization is controlled by a hyper parameter
  denoted as $\lambda$.

Using Lasso models in logistic regression offers benefits such as
automatic feature selection, better generalization to new data, model
stability, improved interpretability, handling multicollinearity, and
variable importance assessment.

### 5.3.1 Fit and validate LASSO logistic regression

``` r
train.control = trainControl(method = "cv",
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
lasso_log_reg$results
```

    ##   alpha lambda  Accuracy    Kappa  AccuracySD     KappaSD
    ## 1     1   0.00 0.9047933 0.135881 0.001594876 0.009385144
    ## 2     1   0.25 0.9033051 0.000000 0.001725910 0.000000000
    ## 3     1   0.50 0.9033051 0.000000 0.001725910 0.000000000
    ## 4     1   0.75 0.9033051 0.000000 0.001725910 0.000000000
    ## 5     1   1.00 0.9033051 0.000000 0.001725910 0.000000000

Obtained the best tuning parameter $\lambda$ value is

``` r
lasso_log_reg$bestTune$lambda
```

    ## [1] 0

Plot obtained accuracy for different $\lambda$ values.

``` r
plot(lasso_log_reg)
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

Calculate log loss for train data set

``` r
train_predictions = predict(lasso_log_reg, 
                             newdata = train_data %>% dplyr::select(-Diabetes_binary), 
                             type = "prob")

predicted_prob_class1 = train_predictions[, 1]
true_labels = as.integer(as.character(train_data$Diabetes_binary))

log_loss_train_lasso = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss for lasso regression model for train dataset", 
            log_loss_train_lasso))
```

    ## [1] "Log Loss for lasso regression model for train dataset 2.97348543580628"

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
print(paste("Log Loss for lasso regression model for validation dataset", 
            log_loss_val_lasso))
```

    ## [1] "Log Loss for lasso regression model for validation dataset 2.97637118816943"

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

### 5.4.1 Fit and validate classification tree model

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

    ##       cp   logLoss    logLossSD
    ## 1  0.000 0.3043922 1.246431e-02
    ## 2  0.001 0.2806844 3.465780e-03
    ## 3  0.002 0.2806193 2.384719e-03
    ## 4  0.003 0.3110174 1.610988e-02
    ## 5  0.004 0.3182206 7.884718e-06
    ## 6  0.005 0.3182206 7.884718e-06
    ## 7  0.006 0.3182206 7.884718e-06
    ## 8  0.007 0.3182206 7.884718e-06
    ## 9  0.008 0.3182206 7.884718e-06
    ## 10 0.009 0.3182206 7.884718e-06
    ## 11 0.010 0.3182206 7.884718e-06
    ## 12 0.011 0.3182206 7.884718e-06
    ## 13 0.012 0.3182206 7.884718e-06
    ## 14 0.013 0.3182206 7.884718e-06
    ## 15 0.014 0.3182206 7.884718e-06
    ## 16 0.015 0.3182206 7.884718e-06
    ## 17 0.016 0.3182206 7.884718e-06
    ## 18 0.017 0.3182206 7.884718e-06
    ## 19 0.018 0.3182206 7.884718e-06
    ## 20 0.019 0.3182206 7.884718e-06
    ## 21 0.020 0.3182206 7.884718e-06
    ## 22 0.021 0.3182206 7.884718e-06
    ## 23 0.022 0.3182206 7.884718e-06

``` r
plot(tree_model)
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

Obtain log loss for train data set

``` r
models_performace_train[["classification_tree"]] = min(tree_model$results$logLoss)
print(paste("Log Loss for classification tree model for training dataset:", 
            models_performace_train[["classification_tree"]]))
```

    ## [1] "Log Loss for classification tree model for training dataset: 0.280619306677743"

Calculate log loss for validation data set

``` r
val_predictions = predict(tree_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_tree = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss for classification tree for validation dataset", 
            log_loss_val_tree))
```

    ## [1] "Log Loss for classification tree for validation dataset 2.47910604157757"

``` r
models_performace_val[["classification_tree"]] = log_loss_val_tree
```

## 5.5 Random forest model

A Random Forest classification model is a supervised machine learning
model used for classification tasks. It is an ensemble of multiple
decision trees, where each tree predicts the class label of an input
based on a set of features. The final prediction in a Random Forest is
determined through a combination of predictions from individual decision
trees, often using **majority voting** for classification tasks.

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

There are several hyperparameters which could be fine-tuned during
random forest model training. Some which have the biggest affect to
model performance:

- **mtry** - Number of predictor variable randomly selected to be
  sampled at each split of the tree. It controls the level of feature
  randomness in each tree. A smaller mtry may reduce overfitting, while
  a larger mtry can lead to better diversity among trees.
- **ntree** - The number of decision trees (or “trees”) to be grown in
  the forest. Increasing the number of trees can improve the model’s
  accuracy, but it also increases computational cost.

### 5.5.1 Fit and validate random forest model

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
                                HeartDiseaseorAttack+
                                Age+
                                Income,
  data = dplyr::select(train_data, -Diabetes_binary),
  method = "rf",
  metric="logLoss",
  tuneGrid = data.frame(mtry = c(1:4)), 
  trControl = train_control
)

rf_model$results
```

    ##   mtry  logLoss  logLossSD
    ## 1    1 2.445045 0.04582437
    ## 2    2 1.892131 0.03327108
    ## 3    3 1.585629 0.02329645
    ## 4    4 1.408963 0.02809243

``` r
plot(rf_model)
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

Obtain log loss for train data set

``` r
models_performace_train[["random_forest"]] = min(rf_model$results$logLoss)
print(paste("Log Loss for random forest model for training dataset:", models_performace_train[["random_forest"]]))
```

    ## [1] "Log Loss for random forest model for training dataset: 1.40896337382372"

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

    ## [1] "Log Loss: 25.2917197434347"

``` r
models_performace_val[["random_forest"]] = log_loss_val_rf
log_loss_val_rf
```

    ## [1] 25.29172

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

### 5.6.1 Fit and validate support vector machine model

``` r
train_control = trainControl(
  method = "cv",
  number = 5,
  classProbs =  TRUE,
  summaryFunction=mnLogLoss
)

#svm_grid = expand.grid(
#  sigma = c(0.01, 0.1, 1),   # Range of sigma values for the RBF kernel
#  C = c(0.1, 1, 10)          # Range of C values for regularization
#)


svm_grid = expand.grid(
  sigma = c(0.1),   # Range of sigma values for the RBF kernel
  C = c(1, 10)          # Range of C values for regularization
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

    ## line search fails -2.068984 0.2390005 1.937854e-05 6.904634e-06 -1.101632e-07 -1.071605e-07 -2.874707e-12

``` r
svm_model$results
```

    ##   sigma  C   logLoss    logLossSD
    ## 1   0.1  1 0.3085985 0.0008603485
    ## 2   0.1 10 0.3082989 0.0008733599

``` r
plot(svm_model)
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-50-1.png)<!-- -->

``` r
svm_model$bestTune
```

    ##   sigma  C
    ## 2   0.1 10

Obtain log loss for train dataset.

``` r
models_performace_train[["svm"]] = min(svm_model$results$logLoss)
print(paste("Log Loss for support vector machine model for training dataset", 
            models_performace_train[["svm"]]))
```

    ## [1] "Log Loss for support vector machine model for training dataset 0.308298923607785"

Calculate log loss for validation dataset.

``` r
val_predictions = predict(svm_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_svm = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss for support vector machine model for validation dataset", log_loss_val_svm))
```

    ## [1] "Log Loss for support vector machine model for validation dataset 2.16668213769884"

``` r
models_performace_val[["svm"]] = log_loss_val_svm
```

## 5.7 New model - Naive Bayes

The naive Bayes model is a popular supervised machine learning algorithm
used for classification. An advantage of a naive bayes model is it can
require a smaller portion of training data to predict classification.
But naive bayes may be out performed by other models such as boosted
trees and random forests. A break down of Bayes theorem below:

$$ P(c|x) = \frac{P(x|c)*P(c)}{P(x)}$$ Where:

- $P(c|x)$ is the posterior probability of C (class membership) given x
  (predictor).
- $P(c)$ prior probability of class membership.
- $P(x|c)$ is the probability of the predictor given class membership.
- $P(x)$ is the prior probability of the predictor.

### 5.7.1 Fit and validate Naive Bayes model

``` r
train_control = trainControl(
  method = "cv",
  number = 5,
  summaryFunction=mnLogLoss,
  classProbs =  TRUE
)

nb_grid <- expand.grid(
  usekernel = TRUE,
  fL = 1,
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

    ##   usekernel fL adjust  logLoss  logLossSD
    ## 1      TRUE  1    0.5 1.794661 0.01089894
    ## 2      TRUE  1    1.0 1.773013 0.01126460
    ## 3      TRUE  1    1.5 1.773774 0.01116102

``` r
plot(nb_model)
```

![](Education_level_6_report_files/figure-gfm/unnamed-chunk-55-1.png)<!-- -->

Obtain log loss for train dataset.

``` r
models_performace_train[["Naive Bayes"]] = min(nb_model$results$logLoss)
print(paste("Log Loss for naive bayes model for training dataset", 
            models_performace_train[["Naive Bayes"]]))
```

    ## [1] "Log Loss for naive bayes model for training dataset 1.77301336748555"

Calculate log loss for validation dataset.

``` r
val_predictions = predict(nb_model, 
                             newdata = val_data %>% dplyr::select(-Diabetes_binary_transformed), 
                             type = "prob")

predicted_prob_class1 = val_predictions[, 1]
true_labels = as.integer(as.character(val_data$Diabetes_binary))

log_loss_val_nb = calculateLogLoss(predicted_prob_class1, true_labels)
print(paste("Log Loss for naive bayes model for validation dataset", 
            log_loss_val_nb))
```

    ## [1] "Log Loss for naive bayes model for validation dataset 20.0032040616059"

``` r
models_performace_val[["Naive Bayes"]] = log_loss_val_nb
```

All models performance on training dataset based on `log loss` metric:

``` r
models_performace_train
```

    ## $logistic_regression_model
    ## [1] 0.2485875
    ## 
    ## $lasso
    ## [1] 2.973485
    ## 
    ## $classification_tree
    ## [1] 0.2806193
    ## 
    ## $random_forest
    ## [1] 1.408963
    ## 
    ## $svm
    ## [1] 0.3082989
    ## 
    ## $`Naive Bayes`
    ## [1] 1.773013

All models performance on validation dataset based on `log loss` metric:

``` r
models_performace_val
```

    ## $logistic_regression_model
    ## [1] 2.99131
    ## 
    ## $lasso
    ## [1] 2.976371
    ## 
    ## $classification_tree
    ## [1] 2.479106
    ## 
    ## $random_forest
    ## [1] 25.29172
    ## 
    ## $svm
    ## [1] 2.166682
    ## 
    ## $`Naive Bayes`
    ## [1] 20.0032

The best performed model based on train data set is
**logistic_regression_model**.

The best performed model based on validation data set is **svm**.

# 6 Summary

In this report we analyze subset of dataset [Diabetes health indicator
dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/)
for educational level - **College 4 years or more (College graduate)**.
We fit and validate six different machine learning models. Based on
performance on validation dataset the best model is **svm**.
