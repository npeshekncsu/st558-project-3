*Authors: Jacob Press, Nataliya Peshekhodko*

# Goal

The purpose of this repository is to automate the creation of documents using R Markdown. Each document is built for a specific education level, as determined by [EDUCA](https://www.icpsr.umich.edu/web/NAHDAP/studies/34085/datasets/0001/variables/EDUCA?archive=NAHDAP). **Level 1** (Never attended school or only kindergarten) and **Level 2** (Grades 1 - 8) are combined. Each document contains explanatory data analysis for a subset of the data related to the selected education level and includes several classification models built for **Diabetes_binary** prediction. The best-performing model is chosen based on the validation dataset.


# Used packages

  - `tidyverse` - is a collection of R packages, required for data transformation and manipulation
  - `caret` - required for training and evaluating machine learning models
  - `ggplot2` - required for for creating data visualizations and graphics
  - `corrplot` - required for correlation matrix visualizing


# Render code

The code used to create the analyses from a single .Rmd file:

```
library(rmarkdown)
params_list = list("12", "3", "4", "5", "6")
for (level in params_list) {
  params = list(education_level = level)
  render(
    input = "Report.Rmd",
    output_file = paste0("Education_level_", level, "_report.md"),
    runtime = "static",
    params = params,
    output_format = "github_document",
    output_options = list(
      toc = TRUE,
      toc_depth = 4,
      number_sections = TRUE
    )
  )
}
```

# Reports

Reports for specified education levels:

  - Never attended school or only kindergarten or Grades 1 - 8 (Elementary) [Report](Education_level_12_report.md)
  - Grades 9 - 11 (Some high school) [Report](Education_level_3_report.md)
  - Grade 12 or GED (High school graduate) [Report](Education_level_4_report.md)
  - College 1 year to 3 years (Some college or technical school) [Report](Education_level_5_report.md)
  - College 4 years or more (College graduate) [Report](Education_level_6_report.md)