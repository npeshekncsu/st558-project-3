library(rmarkdown)

#params_list = list("12", "3", "4", "5", "6")
params_list = list("12", "3")

for (education_level in params_list) {

  param = list(education_level = education_level)
  print(education_level)
  
  render(
    input = "Readme.Rmd",
    output_file = paste0("Education_level_", education_level, "_report.md"),
    runtime = "static",
    params = param, 
    output_format = "github_document",
    output_options = list(
      toc = TRUE,
      toc_depth = 4,
      number_sections = TRUE
    )
  )
}

  
