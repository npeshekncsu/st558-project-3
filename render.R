#library(rmarkdown)

#params_list = list(education_level = "12", "3")

#render(input="Readme.Rmd", 
#       output_file = "README.md",
#       runtime = "static",
#       params = params_list,
#       output_format = "github_document",
#       output_options = list(
#         toc = TRUE, 
#         toc_depth = 4,
#         number_sections = TRUE))




library(rmarkdown)

params_list = list("12", "3")


for (education_level in params_list) {

  params = list(education_level = education_level)
  
  render(
    input = "Readme.Rmd",
    output_file = paste0("README_", education_level, ".md"),
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

  
