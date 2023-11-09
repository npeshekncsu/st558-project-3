library(rmarkdown)

params_list = list("12", "3", "4", "5", "6")

for (level in params_list) {
  params = list(education_level = level)
  render(
    input = "Readme.Rmd",
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