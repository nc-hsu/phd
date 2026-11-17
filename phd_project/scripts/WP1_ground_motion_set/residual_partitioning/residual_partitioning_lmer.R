# Script for partitioning the residuals 
# --- Reproducibility Block: Package Management ---
# The add-on packages this script uses, and the functions it takes from each:
#   dplyr    - %>%, filter, group_by, n, ungroup, select, all_of, everything, left_join,
#              and the .data pronoun (%>%, all_of, everything and .data are re-exported
#              by dplyr from magrittr, tidyselect and rlang)
#   readr    - write_csv
#   stringr  - str_split_i, str_replace
#   lme4     - lmer, ranef, fixef, VarCorr, isSingular, and lme4's methods for the
#              stats generics sigma and residuals
#   jsonlite - write_json
# Everything else (read.csv, as.factor, as.formula, paste, ...) is base R.
# Only these packages are loaded, so the pinned set in the repo's renv.lock stays small.
required_packages <- c("dplyr", "readr", "stringr", "lme4", "jsonlite")

# The packages are never installed here. They come from renv.lock: renv::restore() in the
# repo root installs the pinned versions into the project library, and the repo's
# .Rprofile makes that library active whenever R starts in the repo root (which is where
# run_r_script starts Rscript). Installing at run time would fetch whatever versions are
# newest, so a missing package stops the script with instructions instead.
# find.package(..., quiet = TRUE) returns an empty result for a package that is not
# installed, without loading anything.
missing_packages <- required_packages[
  vapply(required_packages, function(p) length(find.package(p, quiet = TRUE)) == 0,
         logical(1))]
if (length(missing_packages) > 0) {
  stop("Required R package(s) not installed: ", paste(missing_packages, collapse = ", "),
       ".\nInstall the pinned versions by running renv::restore() in R, started in the ",
       "repository root (or run `python install.py`). This script does not install ",
       "packages itself.", call. = FALSE)
}

# Load the packages. Their start-up messages (attach notes, masking notes) are suppressed
# so that R's stderr, which run_r_script prints, holds only real warnings such as lme4's
# singular-fit or convergence messages.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(lme4)
  library(jsonlite)
})

# --- Command Line Argument Logic ---
# Capture arguments from Python
args <- commandArgs(trailingOnly = TRUE)

# Expected order of arguments:
# 1. residuals_folder: an absolute path as a string
# 2. model_tags: comma-separated string.
# 3. residuals_file_tail: the tail of the filename
# 4. output_csv_tail
# 5. output_json_tail
# 6. n_records_min

# file names follow the pattern: "model_tag"+"file_tail"

# 2. Check if arguments were provided
if (length(args) > 0) {
  # Pipeline Mode: Use passed parameters
  print("Running in Pipeline Mode (Arguments detected)")
  
  residuals_folder    <- args[1]
  model_tags          <- unlist(strsplit(args[2], ","))
  residuals_file_tail <- args[3]
  output_csv_tail     <- args[4]
  output_json_tail    <- args[5]
  n_records_min       <- as.integer(args[6])
  
} else {
  # Standalone Mode: defaults for running the script by hand. The folder is relative to
  # the repository root, so start R there. In the public repo that is also where the
  # pinned packages are available: its .Rprofile only activates the renv project library
  # when R starts in the repo root. file.path() joins with "/", which R accepts on every
  # platform.
  print("Running in Standalone Mode (Using defaults relative to the repo root)")
  
  residuals_folder <- file.path("data_processed", "02_im_correlation_model",
                                "reverse", "residual_partitioning")
  model_tags <-  c("asc", "sslab", "sinter", "vran")
  
  residuals_file_tail <- "_reverse_total_residuals.csv"
  output_csv_tail     <- "_reverse_partitioned_residuals_lmer.csv"
  output_json_tail    <- "_reverse_partitioned_residuals_summary_lmer.json"
  n_records_min <- 3
}

# --- Residual Partitioning ---
# loop through the different models and partition the residuals

for (model in model_tags) {
  print(paste0("Partitioning Residuals for model: ", model))
  # file.path() joins with "/", which works on Windows, Linux and macOS alike.
  data_output_file <- file.path(residuals_folder, paste0(model, output_csv_tail))
  json_results_file <- file.path(residuals_folder, paste0(model, output_json_tail))
  
  # load the data and periods
  residuals_file <- file.path(residuals_folder, paste0(model, residuals_file_tail))
  # check.names = FALSE keeps the IM column names exactly as Python wrote them (e.g.
  # "rotD50_AvgSA([0, 3])_3.0_Delta"); R would otherwise rewrite the brackets, comma and
  # space as dots. The unnamed first column (the row index) is then read as "" rather than
  # "X", so both are renamed to "index".
  df <- read.csv(file=residuals_file, check.names = FALSE)
  names(df)[colnames(df) %in% c("", "X")] <- "index"
  
  # make the event_id and station_code factors (i.e. categorical data)
  df$event_id <- as.factor(df$event_id)
  df$station_code <- as.factor(df$station_code)
  
  # Extract the names of the im columns only
  im_columns <- names(df)
  im_columns <- im_columns[im_columns!="index"]
  im_columns <- im_columns[im_columns!="event_id"]
  im_columns <- im_columns[im_columns!="station_code"]
  im_columns <- im_columns[im_columns!="max_usable_T"]

  # extract the periods from the im column headers
  periods <- im_columns %>%
    str_split_i("_", 3) %>%      # Extract 3rd component directly
    str_replace("None", "0") %>%
    as.numeric()             # Convert to numeric (coerces "None" to NA)

  n_ims <- length(im_columns) 
  summary_results <- list() # initialise the results list
  for (im_i in 1:n_ims){
    # get column name and period
    im <- paste(head(strsplit(im_columns[im_i], "_")[[1]], -1), collapse="_")
    t_im <- periods[im_i]
    
    # filter based on the max usable period and number of records
    if (model == "asc"){ 
      usable_data <- df %>% 
        filter(.data$max_usable_T >= t_im) %>% 
        group_by(event_id) %>%  # group to check the number of event recordings
        filter(n() >= n_records_min) %>% # remove events with less than 3 recordings 
        ungroup()
    } else {
      usable_data <- df %>% 
        filter(.data$max_usable_T >= t_im)
    }
    
    # do the residual partitioning
    # Formula: "im ~ 1" tells the model that the fixed effects have already.
    # been removed from the data and that we are only interested in the random
    # effects. "(1| ...)" indicates that "..." is a random effect that affects the intercept.
    # "(1| A)" + "(1|B) indicates two uncorrelated random effects
    # The response is wrapped in backticks because the IM column names are not
    # syntactic R names (they contain brackets, commas and spaces).
    formula = paste0("`", im_columns[im_i], "` ~ 1 + (1|event_id) + (1|station_code)")
    formula <- as.formula(formula)
    fit <- lmer(formula, usable_data)
    
    # get the random effects
    random_effects <- ranef(fit)
    dBe <- random_effects$event_id
    dS2Ss <- random_effects$station_code
    
    # extract some results and store
    current_results <- list(
      intercept = as.numeric(fixef(fit)["(Intercept)"]),
      mean_dBe = mean(dBe$`(Intercept)`),
      tau = as.numeric(attr(VarCorr(fit)$event_id, "stddev")),
      mean_dS2Ss = mean(dS2Ss$`(Intercept)`),
      phi_S2S = as.numeric(attr(VarCorr(fit)$station_code, "stddev")),
      mean_dWSes = mean(residuals(fit)),
      phi_ss = sigma(fit),
      n_records = length(usable_data$index),
      n_events = length(unique(usable_data$event_id)),
      n_stations = length(unique(usable_data$station_code)),
      # Fit diagnostics, kept with the results so a doubtful fit stays visible later:
      #   singular - TRUE if a variance component is estimated at (or near) zero, which
      #              lme4 reports as "boundary (singular) fit";
      #   convergence_messages - lme4's convergence checks and the optimiser's warnings,
      #              joined into one string ("" when there are none).
      singular = isSingular(fit),
      convergence_messages = paste(c(fit@optinfo$conv$lme4$messages,
                                     unlist(fit@optinfo$warnings)), collapse = "; ")
    )
    summary_results[[im]] <- current_results
    
    # add the random effects as new columns
    dB_cname <- paste(im, "dBe", sep="_")
    dS2S_cname <- paste(im, "dS2Ss", sep="_")
    dW_cname <- paste(im, "dWSes", sep="_")
    
    usable_data[[dB_cname]] <- dBe[match(usable_data$event_id, rownames(dBe)), "(Intercept)"]
    usable_data[[dS2S_cname]] <- dS2Ss[match(usable_data$station_code, rownames(dS2Ss)), "(Intercept)"]
    usable_data[[dW_cname]] <- residuals(fit)
    
    # merge with the original data frame using a left merge
    residuals_only <- usable_data %>% 
      select(index, all_of(c(dB_cname, dS2S_cname, dW_cname)))
    df <- df %>% 
      left_join(residuals_only, by="index")
  }
  
  # sort the data frame and save
  df <- df %>% 
    select(sort(colnames(df))) %>% 
    select(index, event_id, station_code, max_usable_T, everything())
  write_csv(df, data_output_file)
  write_json(summary_results, json_results_file, pretty = TRUE, auto_unbox = TRUE)
}