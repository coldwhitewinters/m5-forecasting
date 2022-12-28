library(tidyverse)
library(fpp3)
library(arrow)


input_files <- sprintf("./data/processed/stores/%d/data.parquet", 0:9)
output_dir <- "./R/metrics/"
dir.create(output_dir)

acc_list <- list()
for (i in seq_along(input_files)) {
  data_orig <- read_parquet(input_files[[i]])
  
  data <- data_orig %>%
    select(item_id, store_id, d, sales) %>% 
    as_tsibble(key = c(item_id, store_id), index = d)
  
  acc <- data %>%
    filter(d <= 1913) %>% 
    model(NAIVE(sales)) %>% 
    forecast(h = 28) %>%
    accuracy(data) %>% 
    select(-.model, -.type)
  
  acc_list[[i]] <- acc
}

acc_df <- reduce(acc_list, rbind)
write_csv(acc_df, paste0(output_dir, "accuracy.csv"))

df <- read_csv("./R/metrics/accuracy.csv")
