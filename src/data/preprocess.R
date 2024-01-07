# Define global variables
N = 50
s = 10


# Read data
history = subset(read.csv("../../data/raw/history.csv"), select=-X)
codecarbon = subset(read.csv("../../data/raw/emissions.csv"), select=-X)
nvidia = subset(read.csv("../../data/raw/monitor.csv"), select=-X)
datasets = read.csv("../../data/raw/datasets.csv")


# Parse attributes
colnames(codecarbon)[colnames(codecarbon) == "timestamp"] = "end"
codecarbon$start = as.POSIXct(codecarbon$start, format = "%Y-%m-%d %H:%M:%OS")
codecarbon$end = as.POSIXct(codecarbon$end, format = "%Y-%m-%d %H:%M:%OS")

colnames(nvidia) = c("timestamp", "utilization_gpu", "utilization_memory", "memory_total", "memory_used", "power_draw", "temperature_gpu")
nvidia$timestamp = as.POSIXct(nvidia$timestamp, format = "%Y-%m-%d %H:%M:%OS")
cols = c("utilization_gpu", "utilization_memory", "memory_total", "memory_used", "power_draw")
nvidia[cols] = lapply(
  nvidia[cols], 
  function(x) as.numeric(sapply(
    x, function(val) (val %>% as.character() %>% strsplit(" ") %>% unlist())[2])
  )
)


# Process accuracies
history$accuracy = ifelse(history$accuracy == 0, 
                          1 / datasets$num_classes[match(history$dataset, datasets$dataset)],
                          history$accuracy)
history$val_accuracy = ifelse(history$val_accuracy == 0, 
                              1 / datasets$num_classes[match(history$dataset, datasets$dataset)], 
                              history$val_accuracy)
history["logodds"] = logitlink(history$accuracy)
history["val_logodds"] = logitlink(history$val_accuracy)


# Merge all sources
hist.carbon = left_join(history, codecarbon, by=c("dataset", "mode", "intervention", "epoch"))
hist.carbon = hist.carbon %>% mutate(dataset = recode(dataset, "visual_domain_decathlon/aircraft" = "aircraft"))
datasets = datasets %>% mutate(dataset = recode(dataset, "visual_domain_decathlon/aircraft" = "aircraft"))
datasets = bind_rows(datasets[11, ], datasets[-11, ])

all_data = hist.carbon %>%
  rowwise() %>%
  mutate(subset_nvidia = nvidia %>%
           filter(timestamp >= start, timestamp <= end) %>%
           subset(select=-timestamp) %>%
           do(data.frame(t(colMeans(.))))) %>%
  unnest(subset_nvidia) %>%
  as.data.frame()
all_data$energy_nvidia = all_data$power_draw * all_data$duration / 3600000

remove(nvidia, history, codecarbon, hist.carbon)


## Add cumulative variables
center = function(df, attribute) {
  return (ave(as.vector(df[,attribute]), df$dataset, FUN = function(x) x - mean(x)) + mean(df[,attribute]))
}

get_cumulative <- function(df, row, attribute) {
  dataset <- row["dataset"]
  mode <- row["mode"]
  intervention <- as.numeric(row["intervention"])
  epoch <- as.numeric(row["epoch"])
  if (mode == "base") {
    return(sum(df[df$dataset == dataset &
                    df$mode == "base" & 
                    df$epoch <= epoch, attribute]))
  }
  else {
    base_energy <- sum(df[df$dataset == dataset & 
                            df$mode == "base" & 
                            df$epoch <= intervention, attribute])
    intervention_energy <- sum(df[df$dataset == dataset & 
                                    df$mode == mode &
                                    df$intervention == intervention & 
                                    df$epoch <= epoch, attribute])
    return(base_energy + intervention_energy)
  }
}

all_data$val_logodds_centered = center(all_data, "val_logodds")
all_data$logodds_centered = center(all_data, "logodds")
all_data$energy_centered = center(all_data, "energy_consumed")
all_data$energy_nvidia_centered = center(all_data, "energy_nvidia")
all_data$duration_centered = center(all_data, "duration")

all_data$energy_centered_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="energy_centered")
all_data$duration_centered_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="duration_centered")
all_data$energy_nvidia_centered_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="energy_nvidia_centered")

all_data$energy_nvidia_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="energy_nvidia")
all_data$energy_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="energy_consumed")
all_data$duration_cumulative <- apply(all_data, 1, get_cumulative, df=all_data, attribute="duration")
all_data$score = all_data$val_accuracy / all_data$energy_nvidia_cumulative


## Add scaled variables
scale_df = function(df, attribute) {
  return (ave(as.vector(df[,attribute]), df$dataset, FUN = function(x) (x - mean(x)) / sd(x)) + mean(df[,attribute]))
}
scale_df2 = function(df, attribute) {
  return (ave(as.vector(df[,attribute]), df$dataset, df$mode, FUN = function(x) (x - mean(x)) / sd(x)))
}


all_data$temperature_gpu_centered = center(all_data, "temperature_gpu")

all_data$val_logodds_scaled = scale_df(all_data, "val_logodds")
all_data$logodds_scaled = scale_df(all_data, "logodds")
all_data$energy_scaled = scale_df(all_data, "energy_consumed")
all_data$energy_nvidia_scaled = scale_df(all_data, "energy_nvidia")
all_data$duration_scaled = scale_df(all_data, "duration")
all_data$temperature_gpu_scaled = scale_df(all_data, "temperature_gpu")
all_data$utilization_gpu_scaled = scale_df(all_data, "utilization_gpu")

all_data$val_logodds_scaled2 = scale_df2(all_data, "val_logodds")
all_data$logodds_scaled2 = scale_df2(all_data, "logodds")
all_data$energy_scaled2 = scale_df2(all_data, "energy_consumed")
all_data$energy_nvidia_scaled2 = scale_df2(all_data, "energy_nvidia")
all_data$duration_scaled2 = scale_df2(all_data, "duration")
all_data$temperature_gpu_scaled2 = scale_df2(all_data, "temperature_gpu")
all_data$utilization_gpu_scaled2 = scale_df2(all_data, "utilization_gpu")


## Write processed data
write.csv(all_data, "../../data/processed/all_data.csv")
