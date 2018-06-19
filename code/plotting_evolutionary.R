library(data.table)
library(ggplot2)
library(plotly)
packageVersion(plotly)

folder = '/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/'

results<- read.table(paste0(folder, 'data/adversarial evolutionary/short_data.txt'), 
                     sep = '|', header = TRUE)

results$num_experts <- as.factor(results$num_experts)
head(results)

curve_fit = data.table(results)[, list(max=max(mean_regret_per_step)), by = num_steps]
curve_fit$constant = sqrt(curve_fit$num_steps) * curve_fit$max
constant = max(curve_fit$constant)
curve_fit

x = seq(5, 250)
curve = data.frame(x = x, y = constant/sqrt(x))

ggplot(results) + geom_point(aes(num_steps, mean_regret_per_step, color = num_experts)) +
  geom_line(data = curve, aes(x, y)) + scale_x_log10() + scale_y_log10()

constant
