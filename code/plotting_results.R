library(ggplot2)

path = '/Users/travisdunlop/Documents/thompson-sampling-thesis/'

results = read.csv(paste0(path, 'data/results.csv'))

results = results[results$cost_per_step != 0,]

ggplot(results, aes(steps, cost_per_step, col = policy)) + 
  geom_point(alpha = 0.25) +
  #geom_smooth(method="loess", se = FALSE) +
  facet_wrap(~ environment, ncol = 2) +
  coord_cartesian(ylim = c(0, 1)) 

sample(results, 5)

(results[results$cost_per_step == 0,])
