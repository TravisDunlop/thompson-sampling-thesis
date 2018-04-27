library(ggplot2)

path = '/Users/travisdunlop/Documents/thompson-sampling-thesis/'

results = read.csv(paste0(path, 'code/data/results.csv'))

ggplot(results, aes(num_steps, total_reward, col = policy)) + geom_point()



