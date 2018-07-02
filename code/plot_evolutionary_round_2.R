
library(ggplot2)
library(data.table)

folder = '/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/'

results<- read.table(paste0(folder, 'data/evolutionary/testing_all_agents_results.csv'), 
                     sep = ',', header = TRUE)

head(results)

results <- results[results$num_turns != 14, ]

results$num_experts <- as.factor(results$num_experts)
head(results)

max_df <- function(df, metric, by_cols) {
  dt <- data.table(df)[, list(metric = max(get(metric))), by = by_cols]
  return(data.frame(dt))
}

plot1_results = results[results$num_actions %in% c(2, 14, 26), ]

plot1_results$num_actions <- as.factor(plot1_results$num_actions)
levels(plot1_results$num_actions) <- c('number of actions: N = 2', 'N = 14', 'N = 26')

max_results <- max_df()

plot1_constants = data.table(plot1_results)[, list(max=max(regret)), by = c('num_turns', 'num_actions', 'agent_type')]
plot1_constants$constant = plot1_constants$max / sqrt(plot1_constants$num_turns)
constants = data.table(plot1_constants)[, list(max=max(constant)), by = c('agent_type', 'num_actions')]
constants

x = seq(10, 100)
curves = merge(data.frame(constants), data.frame(x = x))
curves$y = curves$max * sqrt(curves$x)

plot1_max <- max_df(plot1_results, 'regret', c('agent_type', 'num_actions', 'num_turns'))

plot1_subset = plot1_results[sample(nrow(plot1_results), nrow(plot1_results) / 5), ]

ggplot(plot1_max, aes(num_turns, metric, color = agent_type)) + 
  geom_point(alpha  = 1) +
  geom_point(data = plot1_subset, aes(num_turns, regret), alpha = 0.1) +
  geom_line(data = curves, aes(x, y)) + 
  facet_wrap( ~ num_actions) +
  scale_x_log10() + scale_y_log10() + 
  coord_cartesian(ylim = c(0.9, 30)) +
  labs(x = 'log number of turns (T)', y = 'log total regret', col = 'algorithm',
       title = 'Evolutionary Method Results') +
  theme_bw()
  
ggsave(paste0(folder, 'images/evolutionary/num_turns_by_regret.png'), 
       width = 10, height = 5)

constant
