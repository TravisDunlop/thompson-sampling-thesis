library(data.table)
library(ggplot2)
library(plotly)
packageVersion(plotly)

folder = '/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/'


results <- read.csv(paste0(folder, 'data/game_theory_results.csv'))

head(results)

levels(results$payoff_type) <- c('identity', 'rock paper scissors', 'uniform')

by_payoff <- results[results$num_actions %in% c(2, 15, 29) & results$regret_1 > 1, ]

by_payoff_dt <- data.table(by_payoff)

curve_fit = by_payoff_dt[, list(max=max(regret_1)), by = c('num_turns', 'agent_1', 'payoff_type', 'num_actions')]
curve_fit$constant =  curve_fit$max / sqrt(curve_fit$num_turns)
constants = data.table(curve_fit)[, list(max=max(constant)), by = c('agent_1', 'payoff_type', 'num_actions')]
constants

x = seq(10, 1000)
curves = merge(data.frame(constants), data.frame(x = x))
curves$y = curves$max * sqrt(curves$x)

curves

head(curve_fit)

ggplot(curve_fit, aes(num_turns, max, col = agent_1)) + 
  geom_point(alpha = 0.75) + 
  #geom_point(data = by_payoff, aes(num_turns, regret_1), alpha = 0.01) +
  geom_line(data = curves, aes(x, y)) +
  scale_x_log10() + scale_y_log10() + facet_grid(payoff_type ~ num_actions) + 
  labs(x = 'log number of turns', y = 'log total regret', col = 'policy')


test <- data.frame(x = seq(2, 1000))
test$logx = log(test$x)
test$sqrtx = 5 * sqrt(test$x)


ggplot(test, aes(x, sqrtx)) + geom_line() + scale_x_log10() + scale_y_log10()
