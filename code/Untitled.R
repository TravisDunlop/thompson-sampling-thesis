library(data.table)
library(ggplot2)
library(plotly)
packageVersion(plotly)

folder = '/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/'


results <- read.csv(paste0(folder, 'data/game_theory_results.csv'))

head(results)

results$loss_per_turn <- results$loss_1 / results$num_turns

levels(results$payoff_type) <- c('identity', 'rock paper scissors', 'uniform')

max_regret = data.table(results)[, list(regret=max(regret_1)), by = c('num_turns', 'num_actions', 'agent_1', 'payoff_type')]
max_regret
########################################################################

by_turns <- max_regret[max_regret$num_actions %in% c(2, 15, 29) & max_regret$regret > 1, ]

by_turns$num_actions <- as.factor(by_turns$num_actions)

levels(by_turns$num_actions) <- c('N = 2', 'N = 15', 'N = 29')

by_turns$constant =  by_turns$regret / sqrt(by_turns$num_turns)
constants = data.table(by_turns)[, list(max=max(constant)), by = c('agent_1', 'payoff_type', 'num_actions')]
constants

x = seq(10, 1000)
curves = merge(data.frame(constants), data.frame(x = x))
curves$y = curves$max * sqrt(curves$x)

head(curves)

head(curve_fit)

ggplot(by_turns, aes(num_turns, regret, col = agent_1)) + 
  geom_point(alpha = 0.75) + 
  geom_line(data = curves, aes(x, y)) +
  scale_x_log10() + scale_y_log10() + facet_grid(payoff_type ~ num_actions) + 
  labs(x = 'log number of turns (T)', y = 'maximum log total regret', col = 'policy',
       title = 'Constant-Sum Game Results') +
  theme_bw()

ggsave(paste0(folder, 'images/game theory/num_turns_by_regret.png'), 
       width = 10, height = 5)

########################################################################

test <- data.frame(x = seq(2, 1000))
test$logx = log(test$x)
test$sqrtx = 5 * sqrt(test$x)

ggplot(test, aes(x, sqrtx)) + geom_line() + scale_x_log10() + scale_y_log10()

########################################################################

unique(results$num_turns)

by_actions <- max_regret[max_regret$num_turns %in% c(10, 100, 1000) & max_regret$regret > 1, ]

by_actions$num_turns <- as.factor(by_actions$num_turns)

levels(by_actions$num_turns) <- c('T = 10', 'T = 100', 'T = 1000')

curve_fit$constant =  curve_fit$max / sqrt(curve_fit$num_turns)
constants = data.table(curve_fit)[, list(max=max(constant)), by = c('agent_1', 'payoff_type', 'num_actions')]
constants

x = seq(10, 1000)
curves = merge(data.frame(constants), data.frame(x = x))
curves$y = curves$max * sqrt(curves$x)

curves

head(curve_fit)

ggplot(by_actions, aes(num_actions, regret, col = agent_1)) + 
  geom_point(alpha = 0.75) + 
  #geom_point(data = by_payoff, aes(num_turns, regret_1), alpha = 0.01) +
  #geom_line(data = curves, aes(x, y)) +
  scale_y_log10() + facet_grid(payoff_type ~ num_turns) + 
  labs(x = 'number of actions (N)', y = 'maximum log total regret', col = 'policy', 
      title = 'Constant-Sum Game Results - analysis of maximum regret')

