library(ggplot2)
library(plotly)
packageVersion(plotly)

path = '/Users/travisdunlop/Documents/thompson-sampling-thesis/'

results = read.csv(paste0(path, 'data/results.csv'))

results = results[results$cost_per_step != 0,]

p <- ggplot(results, aes(steps, cost_per_step, col = policy)) + 
  geom_point(alpha = 0.25) +
  #geom_smooth(method="loess", se = FALSE) +
  facet_wrap(~ environment, ncol = 2) +
  coord_cartesian(ylim = c(0, 1)) 

pltly <- ggplotly(p)

htmlwidgets::saveWidget(as_widget(pltly), paste0(path, 'images/results.html'))

results[sample(nrow(results), 10),]
