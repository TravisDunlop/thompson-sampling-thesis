library(data.table)
library(ggplot2)
library(plotly)
packageVersion(plotly)

folder = '/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/'

files = list.files(paste0(folder, 'data'), pattern="*.csv", full.names = T)

results <- do.call(rbind, lapply(files, fread))

results = results[results$regret_per_step != 0,]

p <- ggplot(results, aes(steps, regret_per_step, col = policy)) + 
  geom_point(alpha = 0.25) +
  #geom_smooth(method="loess", se = FALSE) +
  facet_wrap(~ environment, ncol = 2) 
  #coord_cartesian(ylim = c(0, 1)) 

print(p)

pltly <- ggplotly(p)

htmlwidgets::saveWidget(as_widget(pltly), paste0(folder, 'images/results.html'))

results[sample(nrow(results), 10),]


