#### ---- gardenPlot
d <- tibble(position = c((1:4^1) / 4^0,
                         (1:4^2) / 4^1,
                         (1:4^3) / 4^2),
            draw = rep(1:3, times = c(4^1, 4^2, 4^3)),
            fill = rep(c("b", "r"), times = c(1, 3)) %>%
            rep(., times = c(4^0 + 4^1 + 4^2)))

lines_1 <- tibble(x = rep((1:4), each = 4),
                  xend = ((1:4^2) / 4),
                  y = 1,
                  yend = 2)

lines_2 <- tibble(x = rep(((1:4^2) / 4), each = 4),
                  xend = (1:4^3)/(4^2),
                  y = 2,
                  yend = 3)

d %>%
      ggplot(aes(x = position, y = draw)) +
        ggtitle('Potential paths to a sample of size 3') +
        xlab('conjecture: 1 blue & 3 red marbles') + ylab('draws') +
        geom_segment(data = lines_1,
                     aes(x = x, xend = xend, y = y, yend = yend),
                     size = 1/3) +
        geom_segment(data = lines_2,
                     aes(x = x, xend = xend, y = y, yend = yend),
                         size  = 1/3) +
        geom_point(aes(fill = fill), shape = 21, size = 3) +
        scale_y_continuous(breaks = 1:3) +
        scale_fill_manual(values  = c("navy", "red")) +
        theme(panel.grid.minor = element_blank(), legend.position = "none",
              plot.title = element_text(hjust = 0.5))
