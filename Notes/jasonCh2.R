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

#### ---- globePlot
x <- seq(0,1, length=1000)
outcomes <- c('W', 'L', 'W', 'W', 'W', 'L', 'W', 'L', 'W')
a <- c(1, 2, 2, 3, 4, 5, 5, 6, 6, 7)
b <- c(1, 1, 2, 2, 2, 2, 3, 3, 4, 4)
par(mfrow = c(3,3))
for (i in 2:10){
  plot(x, dbeta(x, shape1 = a[i], shape2 = b[i]), ylim = c(0,3),
       xlab = 'parameter value', ylab = 'density', bty ='n')
  points(x, dbeta(x, shape1 = a[i-1], shape2 = b[i-1]), col = 'grey')
  points(x, dbeta(x, shape1 = a[i], shape2 = b[i]))
  mark <- c(rep(1,i-2), rep(2,1), rep(1,10-i))
  text(seq(.1, .9, by = .1), 2.9, outcomes, font = mark,
       col = recode(mark, '1' = 'grey', '2' = 'black'))
 }
