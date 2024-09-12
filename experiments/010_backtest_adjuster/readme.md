# Adjuster

We ran the adjuster on the backtest results, they improved the results very slightly.
We see larger improvements in live.

The adjuster looks back at the error over the last 7 days for each half hour period,
then minus this of the next predictions.

![aduster](./mae.png)
