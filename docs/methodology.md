# DecisionOps: methodology

Train on days 28–299, calibrate residual scenarios on 300–359, select models/policies on 360–419 and test on 420–599. Forecasts are one day ahead with observed prior-day inputs; residual scenarios preserve joint daily site errors. Costs are hypothetical: shortage 9 units, excess 1 unit, so the unconstrained critical quantile is 0.9. The shared daily capacity is 1,000. Marginal expected savings are decreasing for each site; allocating each next unit to the largest positive saving solves this particular integer convex allocation problem. The solver is tested against exhaustive enumeration on a small instance.

## Assumptions and limits

Empirical residual intervals assume stable errors and do not guarantee nominal coverage; measured coverage is reported. Fixed synthetic costs are not business ROI. A single shared constraint and separable costs permit the greedy solution; lead times, minimum staffing, network constraints and nonlinear interactions require a different optimiser. No ranking reversal is manufactured: the lower-MAE model may also be the lower-cost model. The key demonstration is that decision policies must be evaluated on operational cost, not accuracy alone.

## Extension plan

Add rolling-origin folds, capacity stress tests, conformal intervals, lead times and mixed-integer constraints for realistic staffing or inventory.
