import heapq
import numpy as np
from sklearn.linear_model import Ridge


def simulate(seed=17, days=600):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    values = []
    for site in range(3):
        seasonal = 35 * np.sin(2 * np.pi * t / 7 + site * 0.4) + 20 * np.cos(
            2 * np.pi * t / 28
        )
        baseline = 190 + site * 45 + t * 0.12 + seasonal
        # Rare positive spikes make mean/median errors and shortage costs differ.
        noise = rng.normal(0, 13, days) + rng.binomial(1, 0.09, days) * rng.uniform(
            65, 120, days
        )
        values.append(np.maximum(np.rint(baseline + noise), 0))
    return np.array(values).T


def costs(actual, capacity, under=9.0, over=1.0):
    if under <= 0 or over <= 0:
        raise ValueError("Costs must be positive")
    return under * np.maximum(np.asarray(actual) - capacity, 0) + over * np.maximum(
        np.asarray(capacity) - actual, 0
    )


def optimal_quantile(under, over):
    if under <= 0 or over <= 0:
        raise ValueError("Costs must be positive")
    return under / (under + over)


def allocate(samples, budget, under=9.0, over=1.0):
    """Exact integer minimiser of separable empirical convex cost under one shared cap.

    Each unit goes to the site with greatest remaining expected marginal saving.
    Stops when more capacity has no positive expected value. No minimum allocations.
    """
    samples = np.asarray(samples, float)
    if (
        samples.ndim != 2
        or not len(samples)
        or budget < 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Invalid demand scenarios or capacity budget")
    allocations = np.zeros(samples.shape[1], int)

    def benefit(site, units):
        column = samples[:, site]
        return float(
            np.mean(
                costs(column, units, under, over)
                - costs(column, units + 1, under, over)
            )
        )

    heap = [(-benefit(j, 0), j) for j in range(samples.shape[1])]
    heapq.heapify(heap)
    for _ in range(int(budget)):
        negative, j = heapq.heappop(heap)
        if negative >= 0:
            break
        allocations[j] += 1
        heapq.heappush(heap, (-benefit(j, int(allocations[j])), j))
    return allocations


def features(values):
    origins = np.arange(28, len(values))
    x = np.array(
        [
            np.r_[
                values[t - 1],
                values[t - 7],
                values[t - 14],
                values[t - 28],
                t,
                np.sin(2 * np.pi * t / 7),
                np.cos(2 * np.pi * t / 7),
                np.sin(2 * np.pi * t / 28),
                np.cos(2 * np.pi * t / 28),
            ]
            for t in origins
        ]
    )
    return origins, x


def benchmark(values, budget=1000):
    days, x = features(values)
    target = values[days]
    fit = days < 300
    calibration = (days >= 300) & (days < 360)
    selection = (days >= 360) & (days < 420)
    test = days >= 420
    model = Ridge(alpha=20).fit(x[fit], target[fit])
    predictions = dict(
        seasonal_naive=values[days - 7], ridge=np.maximum(model.predict(x), 0)
    )
    results = {}
    details = {}
    for name, pred in predictions.items():
        residuals = target[calibration] - pred[calibration]
        lower = pred + np.quantile(residuals, 0.05, axis=0)
        upper = pred + np.quantile(residuals, 0.95, axis=0)
        mean_capacity = []
        optimal_capacity = []
        for row in pred:
            capacity = np.maximum(np.rint(row), 0).astype(int)
            if capacity.sum() > budget:
                capacity = np.floor(capacity * budget / capacity.sum()).astype(int)
            mean_capacity.append(capacity)
            optimal_capacity.append(allocate(np.maximum(row + residuals, 0), budget))
        mean_capacity = np.array(mean_capacity)
        optimal_capacity = np.array(optimal_capacity)

        def metrics(mask):
            return dict(
                mae=float(np.mean(abs(target[mask] - pred[mask]))),
                interval_90_coverage=float(
                    np.mean(
                        (target[mask] >= lower[mask]) & (target[mask] <= upper[mask])
                    )
                ),
                mean_policy_cost=float(costs(target[mask], mean_capacity[mask]).sum()),
                optimised_policy_cost=float(
                    costs(target[mask], optimal_capacity[mask]).sum()
                ),
                shortage_units=int(
                    np.maximum(target[mask] - optimal_capacity[mask], 0).sum()
                ),
                maximum_allocated=int(optimal_capacity[mask].sum(axis=1).max()),
            )

        results[name] = dict(selection=metrics(selection), test=metrics(test))
        details[name] = dict(
            pred=pred, lower=lower, upper=upper, capacity=optimal_capacity
        )
    chosen_mae = min(results, key=lambda n: results[n]["selection"]["mae"])
    chosen_cost = min(
        results, key=lambda n: results[n]["selection"]["optimised_policy_cost"]
    )
    report = dict(
        data="SYNTHETIC daily demand, 3 sites, seeded asymmetric spikes",
        protocol=dict(
            train_days=[28, 300],
            residual_calibration=[300, 360],
            policy_selection=[360, 420],
            test=[420, len(values)],
        ),
        economics=dict(
            shortage_unit_cost=9,
            excess_unit_cost=1,
            critical_quantile=optimal_quantile(9, 1),
            daily_budget=budget,
        ),
        models=results,
        validation_mae_selected=chosen_mae,
        validation_cost_selected=chosen_cost,
        limitations="Empirical residual scenarios assume stable errors; correlated sites retained per-day. Intervals are empirical, not guaranteed conformal coverage. Cost units hypothetical.",
    )
    rows = []
    selected = details[chosen_cost]
    for i in np.flatnonzero(test):
        for site in range(3):
            rows.append(
                dict(
                    day=int(days[i]),
                    site=site,
                    actual=float(target[i, site]),
                    forecast=float(selected["pred"][i, site]),
                    lower=float(selected["lower"][i, site]),
                    upper=float(selected["upper"][i, site]),
                    capacity=int(selected["capacity"][i, site]),
                )
            )
    return report, rows
