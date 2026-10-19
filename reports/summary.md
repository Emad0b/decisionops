# DecisionOps backtest

Synthetic demand, hypothetical shortage/excess costs 9:1, shared daily capacity 1000.

- seasonal_naive: MAE 34.29; mean-capacity cost 98718; optimised cost 61807; interval coverage 0.878.
- ridge: MAE 20.33; mean-capacity cost 63198; optimised cost 51493; interval coverage 0.887.

Selection uses days 360–419 only. Lowest-MAE selection: ridge; lowest-cost selection: ridge. No test-based winner selection. Forecast accuracy and decision cost are different objectives; the demo does not guarantee a ranking reversal.
