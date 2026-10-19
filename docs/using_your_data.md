# Reusable API

```python
from decisionops.core import allocate
# Demand scenarios: [scenarios,sites]; costs from the intended business.
capacity = allocate(demand_scenarios, budget=1000, under=9, over=1)
```

Validate units, timestamps, missingness and evaluation assumptions. The simulator is not a substitute for representative labelled data.
