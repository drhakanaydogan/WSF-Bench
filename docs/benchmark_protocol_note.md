# Benchmark protocol note

WSF-Bench uses task-specific panels rather than forcing production and trade indicators into one artificial common-country structure. The primary evaluation unit is the panel-target-split block.

The chronological windows are defined as disruption-relevant operating environments:

- pre-pandemic benchmark window,
- pandemic-shock holdout,
- post-shock evaluation window.

The regime windows are not treated as statistically estimated breakpoints. They are used to evaluate whether forecasting performance and execution validity remain stable across operationally meaningful periods.

Pooled LightGBM execution is classified using the following codes:

- `V0`: execution-valid pooled LightGBM run,
- `E1`: empty effective train/test design after feature construction and filtering,
- `E2`: empty or non-alignable feature matrix,
- `E3`: model fitting or prediction exception.

Fallback-contingent outputs are retained for auditability but are not treated as valid pooled LightGBM wins.
