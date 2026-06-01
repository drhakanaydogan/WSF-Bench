# Code availability

The repository contains the Python modules and scripts required to reproduce the analytical workflow. The code is organised into reusable modules for preprocessing, leakage-safe feature construction, model estimation, evaluation, execution-validity logging, figure generation, and diagnostic summaries.

The implementation is deterministic where random seeds are used. Pooled LightGBM and the recurrent challenger use fixed random seeds to support reproducibility.
