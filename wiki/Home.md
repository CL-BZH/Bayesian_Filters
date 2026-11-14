# Bayesian_Filters wiki

*Where is the mower?* The [repository](https://github.com/CL-BZH/Bayesian_Filters) answers the
question with every classical recursive Bayesian filter, all run on the same simulated world.
The explanations and derivations live in the **notebooks**. This wiki is a map of them, plus
reference pages.

## Reading order

| # | Notebook | Key idea |
|---|---|---|
| 00 | [The mower's world](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/00_the_mower_world.ipynb) | The state-space model: noisy controls in, noisy measurements out |
| 01 | [Bayes filter](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/01_bayes_filter.ipynb) | Predict = Chapman–Kolmogorov, update = Bayes; a grid does it exactly, but scales as `n^d` |
| 02 | [Kalman filter](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/02_kalman_filter.ipynb) | Linear + Gaussian: the Bayes filter in closed form; NEES tells you if the covariance is honest |
| 03 | [EKF](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/03_extended_kalman_filter.ipynb) | Linearise around the estimate; over-confident when the uncertainty is large |
| 04 | [UKF](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/04_unscented_kalman_filter.ipynb) | Propagate sigma points, not Jacobians; consistent where the EKF is not |
| 05 | [Particle filter](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/05_particle_filter.ipynb) | Weighted samples: any shape of belief, at a price in N |
| 06 | [Showdown](https://github.com/CL-BZH/Bayesian_Filters/blob/main/notebooks/06_showdown.ipynb) | Which filter when |

Prerequisites: Gaussian conditioning, Bayes' rule, a little linear algebra. Notebook 01 derives
everything else.

## Reference pages

* [[Notation]]: symbols used across the notebooks
* [[Models]]: the motion and sensor models with their Jacobians
* [[Pitfalls]]: the bugs every filter implementation meets

## Running things

```bash
pip install -r requirements.txt
jupyter lab notebooks/
python -m pytest
python -m bayesian_filters.demo --filter pf --global-init
```

## Next

[Deep-Kalman-Filters](https://github.com/CL-BZH/Deep-Kalman-Filters): what happens when the
motion and sensor models are learned instead of written down.
