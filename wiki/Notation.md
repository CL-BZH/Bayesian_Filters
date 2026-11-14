# Notation

| Symbol | Meaning | In code |
|---|---|---|
| $x_t = (p^x_t, p^y_t, \theta_t)^\top$ | pose at step $t$; $\theta \in [-\pi, \pi)$ | `x`, `run.truth[t]` |
| $u_t = (s_t, \delta_t)^\top$ | control: turn by $\delta$, then walk $s$ | `u`, `run.controls[t]` |
| $\varepsilon_t \sim \mathcal N(0, M)$ | control noise, $M = \operatorname{diag}(\sigma_s^2, \sigma_\delta^2)$ | `world.motion_cov` |
| $f(x, u)$ | noise-free motion model | `motion_model` |
| $F = \partial f/\partial x$, $V = \partial f/\partial u$ | motion Jacobians | `motion_jacobians` |
| $\ell_j$ | position of landmark $j$ | `world.landmarks[j]` |
| $h_j(x) = (r_j, \phi_j)$ | range and bearing (relative to heading) of landmark $j$ | `range_bearing` |
| $H = \partial h/\partial x$ | measurement Jacobian, rows $(r_1, \phi_1, r_2, \phi_2, \dots)$ | `range_bearing_jacobian` |
| $R$ | measurement noise covariance, $\operatorname{diag}(\sigma_r^2, \sigma_\phi^2)$ per landmark | `world.range_bearing_cov` |
| $z_t$ | measurement after control $u_t$ | `run.z_rb[t]`, `run.z_gps[t]`, `run.z_compass[t]` |
| $\operatorname{bel}(x_t)$ | posterior $p(x_t \mid z_{1:t}, u_{1:t})$ | `flt.mean`, `flt.cov` (Gaussian summary) |
| $\overline{\operatorname{bel}}(x_t)$ | prediction $p(x_t \mid z_{1:t-1}, u_{1:t})$ | state after `flt.predict(u)` |
| $\hat x, P$ | posterior mean and covariance | `flt.mean`, `flt.cov` |
| $\bar x, \bar P$ | predicted mean and covariance | |
| $y = z - h(\bar x)$ | innovation (bearings wrapped) | `stack_residual` |
| $S$ | innovation covariance | `ekf.last_S` |
| $K$ | Kalman gain | |
| $\chi_j$, $W^{(m)}_j$, $W^{(c)}_j$ | sigma points and their weights | `sigma_points`, `merwe_weights` |
| $x^{(i)}, w^{(i)}$ | particle $i$ and its weight | `pf.particles[i]`, `pf.weights[i]` |
| ESS | effective sample size $1/\sum_i (w^{(i)})^2$ | `effective_sample_size` |
| NEES | $(x - \hat x)^\top P^{-1} (x - \hat x)$, $\chi^2_n$ for a consistent filter | `nees`, `nees_bounds` |

## Indexing convention

`simulate` returns arrays where control `controls[t]` moves the turtle from `truth[t]` to
`truth[t+1]`, and measurement index `t` (`z_rb[t]`, …) is taken at `truth[t+1]`. A
`FilterResult` follows the truth: `means[0]` is the initial belief and `means[t+1]` the belief
after predicting with `controls[t]` and updating with measurement `t`.

## Default noise levels (`World`)

| | value |
|---|---|
| step noise $\sigma_s$ | 0.4 (steps are 4 units) |
| turn noise $\sigma_\delta$ | 0.03 rad |
| range noise $\sigma_r$ | 5 |
| bearing noise $\sigma_\phi$ | 0.05 rad |
| GPS noise $\sigma_{\text{gps}}$ | 8 |
| compass noise $\sigma_c$ | 0.02 rad |
| arena | $[-160, 160]^2$ |
