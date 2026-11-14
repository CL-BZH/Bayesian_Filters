# Pitfalls

The bugs that every filter implementation meets.

## Angles

* **Wrap every angle difference** to $[-\pi, \pi)$: bearing innovations, heading errors,
  sigma-point residuals, particle-filter likelihoods. A predicted bearing of $3.10$ and a
  measured one of $-3.12$ differ by $0.06$ rad, not $-6.22$. Forgetting this throws the
  estimate across the map whenever a landmark crosses $\pm\pi$.
* **Average angles on the circle**: $\operatorname{atan2}\big(\sum w_i \sin a_i, \sum w_i\cos a_i\big)$.
  The arithmetic mean of $179°$ and $-179°$ is $0°$, the opposite direction. This matters for
  UKF means and for particle-filter estimates.
* Wrap the state's heading after every update.

## Kalman filters

* **Check Jacobians with finite differences.** It takes three lines (notebook 03, `tests/`).
* **Use the Joseph form** $P = (I-KH)\bar P(I-KH)^\top + KRK^\top$, or at least symmetrise $P$.
  The short form $(I-KH)\bar P$ drifts away from symmetric positive definite.
* Solve, do not invert: `K = np.linalg.solve(S, H @ P).T`.
* **Test consistency, not only accuracy.** A filter with a small error and a much too small
  covariance is a time bomb. Use NEES in simulation and NIS on real data.
* The EKF is **over-confident when the uncertainty is large** compared with the curvature
  of $f$ and $h$, e.g. after a poor initialisation (NEES ≈ 300 in notebook 03).

## Unscented transform

* The spread is $\sqrt{n+\lambda}$ times a square root of $P$, **not** $(n+\lambda)$ times it.
* Generate **all** $2n+1$ points, including those of the augmented noise dimensions.
* $\lambda = 3 - n$ gives a **negative** centre weight for $n > 3$, and then $P$ may stop being
  positive definite. Prefer scaled sigma points with $W_0 \ge 0$ (e.g. $\alpha = 1$, $\beta = 2$, $\kappa = 0$).
* A three-line unit test catches all of these: generate points for a known Gaussian and check
  that their weighted mean and covariance give it back.

## Particle filters

* **Work with log-weights** and `logsumexp`. Raw likelihoods underflow to 0 with a few precise
  measurements.
* **Resample only when needed** (ESS below $N/2$, say). Each resampling destroys diversity.
* Prefer **systematic** (or stratified) resampling to multinomial.
* **Sample impoverishment.** When the likelihood is much narrower than the spacing between
  particles (global localisation with precise sensors), one particle takes all the weight.
  Remedies: more particles, roughening, a tempered likelihood, sensor resetting.
* **Mode collapse.** Random resampling makes mode masses drift, and a mode that loses its last
  particle never comes back (notebook 05, symmetric landmarks).
* **Recovery** from a kidnapping needs new particles somewhere new: augmented MCL injects them
  when the short-term likelihood drops below the long-term one.
* Keep the algorithm separate from the drawing code: one `(N, d)` array, not $N$ objects.

## Simulation

* The filter's model and the simulator must agree, or you are testing the mismatch. A mower
  that jumps 180° at a wall cannot be tracked by any smooth motion model.
* Give the sensor the noise that the filter assumes.
* For NEES tests, draw the initial estimate from $\mathcal N(x_0, P_0)$; do not start at the truth.
