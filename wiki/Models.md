# Models

## Motion

The mower turns by $\delta$, then drives $s$ along its new heading $\theta' = \theta + \delta$:

$$
f(x, u) = \begin{pmatrix} p^x + s\cos\theta' \\ p^y + s\sin\theta' \\ \theta' \end{pmatrix},
\qquad x_{t+1} = f(x_t, u_t + \varepsilon_t),\quad \varepsilon_t \sim \mathcal N(0, M).
$$

$$
F = \begin{pmatrix}1&0&-s\sin\theta'\\0&1&s\cos\theta'\\0&0&1\end{pmatrix},\qquad
V = \begin{pmatrix}\cos\theta'&-s\sin\theta'\\ \sin\theta'&s\cos\theta'\\0&1\end{pmatrix},\qquad
Q \approx V M V^\top .
$$

The noise is in the control. Turn noise therefore becomes position noise *across* the path, and
after a few steps the true distribution is curved (a "banana"), not elliptical. The UKF
handles this with an augmented state $(x, \varepsilon)$ of dimension 5; the particle filter
simply samples $\varepsilon$.

The commands follow a figure of eight (drive 4, turn $\pm\pi/50$, flip direction each loop),
plus a steering term towards the centre near the walls. The filters are told the full command.

## Range–bearing to landmark $j$

With $d = \ell_j - p$ and $q = \lVert d \rVert^2$:

$$
h_j(x) = \begin{pmatrix}\sqrt q \\ \operatorname{atan2}(d_y, d_x) - \theta\end{pmatrix},\qquad
H_j = \begin{pmatrix} -d_x/\sqrt q & -d_y/\sqrt q & 0 \\ d_y/q & -d_x/q & -1 \end{pmatrix}.
$$

Several landmarks: stack the rows and use a block-diagonal $R$.

**Data association.** With `association="known"` the sensor says which landmark it saw. With
unknown association:
* the EKF picks, for each measurement, the landmark with the smallest Mahalanobis distance
  (`association="nearest"`): a hard decision, fragile when the estimate is far off;
* the particle filter marginalises:
  $p(z \mid x) = \frac1M \sum_j \mathcal N(z; h_j(x), R)$ (`association="unknown"`).

## Linear variant (notebook 02)

With a compass reading $h_t$ the position follows a linear model,

$$
p_{t+1} = p_t + s_t\begin{pmatrix}\cos h_t\\ \sin h_t\end{pmatrix} + w_t,\qquad
Q_t = G\,\operatorname{diag}(\sigma_s^2, \sigma_c^2)\,G^\top,\quad
G = \begin{pmatrix}\cos h & -s\sin h\\ \sin h & s\cos h\end{pmatrix},
$$

and the GPS gives $z = p + v$, $v \sim \mathcal N(0, \sigma_{\text{gps}}^2 I)$.

## Grid variant (notebook 01)

Position-only table on $4\times4$ cells. The heading comes from the compass. Predict = shift +
Gaussian blur, update = multiply by the range (or range–bearing) likelihood.
