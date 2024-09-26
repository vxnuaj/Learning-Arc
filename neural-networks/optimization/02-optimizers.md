## Momentum

Momentum is the means for adding 'velocity' to gradient descent.

Given a gradient, $g_t = \frac{1}{n} \sum_{i\in n} \frac{∂L(x_i, w_{t-1})}{∂w_{t-1}}$, computed across all $T$ iterations, the optimization path along the loss surface $L$ is prone to be very stochastic with a high amount of variance.

To approach this issue, we can implement a type of variance reduction using a 'leaky average' or exponentially weighted average.

Exponentially weighted average are essentially computed as:

$V_t = \beta V_{t-1} + ( 1 - \beta ) \theta_t$

where $\beta$ is the smoothign factor and $\theta$ is the value at the current time steps.

Over multiple iterations, the formula replaces a given value of $\theta$ with $V$, a value that considers not only $\theta_t$ but previous values of $\theta$, $\theta_{t-n}$ where $n>0$, with exponentially decaying weights for earlier $\theta$ values as iterations go on.

For a time step $t$, as we move away from $t$, each older value, $V_{t}$ will have a decreased weight, as new values are constnatly being overshadowed by values of $\theta$ at the current $t$.

We apply the same principle to our gradients,as:

$v_t = \beta v_{t-1} + (1 - \beta) g_t$

where $g_t$ is the gradient at the current time step.

Then in the update rule, 

$w_t = w_{t-1} - \alpha g_t$

$g_t$ becomes $v_t$ as,

$w_t = w_{t-1} - \alpha v_t$

where the velocity term, $v$, provides an averaged gradient given by the exponentially weighted average.

Thereby, there is less variance in the optimization path. It's more direct, and *accelerated*.

The choice of $\beta$ matters, it's an important hyperparameter.

For large values of $\beta$, previous gradients will have more importance over the gradient at the $t$ time step. 

For smaller values of $\beta$, current gradients will have more improtance over the past gradients.


