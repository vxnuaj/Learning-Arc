## Learning Rate Scheduling

Learning rate matters!

If it's too high, while your model might learn to recognize patterns quickly, when attempting to converge onto the minima, chances are learning rate will be to large to find the precise set of parameters that yield $∂\mathcal{L}(\theta, y) = 0 \text{ for all } \theta_i \text{ where i is the ith layer}$.

Of course, if it's too small, the model will learn very slowly.

You can schedule the learnign rate, but the rate of decay is very important, you don't want the value of $\alpha$ to be too big or small at the wrong time, $\tau$. Otherwise, your model might end up iterating, away from the minima via stochastic gradient ascent.

There are many, many means to schedule a learning rate.

Lots of them invovle the decrease of the learning rate over time, such as $\alpha = \alpha \cdot \eta$, where $\eta$ is the factor we decrease $\alpha$ by, or step-wise annealing, or cosine annealing.

Interesingly, cosine annealing, in the original paper, suggested the use of cosine annealing with warm restasrts. After each warm restart of the training process for a neurl network, restart the learning rate scheduler and continue on.

Cosine Annealing = $\alpha_t = \alpha_T + \frac{\alpha_0 - \alpha_T}{2}(1 + cos(\frac{\pi t}{T}))$, where $\alpha_T$ is the minimum learning rate, $\alpha_0$ is the initial learning rate, and $\alpha_t$ is the current learning rate.

Cosine annealing restarts at once $\alpha_T$ is reached, allowing for automatic warm restarts of the training process.

The important thing to takeaway is not the specific formuals for all different learning rate schedulers but to note that there area variety to choose from and area all empirically dependent on your problem and neural network architecture.

The proper learning rate can stochasically shoot your model & parameters out of the global minima or complement smooth convergence.

Either or, experimentation is key, but follow Occams Razor.

Some key points:

- **Decreasing the learning rate over training might reduce overfitting, though in theory, it is not certain why**
- A step-wise annealing of the learning rate is effective and simple.
- Learning Rate warm-up can prevent divergence, as you don't overshoot too quick in the beginning. This is crucial as you have no idea where you are in the loss landscape and might want to be a bit more careful (think your yourself, positioned at the coordinates of your parameters, where you ahve no idea where you are, in a dark threatening forest. You have limited steps (compute time) to find shelter (global minima). I'd take careful steps.)
- Different amounts of learning rate scheduling can lead to different levels of overfitting, despite the same training error / accuracy. Of course, this indicates local minima.