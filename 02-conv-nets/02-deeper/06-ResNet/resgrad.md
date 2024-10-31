# Gradients for Residual Connections

> *I don't understand things if I can't write them out from first principles*

Here's how we always have a $∂$ of at least $1$ for a layer with a Residual Connection.

<div align = center>

### Forward

```math
x_2 = \text{ReLU}(F_1(x_1)) 
\\[3mm]
x_3 = \text{ReLU}(F_2(x_2) + x_1)
\\[3mm]
x_4 = \text{ReLU}(F_3(x_3))
\\[3mm]
x_5 = \text{ReLU}(F_4(x_4) + x_3)
\\[3mm]
L = \text{CrsEnt}(p \cdot log(x_5))
```

### Backward

```math

\frac{∂L}{∂x_5} = x_5 - p
\\[5mm]
\frac{∂L}{∂x_4} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})
\\[5mm]
\frac{∂L}{∂x_3} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3}) + (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_3}) = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3}) + 1
\\[5mm]
\frac{∂L}{∂x_2} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})
\\[5mm]
\frac{∂L}{∂x_1} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})(\frac{∂x_2}{∂x_1}) + (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_1}) = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})(\frac{∂x_2}{∂x_1}) + 1
```

<br>

```math

(\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_1}) = 1
\\[3mm]
\text{and}
\\[3mm]
(\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_3}) = 1
```

as when we take the $∂$ w.r.t to $∂x_1$ or $∂x_3$ propagating to the residual connection, we end up taking the gradient of $∂x_1$ and $∂x_3$ w.r.t to itself, such that it is equal to $1$ 

Simpler example:

```math

\frac{∂y}{∂x_1} = \frac{∂}{∂x_1}(F(x_2) + x_1) = \frac{∂F(x_2)}{∂x_1} + \frac{∂x_1}{∂x_1} = 0 + 1 = 1

```

as $\frac{∂F(x)}{∂x_1}$ is $0$ given that $∂F(x)$ is treated as a constant and a $∂$ of a variable w.r.t to itself is always $1$.

Hence, for residual connections, $F(x_l) + x_{l - 2}$, the gradient with respect a given residual connection will always be $1$.<br>
Thereby, for the given layers that include a residual connection, in this case $F_4$ and $F_2$, we'll always have a $∂ > 1$, such that we end up eliminating the vanishing gradient problem for those layers, and perhaps earlier layers, though this is dependent on how large your residual block is.

The deeper your residual block is, the more layers the $∂$'s will backpropagate through without the alleviating residual connection, such that your set of gradients can still vanish, until ythe gradients meet a residual connection.