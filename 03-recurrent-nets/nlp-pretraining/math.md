Some review, some not.

### Cauchy-Scharz Inequality

The dot product of two vectors is always less than or equal to the product of their magnitudes.

```math

\vec{a}^T \cdot \vec{b} ≤ ||a||\cdot||b||

```

### Cosine Similarity

Normalized dot product, by the product of the magnitudes of both vectors, yielding $\cos \theta$

```math

\cos \theta = \frac{\vec{a} \cdot\vec{b}}{||\vec{a}|| \cdot ||\vec{b}||}

```

where we can get the final angle between both vectors as:

```math
\theta = \cos^{-1} \left( \frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| \cdot ||\vec{b}||} \right) = \cos^{-1}(\cos \theta)

```

The closer to $1$ of $\cos \theta$ is, then more similar two vectors are.

The closer to $0$ $\theta$ is, the more similar two vectors are.

### Singular Values

The square roots of the non-negative eigenvalues of the covariance matrix, typically denoted as $\sigma$. The count of total non-zero singular values denotes the $\text{Rank()}$ of the matrix.

### Inverse of Orthonormal Matrices

If the matrix is orthonormal, denoted as perhaps $U$, the inverse is it's transpose, $U^T = U^{-1}$.