## LeNet-5

Built by Yann LeCun, 1989 -- the foundational CNN.

**Architecture**

1. **Input Layer**:
   - **Input Size**: $32 \times 32$ pixels (grayscale images)
   
2. **Layer 1 - Convolutional Layer (C1)**:
   - **Filters**: 6 filters
   - **Filter Size**: $5 \times 5$
   - **Stride**: 1
   - **Output Size**: $28 \times 28$ (calculated as $(32 - 5 + 1) = 28$)
   - **Activation Function**: Tanh (hyperbolic tangent)

3. **Layer 2 - Subsampling Layer (S2)**:
   - **Type**: Average Pooling
   - **Pool Size**: $2 \times 2$
   - **Stride**: 2
   - **Output Size**: $14 \times 14$ (calculated as $(28 / 2) = 14$)

4. **Layer 3 - Convolutional Layer (C3)**:
   - **Filters**: 16 filters
   - **Filter Size**: $5 \times 5$
   - **Stride**: 1
   - **Output Size**: $10 \times 10$ (calculated as $(14 - 5 + 1) = 10$)
   - **Activation Function**: Tanh
   - **Note**: Not all filters are connected to all input feature maps, which reduces parameters.

5. **Layer 4 - Subsampling Layer (S4)**:
   - **Type**: Average Pooling
   - **Pool Size**: $2 \times 2$
   - **Stride**: 2
   - **Output Size**: $5 \times 5$ (calculated as $(10 / 2) = 5$)

6. **Layer 5 - Convolutional Layer (C5)**:
   - **Filters**: 120 filters
   - **Filter Size**: $5 \times 5$
   - **Stride**: 1
   - **Output Size**: $1 \times 1$ (calculated as $(5 - 5 + 1) = 1$)
   - **Activation Function**: Tanh

        C5 is labeled as a convolutional layer instead of a fully connected layer, because if LeNet-5 input becomes larger and its structure remains unchanged, its output size will be greater than 1x1, i.e. not a fully connected layer.

        > In the original paper, C5 was meant to be a convolution. While it is equivalent to a FC layer if the input remains to be $32 \times 32$, this won't be the case if the input varies in size.

7. **Layer 6 - Fully Connected Layer (F6)**:
   - **Units**: 84
   - **Activation Function**: Tanh

8. **Output Layer**:
   - **Units**: 10 (for digit classification: 0 to 9)
   - **Activation Function**: Softmax (previusly were RBF units, now purely softmax for simplicity)