1. **Function Definition**:
   - Create a function called `conv1d` that takes three parameters:
     - `input_vector`: A 1D NumPy array (signal).
     - `kernel`: A 1D NumPy array (filter/kernel).
     - `stride` (optional): An integer that specifies the stride of the convolution (default is 1).
     - `padding` (optional): An integer that specifies the amount of zero-padding added to both sides of the input (default is 0).

2. **Padding**:
   - Implement zero-padding on the input vector based on the `padding` parameter. This means adding zeros to both ends of the input vector.

3. **Output Size Calculation**:
   - Calculate the size of the output vector based on the input size, kernel size, stride, and padding. Use the following formula:
     \[
     \text{output\_size} = \left\lfloor \frac{\text{input\_size} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1
     \]

4. **Convolution Operation**:
   - Initialize an output vector of the calculated size filled with zeros.
   - Loop over the input vector, applying the kernel at each position according to the stride.
   - For each position, compute the dot product of the kernel with the corresponding section of the input vector and store the result in the output vector.

5. **Return the Output**:
   - Return the resulting output vector after completing the convolution operation.

## Example Usage
```python
import numpy as np

input_vector = np.array([1, 2, 3, 4, 5])
kernel = np.array([0.2, 0.5, 0.3])
output = conv1d(input_vector, kernel, stride=1, padding=1)
print(output)