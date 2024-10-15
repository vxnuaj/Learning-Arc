### **Backpropagation Process for Each Layer**

#### **1. Layer 2: Convolution (C)**

**Input Shape:** (10, 3, 29, 28)  
**Kernel Shape:** (2, 3, 3, 2)  
**Stride:** (2, 2)  
**Bias Shape:** (2, 1, 1)  
**Output Shape:** (10, 2, 14, 14)

**Forward Pass Recap:**

* You apply 2 filters of shape (3,3) to the input of shape (10,3,29,28) with a stride of (2,2), and the output is of shape (10,2,14,14).

**Backward Pass (Backpropagation) Steps:**

1. **Gradient w.r.t Output (Loss Gradient):**
    * You start with ∂L/∂Z₂, the gradient of the loss with respect to the output, which has shape (10,2,14,14).
  
2. **Gradient w.r.t Weights (Kernels):**
    * To compute the gradient with respect to the filters (kernels), you'll convolve the **gradient of the loss with respect to the output** ∂L/∂Z₂ with the **input** to the layer:
    ∂L/∂W₂ = convolve(∂L/∂Z₂, input at layer 2)
    * The result will have the same shape as the kernels: (2,3,3,2).

3. **Gradient w.r.t Input (Backpropagate through convolution):**
    * To compute the gradient with respect to the input to this layer, you will convolve the **kernels** with the **gradient of the loss with respect to the output**:
    ∂L/∂X₂ = convolve(∂L/∂Z₂, W₂)
    * This result will have shape (10,3,29,28), matching the input shape.

4. **Gradient w.r.t Bias:**
    * You compute the gradient with respect to the bias by summing the gradient w.r.t the output over the spatial dimensions:
    ∂L/∂b₂ = ∑∂L/∂Z₂
    * The result will have shape (2,1,1), matching the bias shape.

---

#### **2. Layer 3: Max-Pooling (MP)**

**Input Shape:** (10, 2, 14, 14)  
**Kernel Shape:** (2, 2)  
**Stride:** (3, 3)  
**Output Shape:** (10, 2, 5, 5)

**Forward Pass Recap:**

* You apply max-pooling with a (2,2) kernel and stride (3,3), reducing the input size of (10,2,14,14) to (10,2,5,5).

**Backward Pass (Backpropagation) Steps:**

1. **Gradient w.r.t Output (Loss Gradient):**
    * You start with ∂L/∂Z₃, the gradient of the loss with respect to the output, which has shape (10,2,5,5).

2. **Gradient w.r.t Input (Max-Pooling Gradient):**
    * Max-pooling simply selects the maximum value in each pooling region during the forward pass. During backpropagation, you need to propagate the gradient back to the input at the locations of the maximum values.
    * You use the indices of the maximum values computed during the forward pass and assign the gradient from ∂L/∂Z₃ to the corresponding locations in the input.
    * This operation doesn't require a kernel or convolution. Instead, you **scatter the gradient back** to the locations of the maximum values, and the result will have shape (10,2,14,14), matching the input shape to this layer.

3. **Gradient w.r.t Bias:**
    * Max-pooling has no bias, so you skip this step for max-pooling.

---

#### **3. Layer 4: Fully Connected (FC)**

**Input Shape:** (10, 2, 5, 5)  
**Weight Shape:** (32, 50)  
**Bias Shape:** (32, 1)  
**Output Shape:** (32, 10)

**Forward Pass Recap:**

* You flatten the input (10,2,5,5) into a vector of shape (10,50), then multiply by the weights (32,50) and add the bias to produce the output (32,10).

**Backward Pass (Backpropagation) Steps:**

1. **Gradient w.r.t Output (Loss Gradient):**
    * You start with ∂L/∂Z₄, the gradient of the loss with respect to the output, which has shape (32,10).

2. **Gradient w.r.t Weights:**
    * To compute the gradient with respect to the weights, you multiply the **transposed input** (50,10) with the gradient of the loss ∂L/∂Z₄:
    ∂L/∂W₄ = ∂L/∂Z₄ ⋅ X₄ᵀ
    * The result will have shape (32,50), matching the shape of the weights.

3. **Gradient w.r.t Input (Backpropagate through FC):**
    * To compute the gradient with respect to the input to this layer, you multiply the **gradient of the loss** ∂L/∂Z₄ with the **transposed weights**:
    ∂L/∂X₄ = W₄ᵀ ⋅ ∂L/∂Z₄
    * The result will have shape (10,2,5,5), matching the input shape to this layer.

4. **Gradient w.r.t Bias:**
    * To compute the gradient with respect to the bias, you sum the gradient of the loss over the batch:
    ∂L/∂b₄ = ∑∂L/∂Z₄
    * The result will have shape (32,1), matching the bias shape.
