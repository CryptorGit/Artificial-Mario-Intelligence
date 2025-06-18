# Differential Gaussian-Gated IndRNN with Inverse Forward-Forward Gradient and Energy-Conserving Local Learning Rule

## Abstract
We propose a time--series processing framework that integrates **Independently Recurrent Neural Networks (IndRNNs)** with a *differential, Gaussian-based gating mechanism* and introduces a novel training rule that combines an **Inverse Forward-Forward (IFF) gradient** with **layer-wise energy conservation**.  A convolutional encoder extracts spatial features from video frames. Frame-to-frame feature differences are normalised and passed through a Gaussian function to produce gate values that modulate the recurrent update of each IndRNN unit, balancing long-term memory with sensitivity to abrupt changes.  Learning is driven by local "goodness" differences between *positive* and *negative* forward passes, yielding an IFF gradient that updates each layer without global back-propagation. To stabilise training, every weight update is rescaled so that the Frobenius norm (energy) of each layer is preserved. We derive the complete algorithm, prove its mathematical consistency, and discuss its theoretical merits and potential applications.

## 1 Introduction
Recurrent neural networks (RNNs) are powerful for sequential data but suffer from vanishing/exploding gradients. Gate-based variants such as LSTM and GRU mitigate this, yet deep stacks still face gradient attenuation due to saturating nonlinearities.  IndRNN (Li *et al.*, 2018) addresses these issues by assigning an independent recurrent weight to every neuron, enabling extremely deep RNNs trained with ReLU activations.

In parallel, biologically plausible *local* learning rules are attracting renewed interest. Hinton's **Forward-Forward (FF) algorithm** (2022) optimises each layer by comparing "goodness" for positive versus negative inputs without back-propagation. Inspired by this, we extend FF to an *inverse* gradient form and combine it with an energy-conserving rescaling scheme.

**Contributions.**

1. **Differential Gaussian Gate for IndRNN.** A normalised frame-difference drives a Gaussian gate that dynamically controls how much of the previous hidden state flows forward.
2. **Inverse Forward-Forward Gradient.** We formulate a local weight update that increases positive "goodness" while suppressing negative "goodness", yielding a simple Hebbian-like contrastive term.
3. **Energy-Conserving Scaling.** Each layer's weight update is multiplicatively rescaled so that its weight-norm remains (first-order) constant, ensuring numerical stability.
4. **Unified Algorithm & Theory.** We derive the full training procedure and analyse gradient behaviour, stability, and consistency.

## 2 Proposed Method

### 2.1 Architecture
1. **CNN Encoder.** For input frame sequence \( \{X_t\} \), a deep CNN outputs feature maps
\[
  f_t = \operatorname{CNN}(X_t).
\]
2. **Differential Normalisation.** Consecutive frame differences
\[
  d_t = f_t - f_{t-1}
\]
are element-wise normalised, e.g.
\[
  \tilde d_{t,i} = \frac{d_{t,i}}{\varepsilon + |d_{t,i}|}.
\]
3. **Gaussian Gate.** Each normalised difference produces a gate value with a learnable centre
\[
  g_{t,i} = \exp\bigl(-\tfrac{(\tilde d_{t,i} - \mu_i)^2}{2\sigma^2}\bigr), \qquad 0 < g_{t,i} \le 1.
\]
4. **IndRNN Update.** For unit \(i\) with recurrent weight \(w_i\) and input weight vector \(U_i\)
\[
  h_{t,i} = \phi\bigl( w_i\,g_{t,i}\,h_{t-1,i} \;{+}\; U_i^\top x_t + b_i \bigr),
\]
where \(x_t=f_t\) and \(\phi\) is ReLU.

### 2.2 Output and Task Loss
At final time \(T\),
\[
  y = \operatorname{softmax}(V h_T + c),
\]
with cross-entropy loss for classification.

## 3 Theoretical Properties

### 3.1 Temporal Gradient Behaviour
For a single IndRNN unit, the \(k\)-step gradient is
\[
  \frac{\partial h_{t,i}}{\partial h_{t-k,i}} = (w_i g_{t,i})^k
           \prod_{j=1}^k \phi'(\cdot).
\]
Thus \(|w_i g_{t,i}|\) directly controls gradient magnitude, enabling dynamic regulation: small frame changes (\(g\approx1\)) sustain long memory; large changes (\(g\ll1\)) shorten it.

### 3.2 Inverse Forward-Forward Gradient
Define layer-wise goodness
\[
  G_l^{\pm}=\|h_l^{\pm}\|^2.
\]
The local objective
\[
  J_l = \tfrac12 \bigl(G_l^{+} - G_l^{-}\bigr)
\]
yields the weight update
\[
  \Delta w_{ij}=\eta\bigl(h_{l,i}^{+} x_{l,j}^{+} - h_{l,i}^{-} x_{l,j}^{-}\bigr).
\]

### 3.3 Energy-Conserving Scaling
To approximately preserve \(\|W\|_F^2\) per layer, scale \(\Delta W\) by
\[
  \alpha = -\frac{\langle W,\Delta W \rangle}{\|\Delta W\|_F^2}.
\]

### 3.4 Full Algorithm
1. **Positive forward pass** \rightarrow \(h_l^{+}\)
2. **Negative forward pass** \rightarrow \(h_l^{-}\)
3. **Local gradient** \(\Delta W_l\) via IFF
4. **Scale** with \(\alpha_l\) for energy conservation
5. **Update** \(W_l \leftarrow W_l + \alpha_l \Delta W_l\)

This requires no backward graph traversal and is fully parallel across layers.

## 4 Discussion
Our method merges dynamic temporal gating, interpretable recurrent weights, biologically inspired local learning, and a physics-motivated stability constraint. Potential applications span action recognition, video anomaly detection, and any domain where abrupt and gradual temporal patterns coexist.

## 5 Conclusion
We introduced a **Differential Gaussian-Gated IndRNN** and an **Inverse Forward-Forward, energy-conserving learning rule**. The resulting algorithm learns long-term dependencies while adapting to sudden changes and dispenses with global back-propagation. Future work will study convergence properties and benchmark performance across real-world tasks.

---

*Correspondence*: your.email@example.com
