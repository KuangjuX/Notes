# Linear Attention

$$
O = softmax(QK^T)V
$$

传统的 Attention 因为 softmax 没办法先算右边的矩阵乘，所以整个复杂度变成了 O($n^2$)，由于 d << n，如果可以先算右边的矩阵乘整体复杂度可以变成O(n)。

$$
Attention(Q, K, V)_i = \sum_{j=1}^n \frac{e^{{q_i}^T k_j}}{\sum_{j=1}^n e^{{q_i}^T k_j}} v_j
$$

一般化定义为：

$$
Attention(Q, K, V)_i = \sum_{j=1}^n \frac{k({q_i}^T k_j)}{\sum_{j=1}^n k({q_i}^T k_j)} v_j
$$

其中核函数可以表示为：

$$
k(x, y) = <\phi(x), \phi(y)>
$$

将 softamx 替换成核函数，Attention 可以被简化为：

$$
Attention(Q, K, V)_i = \frac{\sum_{j=1}^n \phi(q^t)\phi(k_i)^Tv_i}{\sum_{j=1}^n \phi(q^t)\phi(k_i)^T} = \frac{\phi(q^t) \sum_{j=1}^n \phi(k_i)^Tv_i}{\phi(q^t)\sum_{j=1}^n\phi(k_i)^T}
$$

## Gated Linear Attention

针对 Linear Attention 做硬件优化。

- Occupancy.
- Specialized compute units.
- Memory hierarchy.

![](LinearAttention/fwd_pass.png)

Flash Linear Attention 算法有一个 materialize 参数来控制是否要冲计算 S，无论是否要重计算 S 都要分块加载 Q, K, V 到共享内存中，然后可以重用共享内存上的块状 Tensor 来避免多次加载 HBM。

当 materialize 为 True 时，当 $ Q[n] $ 被加载到 SRAM 时，$ Q[n]S $ 和 $ (Q[n]K^T[n] \bigotimes MV)[n] $ 可以在片上计算，避免再次加载 $O[n]$。

当 materialize 为 False 时，算法首先在 HBM 中把块间递归的结果存下来，然后将所有 $S[n]$ 都并行计算在 HBM 中，该算法有更好的并行性，但增加了内存占用。非重计算版本顺序计算 $ S[n] $，并使用 SRAM 暂时存储 $ S[n] $。这种策略在内存上更高效，但缺乏序列级别的并行性。

![](LinearAttention/fig1.png)

materialize 为 False 的情况下，Q，K，V 都是从 HBM 加载到 SRAM 上，每次会计算出一个新的隐藏状态 S，S 一直存储在 SRAM 上面，整体计算是串行的。对于 materialize 为 True 的情况，首先计算 KV 酸楚 S 并将 S 保存到 HBM 上，这部分是串行的，计算玩 S 后可以通过 CHunk 并行计算出 $ O_i $。

![](LinearAttention/fig2.png)