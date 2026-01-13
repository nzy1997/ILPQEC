# ILP Model for QEC Decoding

Given:
- Binary parity-check matrix $H \in \{0,1\}^{m \times n}$
- Syndrome $s \in \{0,1\}^m$
- Weights $w \in \mathbb{R}^n$ (typically $w_j = \log((1-p_j)/p_j)$)

**Decision Variables:**
- $e_j \in \{0,1\}$ for $j = 0,...,n-1$ (error indicators)
- $a_i \in \mathbb{Z}_{\ge 0}$ for $i = 0,...,m-1$ (auxiliary for mod-2 linearization)

**Objective:**
$$
\min \sum_{j=0}^{n-1} w_j \cdot e_j
$$
**Constraints (mod-2 linearization):**

$$
\sum_{j=0}^{n-1} H_{i,j} \cdot e_j = s_i + 2 \cdot a_i \quad \text{for } i = 0,...,m-1
$$
