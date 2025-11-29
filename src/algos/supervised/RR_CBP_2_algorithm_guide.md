# RR-CBP Algorithm Guide (Mathematical + Pseudocode)

This document is an **algorithmic guide** for implementing

* **Continual Backpropagation (CBP)**,
* **Rank-Restoring CBP (RR-CBP)**, and
* **RR-CBP with Σ-aware energy control (RR-CBP-E)**,

in *any* deep-learning framework (PyTorch, JAX, TensorFlow, custom C++/CUDA, etc.).

It focuses on **equations, invariants, and pseudocode**, and avoids framework-specific
engineering. A coding agent can use this as a blueprint and adapt it to an existing
training loop and module structure.

---

## 1. Base model and notation

* Layers indexed by $\ell = 1,\dots,L$.

* Width of layer $\ell$: $N_\ell$; input dimension to layer $\ell$ is
  $d_\ell = N_{\ell-1}$.

* For a batch (or EMA across recent data) at layer $\ell$, define the upstream
  features (column-wise) as
  $$
  H^{(\ell-1)} \in \mathbb{R}^{d_\ell \times m},
  $$
  where column $\alpha$ is $h^{(\ell-1)}(x^\alpha)$ and $m$ is batch size.

* **Incoming weights** at layer $\ell$ are stacked as a matrix
  $$
  W^{(\ell)} \in \mathbb{R}^{N_\ell \times d_\ell},
  $$
  where row $w^{(\ell)\top}_i$ is the incoming weight vector of unit $i$.

* **Preactivations and activations**:
  $$
  A^{(\ell)} = W^{(\ell)} H^{(\ell-1)},
  \qquad
  X^{(\ell)} = \phi\big(A^{(\ell)}\big),
  $$
  with $\phi$ applied elementwise.

Throughout, we often work at a **fixed layer** and drop the layer index, writing
$W \in \mathbb{R}^{N \times d}$ and $H \in \mathbb{R}^{d \times m}$.

---

## 2. Empirical covariance and Σ-geometry

### 2.1 Empirical covariance

For a fixed layer, define the empirical (or EMA) covariance of upstream features
$$
\Sigma
:=  \frac{1}{m} H H^\top
\in  \mathbb{R}^{d \times d}.
$$

We assume either $\Sigma \succ 0$ or, more generally, $\Sigma \succeq 0$ with support
subspace $S = \operatorname{im}(\Sigma)$.

### 2.2 Σ-inner product and norm

For vectors $u,v \in \mathbb{R}^d$ define
$$
\langle u, v \rangle_\Sigma := u^\top \Sigma v,
\qquad
|u|_\Sigma := \sqrt{u^\top \Sigma u}.
$$

When working in this geometry, we **identify** each incoming row $w_i^\top$ of $W$
with a column $w_i \in \mathbb{R}^d$.

### 2.3 Kept set and restricted Gram

Let $K$ be the index set of **kept** units at a layer, $|K| = k$. Collect their
incoming vectors (as columns) into
$$
V = [v_1,\dots,v_k] \in \mathbb{R}^{d \times k},
\qquad
v_j := w_{i_j},; i_j \in K.
$$
The **restricted unit Gram** (in the Σ-geometry) is
$$
G_{\mathrm{keep}} := V^\top \Sigma V \in \mathbb{R}^{k \times k}.
$$
Equivalently, in row form let $W_{\mathrm{keep}}$ be the submatrix of $W$ with rows
$i \in K$. Then
$$
G_{\mathrm{keep}}
= W_{\mathrm{keep}} \Sigma W_{\mathrm{keep}}^\top.
$$

### 2.4 Σ-orthogonal projector onto span$(V)$

Assuming $\Sigma \succ 0$ and $V$ has full column rank, the orthogonal projector in
Σ-geometry onto $\mathrm{span}(V)$ is
$$
\boxed{; P_\Sigma
= V,(V^\top \Sigma V)^{-1} V^\top \Sigma ;} \tag{2.1}
$$

**Whitened form.** Let $T = \Sigma^{1/2}$ be any SPD square root, and define
$$
\widetilde V := T V.
$$
Then the Euclidean projector in whitened coordinates is
$$
\Pi := \widetilde V, (\widetilde V^\top \widetilde V)^{-1} \widetilde V^\top,
$$
so that
$$
\boxed{; P_\Sigma = \Sigma^{-1/2} \Pi \Sigma^{1/2} ;} \tag{2.2}
$$
which is equivalent to (2.1).

When $\Sigma$ is only PSD, restrict all operations to its support subspace $S$.

---

## 3. Pre-activation Gram and whitening

At a fixed layer, preactivations are
$$
A = W H \in \mathbb{R}^{N \times m}.
$$
Define the **unit-by-unit preactivation covariance (unit Gram)**
$$
G_{\mathrm{pre}} := \frac{1}{m} A A^\top = W \Sigma W^\top \in \mathbb{R}^{N \times N}.
$$
Introduce the **whitened incoming matrix**
$$
\widetilde W := W \Sigma^{1/2} \in \mathbb{R}^{N \times d},
$$
so that
$$
G_{\mathrm{pre}} = \widetilde W \widetilde W^\top.
$$

If $W_{\mathrm{keep}}$ is the kept submatrix and $w$ is the new row (in column form),
its whitened counterparts are
$$
\widetilde W_{\mathrm{keep}} = W_{\mathrm{keep}} \Sigma^{1/2},
\qquad
\tilde w = w^\top \Sigma^{1/2}.
$$

If $w$ is $\Sigma$-orthogonal to all kept rows and $|w|*\Sigma^2 = q$, then in
whitened coordinates $\tilde w$ is a Euclidean orthogonal row with
$|\tilde w|*2^2 = q$, and
$$
\widetilde G*{\mathrm{pre}}^{\mathrm{new}}
:= \widetilde W*{\mathrm{new}} \widetilde W_{\mathrm{new}}^\top
===============================================================

\begin{bmatrix}
\widetilde G_{\mathrm{pre}}^{\mathrm{keep}} & 0 \
0 & q
\end{bmatrix}.
$$
This governs rank and conditioning changes.

---

## 4. Base Continual Backprop (CBP)

CBP maintains **per-unit statistics** and occasionally replaces low-utility,
mature units. For each unit $(\ell,i)$ we track:

* Age $a_{\ell,i}$ (number of steps since last reset),
* EMA of activations $f_{\ell,i}$ and its bias-corrected mean $\widehat f_{\ell,i}$,
* Utility EMA $u_{\ell,i}$ and its bias-corrected version $\widehat u_{\ell,i}$,
* Auxiliary EMA $z_{\ell,i}$,
* Local quantities $S^{\mathrm{in}}*{\ell,i}$, $S^{\mathrm{out}}*{\ell,i}$.

### 4.1 CBP per-step updates

For each training step $t = 1,\dots,T$:

1. Standard step: forward pass, loss, backward pass, optimizer update.
2. For each layer $\ell$ and unit $i$:

   * Increment age: $a_{\ell,i} \leftarrow a_{\ell,i} + 1$.
   * Update EMA of activations:
     $$
     f_{\ell,i} \leftarrow \eta f_{\ell,i} + (1-\eta) h_{\ell,i,t},
     \qquad
     \widehat f_{\ell,i} = \frac{f_{\ell,i}}{1 - \eta^{a_{\ell,i}}}.
     $$
   * Compute outgoing and incoming norms (e.g. L1):
     $$
     S^{\mathrm{out}}*{\ell,i} = \sum*{k} |w_{\ell,i,k}|,
     \qquad
     S^{\mathrm{in}}*{\ell,i}  = \sum*{j} |w_{\ell-1,j,i}|.
     $$
   * Instantaneous contribution:
     $$
     c^{\mathrm{inst}}*{\ell,i} = |h*{\ell,i,t} - \widehat f_{\ell,i}|, S^{\mathrm{out}}_{\ell,i}.
     $$
   * Auxiliary EMA:
     $$
     z_{\ell,i} \leftarrow \eta z_{\ell,i} + (1-\eta) c^{\mathrm{inst}}_{\ell,i}.
     $$
   * Adaptivity factor and contribution:
     $$
     a^{\mathrm{adapt}}*{\ell,i} = (S^{\mathrm{in}}*{\ell,i})^{-1},
     \qquad
     y_{\ell,i} = c^{\mathrm{inst}}*{\ell,i}, a^{\mathrm{adapt}}*{\ell,i}.
     $$
   * Utility EMA:
     $$
     u_{\ell,i} \leftarrow \eta u_{\ell,i} + (1-\eta) y_{\ell,i},
     \qquad
     \widehat u_{\ell,i} = \frac{u_{\ell,i}}{1 - \eta^{a_{\ell,i}}}.
     $$

### 4.2 CBP replacement rule (high level)

* Maturity threshold $M$;
* Replacement rate $\rho$.

For each layer $\ell$:

1. Mature set:
   $$
   \mathcal{E}*\ell := { i \mid a*{\ell,i} \ge M }.
   $$
2. Replacement count: $r_\ell := \lceil \rho,N_\ell \rceil$.
3. Replacement indices: let $S_\ell$ be the indices of the $r_\ell$ smallest
   $\widehat u_{\ell,i}$ among $i \in \mathcal{E}_\ell$.
4. Kept indices: $K_\ell := {1,\dots,N_\ell} \setminus S_\ell$.
5. For each $i \in S_\ell$:

   * **Bias transfer** to next layer biases using $\widehat f_{\ell,i}$ and
     outgoing weights.
   * **Reinitialize incoming** weights (in vanilla CBP: random from init distribution).
   * **Zero outgoing** weights.
   * Reset $a_{\ell,i}, f_{\ell,i}, u_{\ell,i}, z_{\ell,i}$ to $0$.

### 4.3 CBP pseudocode (framework-agnostic)

```pseudo
Algorithm CBP (Continual Backpropagation)
Require: step size α, replacement rate ρ, EMA decay η, maturity threshold M

Initialize weights θ ~ D_init
Initialize ages a_{ℓ,i} ← 0 and EMAs f_{ℓ,i}, u_{ℓ,i}, z_{ℓ,i} ← 0 for all (ℓ,i)

for t = 1 to T do
  # standard training step
  y_hat_t ← f_θ(x_t)
  L_t ← L(θ; x_t, y_t)
  g_t ← ∇_θ L_t
  θ ← OptimizerStep(θ, g_t; α)

  for ℓ = 1 to L do
    # per-unit statistics
    for i = 1 to N_ℓ do
      a_{ℓ,i} ← a_{ℓ,i} + 1
      f_{ℓ,i} ← η f_{ℓ,i} + (1-η) h_{ℓ,i,t}
      f_hat_{ℓ,i} ← f_{ℓ,i} / (1 - η^{a_{ℓ,i}})

      S_out_{ℓ,i} ← sum_k |w_{ℓ,i,k}|
      S_in_{ℓ,i}  ← sum_j |w_{ℓ-1,j,i}|

      c_inst_{ℓ,i} ← |h_{ℓ,i,t} - f_hat_{ℓ,i}| * S_out_{ℓ,i}
      z_{ℓ,i}      ← η z_{ℓ,i} + (1-η) c_inst_{ℓ,i}

      a_adapt_{ℓ,i} ← (S_in_{ℓ,i})^{-1}
      y_{ℓ,i}      ← c_inst_{ℓ,i} * a_adapt_{ℓ,i}

      u_{ℓ,i} ← η u_{ℓ,i} + (1-η) y_{ℓ,i}
      u_hat_{ℓ,i} ← u_{ℓ,i} / (1 - η^{a_{ℓ,i}})
    end for

    E_ℓ ← { i | a_{ℓ,i} ≥ M }
    r_ℓ ← ceil(ρ * N_ℓ)
    S_ℓ ← indices of r_ℓ smallest u_hat_{ℓ,i} over i ∈ E_ℓ
    K_ℓ ← {1,…,N_ℓ} \ S_ℓ

    for i in S_ℓ do
      # (1) Bias transfer (layer-specific implementation)
      TransferBiasFromUnit(ℓ, i, f_hat_{ℓ,i}, outgoing_weights_of_unit)

      # (2) Reinit incoming weights (vanilla CBP: random)
      ReinitIncomingWeightsRandom(ℓ, i)

      # (3) Zero outgoing weights
      ZeroOutgoingWeights(ℓ, i)

      # (4) Reset stats
      a_{ℓ,i} ← 0; f_{ℓ,i} ← 0; u_{ℓ,i} ← 0; z_{ℓ,i} ← 0
    end for
  end for
end for
```

---

## 5. Rank-Restoring CBP (RR-CBP)

RR-CBP keeps the **CBP utility logic** but replaces the **random reinit** with a
**Σ-aware geometric reinit** that restores rank and improves conditioning.

### 5.1 Direction in Σ-orthogonal complement (non-saturated case)

Let $W_{\mathrm{keep}}$ be the kept incoming matrix and $P_\Sigma$ the projector (2.1).

1. Draw a random direction $u \sim \mathcal{N}(0, I_d)$.
2. Project into the $\Sigma$-orthogonal complement:
   $$
   \hat w := (I - P_\Sigma) u.
   $$
3. If $|\hat w|*\Sigma > 0$, use this direction and normalize:
   $$
   w*{\mathrm{dir}} := \frac{\hat w}{|\hat w|_\Sigma}.
   $$

This ensures $\langle w_{\mathrm{dir}}, v_j \rangle_\Sigma = 0$ for all kept $v_j$ and
$|w_{\mathrm{dir}}|_\Sigma = 1$.

### 5.2 Saturated case: least-covered direction

If $|\hat w|_\Sigma = 0$ (the $\Sigma$-orthogonal complement is trivial), we use the
**least-covered direction**.

1. Form $V$ from kept unit columns and define
   $$
   M' := \Sigma^{1/2} V V^\top \Sigma^{1/2}.
   $$
2. Let $u_{\min}$ be the eigenvector of $M'$ with **smallest eigenvalue**.
3. Set
   $$
   w_{\mathrm{dir}} := \frac{\Sigma^{-1/2} u_{\min}}
   {|\Sigma^{-1/2} u_{\min}|_\Sigma}.
   $$

This chooses the direction that is **least correlated** with the kept set in
$\Sigma$-geometry and typically improves the smallest eigenvalue of the preactivation
Gram.

### 5.3 Bias centering and outgoing weights

Given a candidate incoming vector $w$ and upstream features $H$:

* Preactivations: $a = w^\top H \in \mathbb{R}^{1 \times m}$.
* Set bias to center $a$:
  $$
  b = -\frac{1}{m} \sum_{\alpha=1}^m a_\alpha
  \quad \text{(or median for robustness)}.
  $$
* Outgoing row: for function preservation, **zero** the outgoing weights of the
  reinitialized unit. Optionally seed with a very small vector in the approximate
  nullspace of the unit’s activations.

### 5.4 RR-CBP pseudocode (reinit block only)

RR-CBP modifies only the **reinit** part of CBP.

```pseudo
Procedure RR_ReinitUnit(ℓ, i, Σ_ℓ, W_keep, H_prev)
  # (1) Compute P_Σ for kept units
  P_Σ ← SigmaProjector(W_keep, Σ_ℓ)

  # (2) Draw direction and project into Σ-orthogonal complement
  u      ← GaussianSample(d_ℓ)         # ~ N(0, I)
  w_hat  ← (I - P_Σ) u

  if ||w_hat||_Σ > 0 then
    w_dir ← w_hat / ||w_hat||_Σ
  else
    w_dir ← LeastCoveredDirection(W_keep, Σ_ℓ)   # normalized in Σ-norm
  end if

  # (3) Bias centering
  a ← w_dir^T H_prev              # preactivations for this unit over batch
  b ← - mean(a)

  # (4) Write incoming weights and bias
  SetIncomingWeights(ℓ, i, w_dir)
  SetBias(ℓ, i, b)

  # (5) Zero outgoing weights
  ZeroOutgoingWeights(ℓ, i)

  # (6) Reset statistics (same as CBP)
  ResetStats(ℓ, i)
end procedure
```

In the **full RR-CBP** algorithm, this procedure replaces
`ReinitIncomingWeightsRandom` in the CBP pseudocode.

---

## 6. Σ-aware energy control (RR-CBP-E)

RR-CBP-E extends RR-CBP by controlling the **Σ-norms** of new units so that:

1. Per-unit **post-activation variance** matches a target (He-like scaling),
2. Layer-level **preactivation energy** obeys an overall budget,
3. In overbudget regimes, we still add nonzero energy with a **rank-restoring floor**.

### 6.1 Activation second-moment constant and per-unit target

Let $h$ denote upstream features with covariance $\Sigma$. For $w \in \mathbb{R}^d$,
$$
a = w^\top h,
\qquad
\operatorname{Var}[a] = |w|_\Sigma^2 = q.
$$
With bias centering, approximate $a \sim \mathcal{N}(0,q)$. Define
$$
\chi_0(\phi) := \mathbb{E}[\phi(z)^2] \quad \text{for } z\sim\mathcal{N}(0,1).
$$
Then
$$
\mathbb{E}[\phi(a)^2] \approx \chi_0(\phi) q.
$$

Let the target post-activation second moment be
$$
v_{\mathrm{tar}} := \frac{1}{d} \operatorname{tr}(\Sigma).
$$
To match this through the nonlinearity, set per-unit target preactivation variance
$$
q_{\mathrm{tar}} := \frac{v_{\mathrm{tar}}}{\chi_0(\phi)},
\quad \Rightarrow \quad
\boxed{; |w|*\Sigma^2 = q*{\mathrm{tar}} ; }.
$$

For ReLU, $\chi_0 = 1/2$, giving the **Σ-aware He rule**
$$
|w|*\Sigma^2 = 2 v*{\mathrm{tar}} = \frac{2}{d} \operatorname{tr}(\Sigma).
$$

### 6.2 Layer energy budget

Define the **layer preactivation energy**
$$
\mathcal{Q}(W;\Sigma) := \sum_{i=1}^N |w_i|*\Sigma^2
= \operatorname{tr}(W \Sigma W^\top).
$$
If each unit targets $q*{\mathrm{tar}}$, the budget is
$$
\boxed{; \mathcal{Q}*{\mathrm{tar}} := N q*{\mathrm{tar}} ; }.
$$
For the kept units $W_{\mathrm{keep}}$ define
$$
\mathcal{Q}*{\mathrm{used}} := \operatorname{tr}(W*{\mathrm{keep}} \Sigma W_{\mathrm{keep}}^\top),
\qquad
\mathcal{Q}*{\mathrm{res}} := \max(\mathcal{Q}*{\mathrm{tar}} - \mathcal{Q}_{\mathrm{used}}, 0).
$$

If we plan to replace $r$ units, a **fair allocation** in the underbudget case is
$$
q_{\mathrm{alloc}} := \min\Big{ q_{\mathrm{tar}}, ; \frac{\mathcal{Q}*{\mathrm{res}}}{r} \Big}.
$$
Each new unit then has
$$
|w|*\Sigma^2 = q_{\mathrm{alloc}}.
$$

### 6.3 Overbudget regime and rank-restoring floor

When $\mathcal{Q}_{\mathrm{res}} = 0$, the layer is already over budget, but we
still want to add nonzero energy to restore rank and improve conditioning.

1. **Rank-restoring floor.** Choose
   $$
   \boxed{; q_{\min} := \tau, \lambda_{\min}(\Sigma),
   \quad \tau \in [10^{-3}, 10^{-1}] ; }.
   $$
   This ensures each new unit adds at least a small positive eigenvalue.

2. **Conditioning target (optional).** Let $\lambda_{\min}^{\mathrm{keep}}$ be the
   smallest eigenvalue of the whitened kept Gram
   $$
   \widetilde G_{\mathrm{pre}}^{\mathrm{keep}}
   := \widetilde W_{\mathrm{keep}} \widetilde W_{\mathrm{keep}}^\top.
   $$
   Choose a target eigenvalue $\lambda_\star$ (e.g.
   $\lambda_\star = \min{1, 2\lambda_{\min}^{\mathrm{keep}}}$) and set
   $$
   q_{\mathrm{alloc}} := \min\big( q_{\mathrm{tar}}, ; \max(q_{\min}, \lambda_\star) \big).
   $$

In whitened coordinates, each new row contributes a block-diagonal eigenvalue
$q_{\mathrm{alloc}}$, lifting the spectrum floor up to
$\min{\lambda_{\min}^{\mathrm{keep}}, q_{\mathrm{alloc}}}$.

### 6.4 RR-CBP-E reinit block (pseudocode)

RR-CBP-E modifies the RR reinit by including a scaling step.

```pseudo
Procedure RR_EnergyAwareReinitUnit(ℓ, i, Σ_ℓ, W_keep, H_prev,
                                   Q_tar_ℓ, Q_used_ℓ, r_ℓ,
                                   χ0, τ)
  # (1) Compute layer targets
  d_ℓ ← input_dim_of_layer(ℓ)
  N_ℓ ← width_of_layer(ℓ)

  v_tar ← (1 / d_ℓ) * trace(Σ_ℓ)
  q_tar ← v_tar / χ0        # χ0 = χ0(φ)
  Q_tar ← N_ℓ * q_tar

  # Q_used_ℓ assumed precomputed from W_keep
  Q_res ← max(Q_tar - Q_used_ℓ, 0)

  if Q_res > 0 then
    q_alloc ← min(q_tar, Q_res / r_ℓ)
  else
    # overbudget: rank-restoring floor and optional eigen-target
    λ_min_Sigma ← smallest_eigenvalue(Σ_ℓ)
    q_min ← τ * λ_min_Sigma

    λ_min_keep ← smallest_eigenvalue( whitened_Gram(W_keep, Σ_ℓ) )
    λ_star     ← min(1, 2 * λ_min_keep)

    q_alloc ← min(q_tar, max(q_min, λ_star))
  end if

  # (2) Direction in Σ-geometry (same as RR-CBP)
  P_Σ ← SigmaProjector(W_keep, Σ_ℓ)
  u      ← GaussianSample(d_ℓ)
  w_hat  ← (I - P_Σ) u

  if ||w_hat||_Σ > 0 then
    w_dir ← w_hat
  else
    w_dir ← LeastCoveredDirection(W_keep, Σ_ℓ)
  end if

  # (3) Scale to match q_alloc in Σ-norm
  s ← ||w_dir||_Σ
  if s > 0 and q_alloc > 0 then
    w_dir ← w_dir * sqrt(q_alloc) / s
  end if

  # (4) Bias centering and outgoing weights (as before)
  a ← w_dir^T H_prev
  b ← - mean(a)
  SetIncomingWeights(ℓ, i, w_dir)
  SetBias(ℓ, i, b)
  ZeroOutgoingWeights(ℓ, i)
  ResetStats(ℓ, i)
end procedure
```

In the **full RR-CBP-E algorithm**, this procedure is called inside the CBP loop
for each unit $i \in S_\ell$ at layer $\ell$.

---

## 7. Summary of key equations (for coding agents)

To implement RR-CBP / RR-CBP-E in any framework, the agent needs to realize:

1. **Empirical covariance (per layer)**
   $$
   \Sigma_\ell = \frac{1}{m} H^{(\ell-1)} H^{(\ell-1)\top}.
   $$

2. **Σ-inner product and norm**
   $$
   \langle u,v \rangle_\Sigma = u^\top \Sigma v,
   \qquad
   |u|_\Sigma^2 = u^\top \Sigma u.
   $$

3. **Σ-orthogonal projector**
   $$
   P_\Sigma = V (V^\top \Sigma V)^{-1} V^\top \Sigma.
   $$

4. **Least-covered direction**
   $$
   M' = \Sigma^{1/2} V V^\top \Sigma^{1/2},
   \quad
   u_{\min} = \arg\min_{|u|*2=1} u^\top M' u,
   \quad
   w \propto \Sigma^{-1/2} u*{\min}.
   $$

5. **Preactivation Gram**
   $$
   G_{\mathrm{pre}} = W \Sigma W^\top = \widetilde W \widetilde W^\top,
   \quad
   \widetilde W = W \Sigma^{1/2}.
   $$

6. **Activation second-moment constant**
   $$
   \chi_0(\phi) = \mathbb{E}[\phi(z)^2], \quad z\sim\mathcal{N}(0,1).
   $$

7. **Per-unit target variance**
   $$
   v_{\mathrm{tar}} = \frac{1}{d} \operatorname{tr}(\Sigma),
   \quad
   q_{\mathrm{tar}} = \frac{v_{\mathrm{tar}}}{\chi_0(\phi)}.
   $$

8. **Layer energy and budget**
   $$
   \mathcal{Q}(W;\Sigma) = \operatorname{tr}(W \Sigma W^\top),
   \quad
   \mathcal{Q}*{\mathrm{tar}} = N q*{\mathrm{tar}},
   $$
   $$
   \mathcal{Q}*{\mathrm{used}} = \operatorname{tr}(W*{\mathrm{keep}} \Sigma W_{\mathrm{keep}}^\top),
   \quad
   \mathcal{Q}*{\mathrm{res}} = \max(\mathcal{Q}*{\mathrm{tar}} - \mathcal{Q}_{\mathrm{used}}, 0).
   $$

9. **Allocated Σ-norm**

   * Underbudget:
     $$q_{\mathrm{alloc}} = \min\big(q_{\mathrm{tar}}, \mathcal{Q}_{\mathrm{res}}/r\big).$$
   * Overbudget:
     $$q_{\mathrm{alloc}} = \min\big(q_{\mathrm{tar}}, \max(q_{\min}, \lambda_\star)\big),
     \quad q_{\min} = \tau\lambda_{\min}(\Sigma).$$

10. **Scaling step**
    $$
    w \leftarrow w \cdot \frac{\sqrt{q_{\mathrm{alloc}}}}{|w|_\Sigma}.
    $$

With these components plus the CBP utility logic, an AI coding agent can integrate
RR-CBP and RR-CBP-E into any existing training codebase.
