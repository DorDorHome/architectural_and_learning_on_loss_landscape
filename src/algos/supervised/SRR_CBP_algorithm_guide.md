
# Soft rank-restory CBP and its variants

## Soft Rank-Restoring Continual Backpropagation (SRR-CBP)



\begin{algorithm}
\caption{Soft Rank-Restoring Continual Backpropagation (SRR-CBP)}
\begin{algorithmic}
  \Require step size $\alpha$, replacement rate $\rho$, EMA decay $\eta$,
           maturity threshold $M$, initial ASO penalty $\lambda_0$, decay rate $\gamma$,
           layer activation function $\phi$ (e.g., ReLU).

  \State Initialize network weights $W \sim \mathcal{D}_{\mathrm{init}}$ \Comment{Layer weights $W^{(\ell)} \in \mathbb{R}^{n_\ell \times d_\ell}$}
  \State Initialize ages $a_{\ell,i} \gets 0$, EMAs $f_{\ell,i},u_{\ell,i},z_{\ell,i} \gets 0$

  \For{$t \gets 1$ \To $T$}
    \State Perform forward pass, save preactivations $A^{(\ell)} = W^{(\ell)} H^{(\ell-1)} \in \mathbb{R}^{n_\ell \times m}$
    \State Save post-activations $X^{(\ell)} = \phi(A^{(\ell)}) \in \mathbb{R}^{n_\ell \times m}$
    
    \State $\mathcal{L}_{\mathrm{task}} \gets \mathcal{L}(W; x_t, y_t)$
    \State $\mathcal{L}_{\mathrm{ASO}} \gets 0$

    \For{$\ell \gets 1$ \To $L$}
      \Comment{Compute Asymmetric Soft Orthogonality (ASO) Loss}
      \State $\mathcal{M}_\ell \gets \{i \mid a_{\ell,i} \ge M\}$ \Comment{Indices of mature units}
      \State $\mathcal{Y}_\ell \gets \{i \mid a_{\ell,i} < M\}$  \Comment{Indices of young units}
      
      \If{$|\mathcal{Y}_\ell| > 0$ and $|\mathcal{M}_\ell| > 0$}
        \State $A_{\mathrm{mature}} \gets \mathrm{stop\_gradient}\big(A^{(\ell)}_{\mathcal{M}_\ell, :}\big)$ \Comment{Crucial: Do not backprop into mature units}
        \For{$i \in \mathcal{Y}_\ell$}
          \State $a_i \gets A^{(\ell)}_{i, :}$ \Comment{Shape: $1 \times m$}
          \State $\mathcal{L}_{\mathrm{ASO}} \gets \mathcal{L}_{\mathrm{ASO}} + \frac{\gamma^{a_{\ell,i}}}{2m^2} \| a_i A_{\mathrm{mature}}^\top \|_2^2$
        \EndFor
      \EndIf
    \EndFor

    \State $\mathcal{L}_{\mathrm{ASO}} \gets \lambda_0 \cdot \mathcal{L}_{\mathrm{ASO}}$ \Comment{Apply global ASO penalty rate}
    \State $\mathcal{L}_{\mathrm{total}} \gets \mathcal{L}_{\mathrm{task}} + \mathcal{L}_{\mathrm{ASO}}$
    \State $g_t \gets \nabla_W \mathcal{L}_{\mathrm{total}}$
    \State $W \gets \mathrm{OptimizerStep}(W, g_t; \alpha)$ \Comment{Automatically orthogonalizes young units}

    \Comment{Utility Tracking and Replacement Phase}
    \For{$\ell \gets 1$ \To $L$}
      \For{$i \gets 1$ \To $n_\ell$}
        \State $a_{\ell,i} \gets a_{\ell,i} + 1$
        \State $h_{\ell,i,t} \gets \frac{1}{m} \sum_{\alpha=1}^m X^{(\ell)}_{i, \alpha}$ \Comment{Mean batch activation for utility tracking}
        \State $f_{\ell,i} \gets \eta f_{\ell,i} + (1-\eta) h_{\ell,i,t}$
        \State $\widehat f_{\ell,i} \gets f_{\ell,i} / (1 - \eta^{a_{\ell,i}})$
        \State $S^{\mathrm{out}}_{\ell,i} \gets \sum_{k=1}^{n_{\ell+1}} |W^{(\ell+1)}_{k,i}|$
        \State $S^{\mathrm{in}}_{\ell,i}  \gets \sum_{j=1}^{d_\ell} |W^{(\ell)}_{i,j}|$
        \State $c^{\mathrm{inst}}_{\ell,i} \gets | h_{\ell,i,t} - \widehat f_{\ell,i} | \cdot S^{\mathrm{out}}_{\ell,i}$
        \State $z_{\ell,i} \gets \eta z_{\ell,i} + (1-\eta) c^{\mathrm{inst}}_{\ell,i}$
        \State $a^{\mathrm{adapt}}_{\ell,i} \gets (S^{\mathrm{in}}_{\ell,i})^{-1}$
        \State $y_{\ell,i} \gets c^{\mathrm{inst}}_{\ell,i} \cdot a^{\mathrm{adapt}}_{\ell,i}$
        \State $u_{\ell,i} \gets \eta u_{\ell,i} + (1-\eta) y_{\ell,i}$
        \State $\widehat u_{\ell,i} \gets u_{\ell,i} / (1 - \eta^{a_{\ell,i}})$
      \EndFor

      \State $r_\ell \gets \lceil \rho\,n_\ell \rceil$
      \State $S_\ell \gets \mathrm{SelectLow}_{r_\ell}\big( \{ \widehat u_{\ell,i} \}_{i \in \mathcal{M}_\ell} \big)$ \Comment{Select least useful mature units}

      \For{$i \in S_\ell$} \Comment{Reset Unit}
        \Comment{1. Bias Transfer}
        \For{$k \gets 1$ \To $n_{\ell+1}$}
          \State $b^{(\ell+1)}_k \gets b^{(\ell+1)}_k + \widehat f_{\ell,i} \cdot W^{(\ell+1)}_{k,i}$
        \EndFor

        \Comment{2. Standard Random Initialization}
        \State $w_{\mathrm{dir}} \sim \mathcal{D}_{\mathrm{init}}$ \Comment{Shape: $1 \times d_\ell$}
        \State $W^{(\ell)}_{i, :} \gets w_{\mathrm{dir}}$ \Comment{Assign incoming weights directly to weight matrix row}
        \State $b^{(\ell)}_i \gets 0$

        \Comment{3. Zero outgoing weights and Reset Stats}
        \For{$k \gets 1$ \To $n_{\ell+1}$}
          \State $W^{(\ell+1)}_{k,i} \gets 0$
        \EndFor
        \State $a_{\ell,i} \gets 0$, $f_{\ell,i} \gets 0$, $u_{\ell,i} \gets 0$, $z_{\ell,i} \gets 0$
      \EndFor
    \EndFor
  \EndFor
\end{algorithmic}
\end{algorithm}



## SRR-CBP-E (With Empirical Variance Scaling)




\begin{algorithm}
\caption{Soft Rank-Restoring Continual Backpropagation with Empirical Energy Budget (SRR-CBP-E)}
\begin{algorithmic}
  \Require step size $\alpha$, replacement rate $\rho$, EMA decay $\eta$,
           maturity threshold $M$, feature variance decay $\beta$, 
           initial ASO penalty $\lambda_0$, decay rate $\gamma$,
           layer activation function $\phi$ (e.g., ReLU),
           stability constant $\epsilon \gets 10^{-8}$.

  \State Initialize network weights $W \sim \mathcal{D}_{\mathrm{init}}$ \Comment{Layer weights $W^{(\ell)} \in \mathbb{R}^{n_\ell \times d_\ell}$}
  \State Initialize ages $a_{\ell,i} \gets 0$, EMAs $f_{\ell,i},u_{\ell,i},z_{\ell,i} \gets 0$
  \State Initialize layer running variances $v_\ell \in \mathbb{R}^{d_\ell} \gets \mathbf{1}$ \Comment{$\mathcal{O}(d)$ instead of $\mathcal{O}(d^2)$}

  \For{$t \gets 1$ \To $T$}
    \State Perform forward pass, save features $H^{(\ell-1)}$
    \State Save preactivations $A^{(\ell)} = W^{(\ell)} H^{(\ell-1)} \in \mathbb{R}^{n_\ell \times m}$
    \State Save post-activations $X^{(\ell)} = \phi(A^{(\ell)}) \in \mathbb{R}^{n_\ell \times m}$
    
    \State $\mathcal{L}_{\mathrm{task}} \gets \mathcal{L}(W; x_t, y_t)$
    \State $\mathcal{L}_{\mathrm{ASO}} \gets 0$

    \For{$\ell \gets 1$ \To $L$}
      \State Update variance trace: $v_\ell \gets \beta\, v_\ell + (1-\beta) \frac{1}{m} \sum_{\alpha=1}^m H^{(\ell-1)}_{:,\alpha} \odot H^{(\ell-1)}_{:,\alpha}$ \Comment{Hadamard (element-wise) square}
      
      \Comment{Compute Asymmetric Soft Orthogonality (ASO) Loss}
      \State $\mathcal{M}_\ell \gets \{i \mid a_{\ell,i} \ge M\}$ \Comment{Indices of mature units}
      \State $\mathcal{Y}_\ell \gets \{i \mid a_{\ell,i} < M\}$  \Comment{Indices of young units}
      
      \If{$|\mathcal{Y}_\ell| > 0$ and $|\mathcal{M}_\ell| > 0$}
        \State $A_{\mathrm{mature}} \gets \mathrm{stop\_gradient}\big(A^{(\ell)}_{\mathcal{M}_\ell, :}\big)$ \Comment{Crucial: Do not backprop into mature units}
        \For{$i \in \mathcal{Y}_\ell$}
          \State $a_i \gets A^{(\ell)}_{i, :}$ \Comment{Shape: $1 \times m$}
          \State $\mathcal{L}_{\mathrm{ASO}} \gets \mathcal{L}_{\mathrm{ASO}} + \frac{\gamma^{a_{\ell,i}}}{2m^2} \| a_i A_{\mathrm{mature}}^\top \|_2^2$
        \EndFor
      \EndIf
    \EndFor

    \State $\mathcal{L}_{\mathrm{ASO}} \gets \lambda_0 \cdot \mathcal{L}_{\mathrm{ASO}}$ \Comment{Apply global ASO penalty rate}
    \State $\mathcal{L}_{\mathrm{total}} \gets \mathcal{L}_{\mathrm{task}} + \mathcal{L}_{\mathrm{ASO}}$
    \State $g_t \gets \nabla_W \mathcal{L}_{\mathrm{total}}$
    \State $W \gets \mathrm{OptimizerStep}(W, g_t; \alpha)$ \Comment{Automatically orthogonalizes young units}

    \Comment{Utility Tracking and Replacement Phase}
    \For{$\ell \gets 1$ \To $L$}
      \For{$i \gets 1$ \To $n_\ell$}
        \State $a_{\ell,i} \gets a_{\ell,i} + 1$
        \State $h_{\ell,i,t} \gets \frac{1}{m} \sum_{\alpha=1}^m X^{(\ell)}_{i, \alpha}$ \Comment{Mean batch activation for utility tracking}
        \State $f_{\ell,i} \gets \eta f_{\ell,i} + (1-\eta) h_{\ell,i,t}$
        \State $\widehat f_{\ell,i} \gets f_{\ell,i} / (1 - \eta^{a_{\ell,i}})$
        \State $S^{\mathrm{out}}_{\ell,i} \gets \sum_{k=1}^{n_{\ell+1}} |W^{(\ell+1)}_{k,i}|$
        \State $S^{\mathrm{in}}_{\ell,i}  \gets \sum_{j=1}^{d_\ell} |W^{(\ell)}_{i,j}|$
        \State $c^{\mathrm{inst}}_{\ell,i} \gets | h_{\ell,i,t} - \widehat f_{\ell,i} | \cdot S^{\mathrm{out}}_{\ell,i}$
        \State $z_{\ell,i} \gets \eta z_{\ell,i} + (1-\eta) c^{\mathrm{inst}}_{\ell,i}$
        \State $a^{\mathrm{adapt}}_{\ell,i} \gets (S^{\mathrm{in}}_{\ell,i})^{-1}$
        \State $y_{\ell,i} \gets c^{\mathrm{inst}}_{\ell,i} \cdot a^{\mathrm{adapt}}_{\ell,i}$
        \State $u_{\ell,i} \gets \eta u_{\ell,i} + (1-\eta) y_{\ell,i}$
        \State $\widehat u_{\ell,i} \gets u_{\ell,i} / (1 - \eta^{a_{\ell,i}})$
      \EndFor

      \State $r_\ell \gets \lceil \rho\,n_\ell \rceil$
      \State $S_\ell \gets \mathrm{SelectLow}_{r_\ell}\big( \{ \widehat u_{\ell,i} \}_{i \in \mathcal{M}_\ell} \big)$ \Comment{Select least useful mature units}

      \For{$i \in S_\ell$} \Comment{Reset Unit}
        \Comment{1. Bias Transfer}
        \For{$k \gets 1$ \To $n_{\ell+1}$}
          \State $b^{(\ell+1)}_k \gets b^{(\ell+1)}_k + \widehat f_{\ell,i} \cdot W^{(\ell+1)}_{k,i}$
        \EndFor

        \Comment{2. Random Direction and Empirical Scaling (Section 1)}
        \State $w_{\mathrm{dir}} \sim \mathcal{N}(0, I_{d_\ell})$ \Comment{Shape: $d_\ell \times 1$}
        \State $a_{\mathrm{raw}} \gets w_{\mathrm{dir}}^\top H^{(\ell-1)}$ \Comment{Shape: $1 \times m$}
        \State $a_{\mathrm{centered}} \gets a_{\mathrm{raw}} - \mathrm{mean}(a_{\mathrm{raw}})$
        
        \Comment{Calculate empirical surviving energy with Div-By-Zero safeguard}
        \State $\hat{\chi}_{\mathrm{emp}} \gets \max\left( \frac{1}{m} \sum_{\alpha=1}^m \phi(a_{\mathrm{centered}, \alpha})^2, \epsilon \right)$
        
        \State $v_{\mathrm{alloc}} \gets \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} (v_\ell)_j$ \Comment{Target layer energy budget}
        \State $\gamma_{\mathrm{scale}} \gets \sqrt{v_{\mathrm{alloc}} / \hat{\chi}_{\mathrm{emp}}}$
        
        \State $W^{(\ell)}_{i, :} \gets \gamma_{\mathrm{scale}} \cdot w_{\mathrm{dir}}^\top$ \Comment{Assign incoming weights directly to weight matrix row}
        \State $b^{(\ell)}_i \gets -\mathrm{mean}(a_{\mathrm{raw}}) \cdot \gamma_{\mathrm{scale}}$ \Comment{Centered bias}

        \Comment{3. Zero outgoing weights and Reset Stats}
        \For{$k \gets 1$ \To $n_{\ell+1}$}
          \State $W^{(\ell+1)}_{k,i} \gets 0$
        \EndFor
        \State $a_{\ell,i} \gets 0$, $f_{\ell,i} \gets 0$, $u_{\ell,i} \gets 0$, $z_{\ell,i} \gets 0$
      \EndFor
    \EndFor
  \EndFor
\end{algorithmic}
\end{algorithm}



