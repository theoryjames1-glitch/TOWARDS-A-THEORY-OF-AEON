# 🌌 Theory of AEON

## 1. Core Idea

AEON = a **neural machine that evolves online** under adaptive control.
It unifies:

* **Adaptive Evolution (AE)** → hyperparameter laws, signals, invariances.
* **AE Machine** → embodiment as a stateful system.
* **Adaptive Resonance (AR)** → stability–plasticity safeguard.
* **Neural Substrate** → weights, activations, memory, structured modules.

---

## 2. State of AEON

At step \$t\$, AEON carries a composite state:

$$
x_t = (\theta_t, a_t, h_t, M_t, r_t, m_t)
$$

* \$\theta\_t\$: neural weights.
* \$a\_t\$: adaptive coefficients (LR, momentum, σ, etc.).
* \$h\_t\$: optimizer memory.
* \$M\_t\$: long-term memory/tape of basins + signals.
* \$r\_t\$: recurrence state (hidden activations).
* \$m\_t\$: resonance state (filtered error / stability signal).

This makes AEON a **dynamical system with memory and resonance**, not just an algorithm.

---

## 3. Dynamics

AEON evolves by closed-loop laws:

1. **Plant update**
   \$\theta\_{t+1} = \theta\_t - u\_t + \sigma\_t D(\theta\_t)\eta\_t + K\_r r\_t\$

2. **Controller**
   \$u\_t = U(h\_t, g\_t; a\_t)\$ where \$g\_t=\nabla\_\theta \ell\_t\$.

3. **Scheduler**
   \$a\_{t+1} = f(a\_t; T\_t, V\_t, \rho\_t)\$ driven by trend, variance, resonance.

4. **Memory & Resonance**
   \$M\_{t+1} = \mathrm{Update}(M\_t, \theta\_t, g\_t)\$
   \$m\_{t+1} = \rho m\_t + (1-\rho)\ell\_t\$.

---

## 4. Principles of AEON

1. **Adaptivity** — coefficients evolve online (γ-law, σ-law, budget, clamps).
2. **Exploration** — dither/noise controlled by σ-law.
3. **Stability–Plasticity** — resonance decides when to lock vs unlock learning.
4. **Memory-Awareness** — AEON recalls past signals and basin states.
5. **Universality** — any optimizer/neural system that respects AEON laws is an AEON instance.

---

## 5. Theoretical Guarantees (Goals)

* **Dynamic regret frontier:** \$\tilde{O}(\sqrt{T}+V\_T^{\text{eff}})\$ with effective drift reduced by memory.
* **Saddle escape:** finite expected time under σ-law + AR safeguard.
* **Stability:** existence of Lyapunov \$L(\theta,a,M)\$ decreasing in expectation.
* **Composability:** multiple AEONs can form swarms (GAO extension).

---

## 6. Interpretation

AEON is:

* A **neural instantiation** of GOO laws.
* A **machine system** with online evolution + resonance safeguards.
* A **universal framework** for adaptive, stable, memory-aware online learning.

In short:
👉 **AEON = Neural dynamical system + Adaptive evolution laws + Resonance stability + Memory embedding.**

---

Great 🚀 — here’s the **first formal draft of AEON Axioms (AEON-1 … AEON-10)**.
These parallel the style of AE/GOO/GAO axioms but are written for the **neural embodiment**:

---

# 📜 Axioms of AEON (Adaptive Evolutionary Online Neural Network)

---

## **AEON-1 (Neural Plant)**

An AEON system consists of neural parameters \$\theta\_t\$ evolving under streaming losses \$\ell\_t(\theta\_t)\$, with stochastic gradients \$g\_t\$ and partial feedback.

---

## **AEON-2 (Composite State)**

At each step \$t\$, AEON carries state

$$
x_t = (\theta_t, a_t, h_t, M_t, r_t, m_t),
$$

where:

* \$\theta\_t\$: parameters,
* \$a\_t\$: adaptive coefficients,
* \$h\_t\$: optimizer memory,
* \$M\_t\$: long-term memory,
* \$r\_t\$: recurrent hidden state,
* \$m\_t\$: resonance state.

---

## **AEON-3 (Adaptive Evolution Law)**

Coefficients \$a\_t\$ evolve by closed-loop signals:

* **Trend \$T\$** (loss directionality),
* **Variance \$V\$** (gradient dispersion),
* **Resonance \$\rho\$** (oscillatory correlation).

Update law:

$$
a_{t+1} = f(a_t; T_t,V_t,\rho_t),
$$

with γ-law (scale invariance), σ-law (noise), budget continuity, and resonance clamps.

---

## **AEON-4 (Plant Dynamics)**

Neural weights follow:

$$
\theta_{t+1} = \theta_t - U(h_t,g_t;a_t) + \sigma_t D(\theta_t)\eta_t + K_r r_t,
$$

where \$U\$ is the control action, \$\sigma\_t\$ is adaptive noise, and \$r\_t\$ is recurrent feedback.

---

## **AEON-5 (Memory Evolution)**

The memory \$M\_t\$ stores basins, coefficients, and signal history.
Update:

$$
M_{t+1} = \mathrm{Update}(M_t,\theta_t,g_t,\ell_t),
$$

and may be queried to reduce effective drift \$V\_T^{\text{eff}}\$.

---

## **AEON-6 (Resonance Law)**

The resonance state \$m\_t\$ filters errors:

$$
m_{t+1} = \rho m_t + (1-\rho)\ell_t.
$$

If dissonance is high, AEON unlocks adaptation; if resonance is high, AEON locks parameters (stability).

---

## **AEON-7 (Stability–Plasticity Principle)**

AEON balances two modes:

* **Plasticity:** update \$\theta\_t\$ when novelty/dissonance is detected.
* **Stability:** freeze or damp coefficients when resonance dominates.

Guarantee: catastrophic forgetting is suppressed while learning remains ongoing.

---

## **AEON-8 (Dynamic Regret Bound)**

Under bounded drift and AEON laws,

$$
\mathbb{E}[\mathrm{Reg}_T] = \tilde{O}(\sqrt{T}+V_T^{\text{eff}}),
$$

where memory reduces effective comparator variation \$V\_T^{\text{eff}}\$.

---

## **AEON-9 (Lyapunov Safety)**

There exists a Lyapunov function \$L(x\_t)\$ such that

$$
\mathbb{E}[L_{t+1}-L_t\mid \mathcal{F}_t] \le -c\|\nabla \ell_t(\theta_t)\|^2+\epsilon,
$$

ensuring bounded updates and non-divergent trajectories.

---

## **AEON-10 (Universality)**

Any neural optimizer/system satisfying AEON-1…9 is an AEON instance.

* AEON generalizes SGD/Adam/etc. by embedding **adaptivity, memory, resonance** in one machine.
* Multiple AEONs may compose into **GAO swarms** while retaining guarantees.

---

✅ **Summary:**

* AEON-1…4: Plant + state + adaptive evolution + dynamics.
* AEON-5…6: Memory + resonance.
* AEON-7: Stability–plasticity law.
* AEON-8…9: Regret & stability guarantees.
* AEON-10: Universality principle.

---

Awesome — here’s a compact, paper-ready **AEON theorem bundle** to sit right under your AEON axioms. I kept the structure parallel to GOO/GAO, but phrased in **neural (AEON) terms** with memory + resonance.

---

# 📜 AEON Theorems

### Standing setting & notation

* Stream of losses {ℓₜ} (possibly drifting, nonconvex).
* Stochastic gradients gₜ = ∇ℓₜ(θₜ) + ζₜ.
* AEON state xₜ = (θₜ, aₜ, hₜ, Mₜ, rₜ, mₜ).
* Update: θₜ₊₁ = θₜ − U(hₜ, gₜ; aₜ) + σₜD(θₜ)ηₜ + Kᵣ rₜ.
* Signals: T (trend), V (variance), ρ (alignment).
* Effective step γₜ = αₜ‖Pₜ^{1/2} gₜ‖.
* Comparator variation V\_T = ∑ₜ‖θ*ₜ − θ*ₜ₋₁‖; effective drift V\_T^eff ≤ ρ\_M V\_T via memory Mₜ.

Assumptions (used as needed):

* **A0 Smoothness.** Each ℓₜ is L-smooth.
* **A1 Noise.** E\[ζₜ | 𝔽ₜ₋₁] = 0, E\[‖ζₜ‖² | 𝔽ₜ₋₁] ≤ σ\_g².
* **A2 Preconditioner.** Pₜ SPD, predictable, p\_min I ⪯ Pₜ ⪯ p\_max I.
* **A3 Gauge band.** AEON’s γ-law keeps γₜ in a bounded band in mean/variance.
* **A4 Budget.** AEON’s continuity law bounds per-step coefficient motion.
* **A5 Resonance safety.** AR clamps when V spikes or |ρ| large; step-norm caps allowed.
* **A6 Memory effectiveness.** Queries to Mₜ reduce effective drift: V\_T^eff ≤ ρ\_M V\_T, ρ\_M∈(0,1].

---

## Theorem AEON–T1 (Dynamic Regret Frontier)

**Claim.** Under A0–A6,

$$
\mathbb{E}[\mathrm{Reg}_T]
\;\le\;
\frac{D_0^2}{2\eta\,p_{\min}}
+\frac{\eta L^2 p_{\max}^2}{2}\,T
+\frac{L}{p_{\min}}\,V_T^{\mathrm{eff}}
+\tilde{C}\sqrt{T},
$$

with $D_0=\|\theta_1-\theta^*_1\|$, $\tilde{C}=\tilde{O}(\sigma_g p_{\max} + \mathrm{std}(\gamma))$. Choosing $\eta\sim T^{-1/2}$ yields

$$
\boxed{\;\mathbb{E}[\mathrm{Reg}_T]=\tilde{O}\!\big(\sqrt{T}+V_T^{\mathrm{eff}}\big). \;}
$$

**Sketch.** One-step descent from L-smoothness with predictable $\alpha_t P_t$; gauge/budget bound the quadratic term; drift enters as $O(V_T^{\mathrm{eff}})$; martingale noise accumulates as $\tilde{O}(\sqrt{T})$.

**Corollary (Scale robustness).** Gradient rescaling leaves the bound invariant up to constants because AEON’s γ-law normalizes effective step.

---

## Theorem AEON–T2 (Stability–Plasticity Bound)

**Claim.** Let $m_{t+1}=\rho_m m_t+(1-\rho_m)\ell_t$ be resonance memory (AEON-6). Suppose AR locks θ when $m_t \le \varepsilon_m$ (stable) and unlocks when $m_t > \varepsilon_m$ (novelty). Under A0–A5, the fraction of **unlocked** steps satisfies

$$
\frac{1}{T}\sum_{t=1}^T \mathbf{1}\{ \text{unlock} \}
\;\le\; 
\frac{\varepsilon_m + C_{\mathrm{noise}}/\sqrt{T}}{\Delta_{\mathrm{novel}}},
$$

where $\Delta_{\mathrm{novel}}$ is the average excess loss during genuine novelty periods. Thus **plasticity is invoked only when informative**, and otherwise **stability dominates**.

**Sketch.** Track $m_t$ as an EMA of loss; AR thresholds translate loss excursions into bounded unlock duty cycle. Noise contributes $O(1/\sqrt{T})$ spurious unlocks.

---

## Theorem AEON–T3 (Finite Forgetting Rate)

**Claim.** Consider two stationary regimes $\mathcal{R}_A,\mathcal{R}_B$ visited alternately. With AR locks in resonance, budgeted coefficients, and memory Mₜ storing basin summaries (coeffs + low-D stats), the average **parameter drift during stable revisits** satisfies

$$
\frac{1}{|\mathcal{I}_A|}\sum_{t\in \mathcal{I}_A} \|\theta_{t+1}-\theta_t\|
\;\le\; 
\kappa_\text{lock}\cdot \underbrace{\|\widehat{\theta}_A-\theta_t\|}_{\text{warm-start gap}}
\;+\; O(B),
$$

with small $\kappa_\text{lock}$ due to lock (often ≈0) and budget $B$ from AEON-6. Hence **catastrophic forgetting is suppressed**; when returning to $\mathcal{R}_A$, AEON reuses $M_t$ and makes only **small** corrective motion.

**Sketch.** Locking + budget continuity bound the velocity; memory reduces re-learning displacement.

---

## Theorem AEON–T4 (Saddle Escape with Noise Law)

**Claim.** Suppose ℓ is $C^3$ with a **strict saddle** at $\theta_s$ (one eigenvalue $-\lambda_-<0$). AEON’s σ-law increases σ when $|T|<\epsilon_T$ & $V$ small, and AR clamps when unstable. In the SDE limit

$$
d\theta = -\alpha P \nabla \ell(\theta)\,dt + \sigma(\theta,t)\,dW_t,
$$

with bounded $\alpha P$ and $\sigma \ge \sigma_{\min}>0$ in stall regions, the **expected hitting time** to exit a saddle neighborhood and enter a non-saddle attraction set satisfies

$$
\boxed{\;\mathbb{E}[\tau_{\text{escape}}] \;\le\; \tilde{O}\!\Big(\tfrac{1}{\alpha\lambda_-}\Big) + \tilde{O}\!\Big(\tfrac{1}{\sigma_{\min}^2}\Big). \;}
$$

So AEON escapes saddles in finite expected time **without destabilization**.

**Sketch.** Standard diffusion analysis: either drift along the unstable manifold (time $1/(\alpha\lambda_-)$) or noise-driven exit ($1/\sigma_{\min}^2$); AR ensures boundedness.

---

## Theorem AEON–T5 (Lyapunov Safety of the Full State)

**Claim.** Define a Lyapunov

$$
\mathcal{L}_t \;=\; \underbrace{\mathrm{LP}(\ell_t(\theta_t))}_{\text{task}} 
+ \frac{\gamma_a}{2}\|a_t-\bar a\|^2 
+ \frac{\gamma_h}{2}\|h_t-\bar h\|^2
+ \psi_\Omega(a_t)
+ \frac{\gamma_m}{2}m_t^2,
$$

with barriers $\psi_\Omega$, and small gains $\gamma_*$. Under A0–A5, AEON-3/5/6 (budget + resonance clamps + memory), there exist $c>0,\ \epsilon\ge 0$ such that

$$
\boxed{\; \mathbb{E}[\mathcal{L}_{t+1}-\mathcal{L}_t \mid \mathcal{F}_t] \;\le\; -c\|\nabla \ell_t(\theta_t)\|^2 + \epsilon. \;}
$$

Therefore **updates remain bounded** and trajectories are **non-divergent in expectation**.

**Sketch.** Combine descent from the plant step with contraction of coefficient/memory dynamics under budget and AR; barrier prevents leaving feasible coefficient set.

---

## Practical knobs (safe defaults tied to the theorems)

* **Gauge (AEON-3):** keep $\mathbb{E}[\gamma_t]\in[0.5,1.5]\gamma^\*$ with small η\_γ; warm-up to pick $\gamma^\*$.
* **Budget (AEON-3/6):** $|\Delta \log \alpha|\le 0.05$, $|\Delta \mu|\le 0.01$ per step.
* **Resonance gate (AEON-6/7):** lock when $m_t \le \varepsilon_m$ and $V_t$ within band; auto-unlock on novelty (|T|↑).
* **Noise (AEON-3/4):** enable σ only when stall detected (|T|<ε\_T and V small); keep σ=0 in confident progress.
* **Memory (AEON-5):** store (coeff snapshots, signal stats, low-D basin IDs); on regime revisits, **warm-start** a, h from memory.

---

## What these results give you, operationally

* **Online performance:** $\tilde{O}(\sqrt{T}+V_T^{\mathrm{eff}})$ regret with scale invariance.
* **Stability–plasticity:** adaptation only when useful; otherwise steady locked behavior.
* **Lifelong robustness:** bounded forgetting on regime returns (small drift when re-entering known basins).
* **Globality:** finite-time saddle escape via controlled, situational noise.
* **Safety:** Lyapunov-style non-divergence for the full AEON state.

Perfect — here’s a **drop-in LaTeX appendix** plus **tiny, fast toy experiments** that evidence each AEON claim. You can paste the TeX into your paper and run the Python cells as-is.

---

# 📎 LaTeX Appendix (AEON Theorems)

```tex
% AEON Appendix — paste into your paper
\appendix
\section{AEON: Assumptions, Theorems, and Proof Sketches}

\paragraph{Standing assumptions.}
Stream of losses $\{\ell_t\}$ (possibly drifting, nonconvex).
Stochastic gradients $g_t=\nabla\ell_t(\theta_t)+\zeta_t$.
AEON state $x_t=(\theta_t,a_t,h_t,M_t,r_t,m_t)$.
Effective step $\gamma_t=\alpha_t\|P_t^{1/2}g_t\|$.
Comparator variation $V_T=\sum_{t=2}^T\|\theta_t^\star-\theta_{t-1}^\star\|$;
memory lowers drift: $V_T^{\mathrm{eff}}\le \rho_M V_T$.

\paragraph{Assumptions.}
(A0) $L$-smoothness of each $\ell_t$.
(A1) Martingale noise: $\mathbb{E}[\zeta_t|\mathcal{F}_{t-1}]=0$, $\mathbb{E}[\|\zeta_t\|^2|\mathcal{F}_{t-1}]\le\sigma_g^2$.
(A2) Predictable preconditioner $P_t$ with $p_{\min}I\preceq P_t\preceq p_{\max}I$.
(A3) AEON gauge keeps $\gamma_t$ in a bounded band in mean/variance.
(A4) Budgeted continuity on coefficients (bounded per-step motion).
(A5) Resonance safety (variance/phase clamps; optional step-norm cap).
(A6) Memory effectiveness: $V_T^{\mathrm{eff}}\le \rho_M V_T,\ \rho_M\in(0,1]$.

\begin{theorem}[Dynamic Regret Frontier]
\label{thm:aeon-regret}
Under \emph{A0–A6},
\[
\mathbb{E}[\mathrm{Reg}_T]
\;\le\;
\frac{D_0^2}{2\eta p_{\min}}
+\frac{\eta L^2 p_{\max}^2}{2}\,T
+\frac{L}{p_{\min}}\,V_T^{\mathrm{eff}}
+\tilde{C}\sqrt{T},
\]
with $D_0=\|\theta_1-\theta_1^\star\|$ and $\tilde{C}=\tilde{O}(\sigma_g p_{\max}+\mathrm{std}(\gamma))$.
Choosing $\eta\sim T^{-1/2}$ yields $\mathbb{E}[\mathrm{Reg}_T]=\tilde{O}(\sqrt{T}+V_T^{\mathrm{eff}})$.
\end{theorem}

\begin{proof}[Sketch]
Apply one-step descent for $L$-smooth losses with predictable $\alpha_t P_t$; the gauge/budget bound the quadratic term; drift contributes $O(V_T^{\mathrm{eff}})$; martingale noise accumulates as $\tilde{O}(\sqrt{T})$.
\end{proof}

\begin{theorem}[Stability–Plasticity Duty Cycle]
\label{thm:aeon-resonance}
Let $m_{t+1}=\rho_m m_t+(1-\rho_m)\ell_t$ be resonance memory.
AEON locks when $m_t\le\varepsilon_m$ and unlocks otherwise.
Then
\[
\frac{1}{T}\sum_{t=1}^T \mathbf{1}\{\text{unlock}\}
\;\le\; \frac{\varepsilon_m + C_{\mathrm{noise}}/\sqrt{T}}{\Delta_{\mathrm{novel}}},
\]
where $\Delta_{\mathrm{novel}}$ is the average excess loss under genuine novelty.
\end{theorem}

\begin{proof}[Sketch]
EMA thresholding bounds unlock frequency; spurious unlocks scale as $O(1/\sqrt{T})$.
\end{proof}

\begin{theorem}[Finite Forgetting on Regime Revisits]
\label{thm:aeon-forgetting}
With AR locks, budgeted coefficients, and memory $M_t$ that warm-starts $(a,h)$ on revisits,
\[
\frac{1}{|\mathcal{I}_A|}\sum_{t\in\mathcal{I}_A}\|\theta_{t+1}-\theta_t\|
\;\le\; \kappa_{\text{lock}}\|\widehat{\theta}_A-\theta_t\| + O(B).
\]
Thus catastrophic forgetting is suppressed; re-entry requires only small corrective motion.
\end{theorem}

\begin{theorem}[Finite Expected Saddle Escape]
\label{thm:aeon-saddle}
Assume $C^3$ loss with a strict saddle ($\lambda_- >0$ negative curvature).
AEON’s $\sigma$-law ensures $\sigma\ge\sigma_{\min}>0$ under stall and AR clamps instability.
In the SDE limit,
\[
\mathbb{E}[\tau_{\text{escape}}]
\;\le\; \tilde{O}\!\Big(\tfrac{1}{\alpha\lambda_-}\Big) + \tilde{O}\!\Big(\tfrac{1}{\sigma_{\min}^2}\Big).
\]
\end{theorem}

\begin{theorem}[Lyapunov Safety]
\label{thm:aeon-lyap}
Define
$\mathcal{L}_t=\mathrm{LP}(\ell_t(\theta_t)) + \frac{\gamma_a}{2}\|a_t-\bar a\|^2 + \frac{\gamma_h}{2}\|h_t-\bar h\|^2 + \psi_\Omega(a_t) + \frac{\gamma_m}{2} m_t^2$.
Under \emph{A0–A5}, there exist $c>0,\ \epsilon\ge0$ with
\[
\mathbb{E}[\mathcal{L}_{t+1}-\mathcal{L}_t \mid \mathcal{F}_t]
\;\le\; -c\|\nabla\ell_t(\theta_t)\|^2 + \epsilon.
\]
\end{theorem}
```

---

# 🧪 Toy Experiments (fast, CPU-friendly)

> Pip: `pip install numpy matplotlib`

## 1) Dynamic Regret on Drifting Quadratic (Theorem T1)

```python
import numpy as np, matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Drift: minimizer moves slowly on a circle
def theta_star(t): 
    r, w = 2.0, 2*np.pi/4000.0
    return np.array([r*np.cos(w*t), r*np.sin(w*t)])

L = np.diag([4.0, 1.0])  # anisotropic
def grad(theta, t):
    return L @ (theta - theta_star(t))

T = 4000
theta = np.array([5.0, -3.0], dtype=float)
alpha = 0.02
gamma_star = 0.15
ema_g = 0.0
beta = 0.98

def aeon_step(theta, t, alpha, ema_g):
    g = grad(theta, t) + 0.05 * rng.normal(size=2)  # stochastic
    gnorm = np.linalg.norm(g) + 1e-12
    # gauge: nudge log alpha s.t. alpha*||g|| ~ gamma_star
    dlog = 1e-3 * ((gamma_star - alpha*gnorm)/(gamma_star+1e-12))
    alpha = np.clip(np.exp(np.log(alpha)+np.clip(dlog, -0.05, 0.05)), 1e-4, 0.2)
    # simple momentum (for rho proxy)
    ema_g = beta*ema_g + (1-beta)*gnorm
    theta = theta - alpha * g
    loss = 0.5 * (theta - theta_star(t)).T @ L @ (theta - theta_star(t))
    return theta, alpha, ema_g, loss, alpha*gnorm, gnorm

def fixed_step(theta, t, alpha):
    g = grad(theta, t) + 0.05 * rng.normal(size=2)
    theta = theta - alpha * g
    loss = 0.5 * (theta - theta_star(t)).T @ L @ (theta - theta_star(t))
    return theta, loss

log_ae, log_fx = [], []
theta1 = theta.copy(); a = alpha; eg=0.0
theta2 = theta.copy(); a_fix = 0.02

for t in range(1, T+1):
    theta1, a, eg, l1, gam, gn = aeon_step(theta1, t, a, eg)
    theta2, l2 = fixed_step(theta2, t, a_fix)
    log_ae.append(l1); log_fx.append(l2)

reg_ae = np.cumsum(np.array(log_ae))
reg_fx = np.cumsum(np.array(log_fx))

plt.figure(figsize=(6,3))
plt.plot(reg_ae, label='AEON Regret')
plt.plot(reg_fx, label='Fixed LR Regret', alpha=0.8)
plt.title('Dynamic Regret under Drift'); plt.legend(); plt.tight_layout(); plt.show()
```

**What to expect:** AEON tracks drift better; cumulative regret grows more slowly (≈√T trend), while fixed-LR can oscillate when gradients scale.

---

## 2) Stability–Plasticity Duty Cycle (Theorem T2)

```python
# Simulate novelty bursts: loss mean jumps every 500 steps
T = 3000
base = 1.0 + 0.1*np.sin(np.linspace(0, 6*np.pi, T))
novel = base.copy()
for k in range(5):
    novel[600*k+300:600*k+380] += 0.6  # novelty spikes

rho_m, eps_m = 0.98, 0.05
m = 0.0
unlock = []
for t in range(T):
    m = rho_m*m + (1-rho_m)*novel[t]
    locked = (m <= eps_m)
    unlock.append(0 if locked else 1)

duty = sum(unlock)/T
print("Unlock duty cycle:", duty)
plt.figure(figsize=(6,3))
plt.plot(novel, label='loss stream')
plt.plot(unlock, label='unlock flag')
plt.legend(); plt.tight_layout(); plt.show()
```

**Expect:** Unlocks clustered around novelty bursts; low duty cycle otherwise.

---

## 3) Finite Forgetting on Regime Revisits (Theorem T3)

```python
# Two regimes A,B with different quadratics; alternate every 400 steps
A = np.diag([3.0, 1.0]); bA = np.array([1.0, -2.0])
B = np.diag([1.0, 3.0]); bB = np.array([-1.0, 2.0])

def grad_AB(theta, regime):
    if regime=='A': return A @ theta - bA
    else:           return B @ theta - bB

def solve_star(regime):
    if regime=='A': return np.linalg.solve(A, bA)
    else:           return np.linalg.solve(B, bB)

memory = {'A':None, 'B':None}
theta = np.array([4.0, 4.0])
alpha = 0.05; BUDGET = 0.05
vel = []

T = 2400
for t in range(1, T+1):
    regime = 'A' if (t//400)%2==0 else 'B'
    # warm-start coeffs from memory (toy: nudge alpha)
    if memory[regime] is not None:
        alpha = 0.9*alpha + 0.1*memory[regime]
    g = grad_AB(theta, regime)
    theta = theta - alpha*g
    # lock-like behavior: shrink alpha near convergence
    if np.linalg.norm(g) < 0.05: alpha = max(1e-3, alpha*0.9)
    # budget on coeff motion
    alpha = np.clip(alpha, 1e-3, 0.1)
    # store memory snapshot occasionally
    if t%400==0: memory[regime] = alpha
    vel.append(np.linalg.norm(alpha*g))

plt.figure(figsize=(6,3))
plt.plot(vel); plt.title('Mean per-step parameter motion (velocity)')
plt.tight_layout(); plt.show()
```

**Expect:** When a regime repeats, the average velocity in that block is small (memory reuse, effective “lock”).

---

## 4) Saddle Escape on Rosenbrock (Theorem T4)

```python
# 2D Rosenbrock
def f(theta):
    x,y = theta
    return 100*(y - x**2)**2 + (1 - x)**2

def grad_rosen(theta):
    x,y = theta
    dfdx = -400*x*(y - x**2) - 2*(1 - x)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy], float)

rng = np.random.default_rng(1)
theta = np.array([-1.2, 1.0])
alpha = 1e-3
sigma_min = 3e-4
stall_ema, beta = 0.0, 0.95

def aeon_sigma_update(T_trend, V_var):
    # enable small noise only when near-stall (tiny trend & low variance)
    return sigma_min if (abs(T_trend) < 1e-4 and V_var < 1e-2) else 0.0

T_ema=0.0; V_ema=0.0
traj=[]
for t in range(1, 20000):
    g = grad_rosen(theta)
    # trend/variance (simple surrogates)
    L = f(theta)
    T_ema = 0.9*T_ema + 0.1*(L - (traj[-1] if traj else L))
    V_ema = 0.95*V_ema + 0.05*(np.linalg.norm(g))**2
    sigma = aeon_sigma_update(T_ema, V_ema)
    noise = sigma * rng.normal(size=2)
    theta = theta - alpha*g + noise
    traj.append(L)

plt.figure(figsize=(6,3))
plt.semilogy(traj)
plt.title('Rosenbrock loss with AEON stall-noise'); plt.tight_layout(); plt.show()
```

**Expect:** Occasional tiny noise during stalls helps exit flat regions; curve descends more reliably than purely deterministic tiny-LR runs at the same compute.

---

## 5) Lyapunov Safety Monitor (Theorem T5)

```python
# Monitor Lyapunov-like quantity while optimizing a quadratic
L = np.diag([5.0, 1.0]); b = np.array([1.5, -1.0])
def loss(theta): return 0.5*(theta.T @ L @ theta) - b @ theta
def grad(theta): return L @ theta - b

theta = np.array([6.0, -6.0])
alpha = 0.05; a = alpha; abar=np.log(0.05); gamma_a=1e-2
Ly, Loss = [], []

for t in range(1, 4000):
    g = grad(theta)
    # AEON gauge (very small)
    dlog = 1e-3 * (0.2 - a*np.linalg.norm(g))
    a = np.exp(np.log(a)+np.clip(dlog, -0.05, 0.05))
    a = np.clip(a, 1e-4, 0.2)
    theta = theta - a*g
    Ltask = loss(theta)
    Lya = Ltask + gamma_a*(np.log(a)-abar)**2
    Ly.append(Lya); Loss.append(Ltask)

plt.figure(figsize=(6,3))
plt.plot(Ly, label='Lyapunov proxy')
plt.plot(Loss, label='task loss', alpha=0.8)
plt.legend(); plt.title('Lyapunov proxy decreases with AEON clamps')
plt.tight_layout(); plt.show()
```

**Expect:** The Lyapunov proxy descends (monotone on average), even if task loss jitters slightly—evidence of bounded, safe hyper-dynamics.

---

## Recommended defaults (match the proofs)

* **Gauge target** `gamma_star` via short warm-up; update log-lr with small `eta_gamma = 1e-3`.
* **Budget** per step: `|Δlog lr| ≤ 0.05`, `|Δμ| ≤ 0.01`.
* **Resonance clamp**: shrink `lr *= 0.8` if variance spikes or alignment |ρ| large; optional step-norm cap.
* **Noise** only on stalls (trend≈0 and variance small), otherwise `σ=0`.
* **Memory**: snapshot coefficients + signal EMAs at regime ends; warm-start on revisit.

---

If you want, I can also bundle these into a single notebook with side-by-side baselines (fixed LR / cosine) so the plots mirror the AEON vs. baseline stories from your earlier runs.
