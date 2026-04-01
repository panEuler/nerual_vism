# Neural-VISM: Critical Methodology Evaluation

> [!NOTE]
> This evaluation is based on the design document ([claude.md](file:///Users/ruyu/code/nerual-vism/claude.md)) **and** the actual implementation in `src/biomol_surface_unsup/`. Where the design and code diverge, I note both.

---

## 1. Problem Formulation

### 1.1 Is the SDF learning objective well-defined without ground-truth supervision?

**Verdict: Partially well-defined, with a critical identifiability gap.**

The core idea — parameterize the solvent-excluded surface (SES) as the zero level set of a neural SDF $\phi_\theta$, then optimize $\theta$ by minimizing a VISM free energy functional — is **conceptually sound** and directly grounded in classical variational solvation theory (Dzubiella, Swanson, McCammon 2006; Zhou et al. 2014). The key insight is correct: the physically meaningful surface is the *minimizer* of the solvation free energy, so you don't need ground-truth labels — the energy functional *is* your supervision.

However, there are important caveats:

**⚠️ The energy functional is incomplete.** The design document's $G_\theta$ (Section 3.1) includes:
- Pressure–volume term ($P \int (1 - H_\epsilon) dx$)
- Surface area ($\gamma_0 \int \delta_\epsilon |\nabla\phi| dx$)
- Mean curvature correction ($-2\gamma_0 \tau \int H \delta_\epsilon |\nabla\phi| dx$)
- LJ body integral ($\rho_0 \int U(x) H_\epsilon dx$)

But the actual implementation (the 5 losses in [loss_builder.py](file:///Users/ruyu/code/nerual-vism/src/biomol_surface_unsup/losses/loss_builder.py)) does **not** implement the full VISM functional. Specifically:

| VISM term | In design doc? | In implementation? |
|-----------|:-:|:-:|
| Pressure × Volume | ✅ | ⚠️ `volume_loss` matches a *target volume fraction*, not pressure × volume |
| Surface area | ✅ | ✅ `area_loss` via $\delta_\epsilon |\nabla\phi|$ |
| Mean curvature correction | ✅ | ❌ **Missing entirely** |
| LJ body integral | ✅ | ❌ **Missing entirely** |
| Electrostatic (PB/GB) | ❌ | ❌ |
| Weak prior (atomic union) | — | ✅ (regularizer, not a VISM term) |
| Containment | — | ✅ (regularizer) |
| Eikonal | — | ✅ (regularizer) |

**This means the actual optimization target is not the VISM free energy** — it's a heavily regularized geometric proxy. Without the LJ term, the surface has no physical reason to retract into hydrophobic clefts. Without curvature, there's no energetic penalty against high-curvature pinch-offs. The system is functionally being driven by area minimization + volume matching + a weak alignment to the van der Waals union — which will produce a *smoothed van der Waals surface*, not a VISM-equilibrium surface.

### 1.2 Are the assumptions physically / geometrically sound?

**Mostly sound, with one structural issue:**

- ✅ **SDF representation for the level-set surface** — standard and well-supported by both level-set methods and neural implicit literature.
- ✅ **Smooth Heaviside / Dirac**  — necessary for differentiability; the $\arctan$-based $H_\epsilon$ and Cauchy $\delta_\epsilon$ are standard choices.
- ⚠️ **Atomic union as weak prior** — the `_batched_atomic_union_field` uses a log-sum-exp smooth-minimum ($\alpha=10$). This is a reasonable differentiable approximation of the union-of-spheres SDF, but $\alpha=10$ is moderately aggressive — at atomic spacings of ~1.5 Å, the smooth-min error can be ~0.1 Å, which is physically non-trivial for SES modeling. Consider making $\alpha$ configurable or using a provably tighter bound (see Section 7).
- ⚠️ **No probe-radius correction** — the design assumes SAS/SES but the implementation's atomic union uses raw vdW radii. The SES is defined as the boundary of the region accessible to a solvent probe (typically 1.4 Å water radius). The union of $(r_i + r_\text{probe})$-spheres gives the SAS, and the SES requires additional reentrant surface construction. Neither appears in the code.

---

## 2. Model Architecture

### 2.1 Permutation invariance and local geometric structure

**Verdict: Correctly invariant, but geometrically under-expressive.**

The architecture is:
```
LocalFeatureBuilder(kNN + RBF) → DeepSets(φ → sum/mean → ρ) → concat(z_local, z_global) → MLP → scalar SDF
```

**Permutation invariance**: ✅ Achieved via sum-pooling over neighbors in `LocalDeepSetsEncoder` and mean-pooling in `GlobalFeatureEncoder`. This is correct.

**Local geometric capture**: ⚠️ Weak. The current feature vector per neighbor is:

```
[rel_pos (3), radius (1), atom_embed (16), rbf(dist) (16), dist (1)] → 37 dims
```

This is passed through a per-neighbor MLP $\phi$ then sum-pooled. The problem: **DeepSets with this architecture is fundamentally limited in capturing angular/geometric structure**. It processes each neighbor independently before pooling, so it cannot represent:

- **Pairwise interactions between neighbors** (e.g., "two neighbors are close together" → pocket/cleft detection)
- **Angular features** (e.g., bond angles, dihedral-like features) that are critical for distinguishing convex vs. concave regions on the molecular surface

This is well-documented in the point cloud learning literature: DeepSets ≈ PointNet, which fails on fine-grained geometric tasks that require neighbor-neighbor interactions.

### 2.2 Better alternatives

**High-priority alternatives (ranked by impact-to-complexity ratio):**

| Architecture | Captures local geometry? | Permutation invariant? | SE(3) equivariant? | Complexity |
|---|:-:|:-:|:-:|:-:|
| Current (DeepSets) | ❌ no neighbor pairs | ✅ | ❌ | Low |
| **PointNet++ (set abstraction)** | ⚠️ hierarchical | ✅ | ❌ | Low |
| **SchNet / DimeNet** | ✅ distance + angles | ✅ | ✅ (invariant) | Medium |
| **EGNN** (Satorras et al. 2021) | ✅ pairwise distances | ✅ | ✅ (equivariant) | Medium |
| **TFN / MACE / NequIP** | ✅ full angular | ✅ | ✅ (equivariant) | High |

**My recommendation:**

1. **Short-term (highest ROI):** Replace the `LocalDeepSetsEncoder` with a **SchNet-style continuous-filter convolution** or **PaiNN** (Schütt et al. 2021). These are:
   - Still simple message-passing networks
   - Naturally handle variable neighbor counts
   - Capture distance-dependent interactions with learnable radial filters
   - Well-tested on molecular property prediction

2. **Medium-term:** The current architecture doesn't leverage the *query point coordinates* in the decoder at all — the `SDFDecoder` just takes the concatenated feature vector. This means the spatial position of the query point only enters through relative distances to neighbors. Adding **positional encoding / Fourier features** of the query point (as the design doc suggests but the implementation doesn't do) would significantly improve high-frequency detail.

3. **Long-term:** For truly detailed surfaces, consider **implicit neural representations** (SIREN / Fourier-feature networks) as the decoder backbone, conditioned on molecular features via FiLM or hypernetwork modulation — which is exactly what the design doc proposes. The `FiLMDecoder` exists but **is not connected** to the main model.

### 2.3 Specific architectural issues in the code

1. **SDFDecoder is only 3 layers** ([sdf_decoder.py:6-12](file:///Users/ruyu/code/nerual-vism/src/biomol_surface_unsup/models/decoders/sdf_decoder.py#L6-L12)) with `SiLU` activations. For SDF prediction, `SiLU`/`GELU` are okay, but **SIREN (sine activations with careful initialization)** dramatically outperforms ReLU-family networks for representing signed distance fields with sharp features (Sitzmann et al. 2020). This is the most impactful single-line change for surface quality.

2. **Global encoder uses absolute coordinates** ([global_features.py:30](file:///Users/ruyu/code/nerual-vism/src/biomol_surface_unsup/features/global_features.py#L30)): `torch.cat([coords, atom_emb, radii], dim=-1)`. This **breaks translation invariance**. The global embedding changes if the molecule is shifted. Use center-of-mass-subtracted coordinates or only rely on pairwise distances.

3. **No skip connections** in either encoder or decoder. For SDF networks, skip connections (as in DeepSDF, OccNet) are important for gradient flow with the eikonal constraint.

---

## 3. Sampling Strategy

### 3.1 Is the sampling sufficient and unbiased?

**Verdict: Reasonable first pass, but statistically biased and physically incomplete.**

The three-group sampling in [sampling.py](file:///Users/ruyu/code/nerual-vism/src/biomol_surface_unsup/datasets/sampling.py):

| Group | Count fraction | Purpose | Issue |
|---|---|---|---|
| Global (bbox uniform) | 50% | Volume/eikonal | ✅ Unbiased for volume integrals |
| Containment (atom-centered jitter) | 25% | Inside-surface anchor | ⚠️ Only samples near atom centers, not deep interior |
| Surface band (rejection from bbox) | 25% | Area/weak prior | ⚠️ Biased toward convex regions |

**Key issues:**

1. **Surface band sampling is biased toward convex, exposed regions.** The procedure samples uniformly in the bbox, then filters by $|\text{SDF}_\text{union}(x)| \leq 0.25$. For molecules with deep pockets/clefts, the *surface area per unit volume* is higher internally, but the *volume of the narrow band* in clefts is smaller. This means concave regions are under-sampled — precisely where the surface shape matters most for solvation.

2. **No importance sampling.** The design document mentions sampling in "LJ potential high-gradient regions" (Section 4.2, point 4), but this is **not implemented**. For a physically meaningful surface, the LJ–gradient region is where the surface will actually sit — missing this entirely means the loss has poor signal where it matters.

3. **Containment jitter is too small.** `containment_jitter = 0.15 * radii` means containment points are within ~0.15 Å of atom centers (for typical $r \approx 1.5$ Å → jitter ≈ 0.22 Å). This is a very thin shell. Any query outside this shell but still "inside" the molecular interior gets no containment signal.

4. **50% global samples is wasteful.** Most bbox-uniform points are far from the surface and contribute near-zero gradient to area loss and weak prior. The eikonal loss benefits from global samples, but 50% is excessive. Consider 20–30% global, 40–50% surface band.

### 3.2 Better practices

- **Stratified surface-band sampling** (Gropp et al. 2020, IGR): Sample points by perturbing surface points along normals with varying magnitudes. This uniformly covers the surface rather than relying on volume-based rejection.
- **Progressive narrowing**: Start with wide band, narrow during training as the SDF converges.
- **Importance sampling for integral estimation**: For the area integral $\int \delta_\epsilon(\phi) |\nabla\phi| dx$, the integrand is sharply peaked near $\phi = 0$. Uniform sampling is highly inefficient; use the current approximation of the zero-level-set to guide sampling adaptively.

---

## 4. Loss Design

### 4.1 Eikonal Loss

```python
penalty = (grads.norm(dim=-1) - 1.0).pow(2)   # ← L2 penalty
```

**Assessment: Correct formulation, but consider L1.**

The $L_2$ eikonal $(|\nabla\phi| - 1)^2$ is standard (Gropp et al. 2020). However, empirically $L_1$ `abs()` (as stated in the design doc: $|\,|\nabla\phi| - 1\,|$) often works better for SDF networks — $L_2$ over-penalizes large deviations near surface singularities (ridges, corners), causing over-smoothing. **The code uses L2 but the design doc says L1 — choose L1.**

The eikonal weight (0.1) is also quite low. In most SDF-learning papers (DeepSDF, IGR, NeuS), eikonal is the *primary* regularizer with weight 0.5–1.0. With weight 0.1, $|\nabla\phi|$ may deviate significantly from 1, meaning $\phi$ is not a *true* distance function — undermining the physical meaning of all other losses that depend on $\phi$ being an SDF.

### 4.2 Area Loss

```python
integrand = smooth_delta(pred_sdf, eps) * grads.norm(dim=-1)
```

**Assessment: Correct coarea formulation, but numerically fragile and potentially degenerate.**

This implements $\int \delta_\epsilon(\phi) |\nabla\phi| \,dx$, which is the smoothed coarea formula for surface area. This is standard level-set methodology.

**Problems:**

1. **Minimizing area alone → collapse to empty surface.** If area is minimized without a lower bound on volume or a containment constraint, the global minimum is $\phi \equiv +\infty$ (everything "outside"). The containment loss provides a counterforce, but the balance is delicate — see Failure Modes.

2. **$\epsilon = 0.1$ is a sensitive hyperparameter.** The Cauchy delta $\delta_\epsilon(\phi) = \frac{\epsilon}{\pi(\epsilon^2 + \phi^2)}$ has effective support $\sim [-3\epsilon, 3\epsilon] = [-0.3, 0.3]$ Å. For molecules with atom radii ~1.5 Å, this is quite narrow. If the SDF prediction is noisy or hasn't converged, most surface-band samples will have $|\phi| > 0.3$ and contribute negligible gradient to the area loss. This creates a **chicken-and-egg problem**: the loss can't guide the surface if the surface isn't already approximately correct.

3. **`grads.norm(dim=-1)` interacts with eikonal.** If eikonal perfectly enforces $|\nabla\phi| = 1$, then the area integrand reduces to $\delta_\epsilon(\phi)$, which is correct. But with $\lambda_\text{eik} = 0.1$, the gradient norm can vary widely, causing the area estimate to be noisy.

### 4.3 Volume Loss

```python
inside = smooth_heaviside(-pred_sdf, eps).mean()
return (inside - target_volume_fraction).pow(2)
```

**Assessment: Problematic design.**

1. **Target volume fraction (0.5) is physically meaningless.** The fraction of space "inside" the molecule depends entirely on the bounding box size, which is arbitrary. A larger bbox → smaller true volume fraction. There is no physical reason why 50% of the bbox should be interior. This loss will either:
   - Be ignored (if $\lambda_\text{vol}$ is small)
   - Force an artificially inflated surface (if $\lambda_\text{vol}$ is large)

2. **Should be pressure × volume, not fraction matching.** In the true VISM, the volume term is $P \int (1 - H_\epsilon(\phi)) dx$, where $P$ is the pressure. This penalizes *total exterior volume*, creating a pressure that contracts the surface. The current fraction-matching is not physically motivated.

**Recommendation:** Either (a) replace with true pressure–volume $P \cdot V_\text{exterior}$, or (b) remove this loss entirely and rely on containment + area balancing. The fraction target is more harmful than helpful.

### 4.4 Weak Prior Loss

```python
target = _batched_atomic_union_field(coords, radii, query_points).detach()
return (pred_sdf[mask] - target[mask]).abs().mean()
```

**Assessment: Useful initialization crutch, but should be annealed.**

Aligning the neural SDF to the atomic union SDF provides a warm start. However:

1. The atomic union is **not the target surface** — the VISM-equilibrium surface differs significantly (smoother, reentrant regions, hydrophobic collapse). Keeping this loss active throughout training prevents the surface from departing from the union.

2. **The `.detach()` is critical and correctly placed** — the target is treated as a fixed reference, not optimized.

3. **Should be annealed to zero** during training (or used only in Phase 1). The design doc mentions two-phase training but the code has no annealing mechanism for loss weights.

### 4.5 Containment Loss

```python
penalty = torch.relu(pred_sdf + margin).pow(2)
```

**Assessment: Correct but incomplete.**

This penalizes $\max(0, \phi(x) + m)^2$ at atom-centered points, i.e., requires $\phi(x) < -m$ near atom centers. This is a valid one-sided constraint.

**Issues:**

1. **Only enforces interior near atom centers.** Interstitial regions (between atoms, inside the molecule but far from any individual atom) have no containment signal. This allows the surface to "intrude" into molecular interiors if an area-minimizing deformation happens to thread between atoms.

2. **Margin = 0.5 is very aggressive.** This requires $\phi < -0.5$ Å near atoms. Given that a typical vdW radius is ~1.5 Å and the SES surface sits at $r + r_\text{probe} \approx 2.9$ Å from center, a margin of 0.5 is reasonable for atoms but may interfere with the surface near tight packing.

### 4.6 Missing losses

> [!WARNING]
> **The most impactful missing loss is the LJ body integral** $\rho_0 \int U_\text{LJ}(x) H_\epsilon(\phi) dx$. This is what gives VISM its physical meaning — the van der Waals interaction between solute atoms and solvent. Without it, the surface has no reason to depend on atom type (hydrophobic vs. hydrophilic), which is the *entire point* of VISM over simpler geometric models like SES.

Other missing constraints:
- **Mean curvature penalty** — present in design doc but not implemented
- **Electrostatic contribution** — acknowledged as absent, but critical for charged residues
- **Solvent exclusion topology** — no mechanism prevents topological defects (handles, self-intersections in the implicit surface at $\phi = 0$)

---

## 5. Physical Consistency

### 5.1 Solvent accessibility

❌ **Not modeled.** The code uses raw vdW radii without a probe radius. The physical SES/SAS requires inflating radii by $r_\text{probe} = 1.4$ Å (water) and then possibly rolling back for the SES reentrant surface. This is not a minor detail — the SAS is ~40% larger than the vdW surface.

### 5.2 Van der Waals radii

⚠️ **Used as input features but not as physics.** Radii enter the neighbor features and the atomic union approximation, but the VISM-specific roles (LJ well depth depends on $\sigma_{ij}$, collision radii determine excluded volume) are not encoded.

### 5.3 Electrostatics

❌ **Absent.** For charged molecules (most proteins), the electrostatic solvation free energy often dominates — sometimes exceeding the surface tension term by orders of magnitude. A useful model must eventually include at least a generalized Born (GB) or Poisson–Boltzmann (PB) approximation.

### 5.4 Translation/rotation invariance

⚠️ **Broken by the global encoder.** As noted above, `GlobalFeatureEncoder` concatenates absolute coordinates. The local features (relative positions + distances) are inherently invariant, so the local encoder is fine. The fix is straightforward: use center-of-mass-relative coordinates or drop coordinates from the global encoder entirely (use only atom embeddings + radii → mean pool).

---

## 6. Potential Failure Modes

### 🔴 6.1 Collapse to trivial solution (Critical)

The area loss has a global minimum at $\phi \to +\infty$ (no surface). The containment loss opposes this but **only at atom centers**. If the model learns $\phi < -m$ at atom centers but $\phi > 0$ everywhere between atoms, the area loss is minimized while containment is satisfied. This produces a surface that is a **union of tiny bubbles around each atom**, not a molecular envelope.

**Mitigation:**
- Add containment points at *interstitial positions* (bond midpoints, Voronoi vertices)
- Increase weak_prior weight early, then anneal
- Add a **connectivity prior** (the surface should enclose a simply connected region)

### 🔴 6.2 Eikonal-area conflict (Critical)

If the eikonal constraint ($|\nabla\phi| = 1$) is too weak, the network can minimize area by making $|\nabla\phi| \to 0$ everywhere — this drives $\delta_\epsilon(\phi) \cdot |\nabla\phi| \to 0$ even when $\phi \approx 0$, collapsing the area to zero without moving the surface. This is a well-known failure mode of level-set area formulations.

**Mitigation:** Increase $\lambda_\text{eik}$ to at least 0.5. Monitor $\mathbb{E}[|\nabla\phi|]$ during training — it should stay near 1.0.

### 🟡 6.3 Volume fraction target mismatch (High)

As discussed, the 0.5 target is arbitrary. If it's too high, the surface inflates beyond physical bounds. If too low, it collapses.

### 🟡 6.4 Gradient through `torch.autograd.grad` is expensive and fragile (Medium)

Both area and eikonal losses use `_safe_query_grads`, which calls `torch.autograd.grad(create_graph=True)`. This:
- Doubles memory usage (storing the computation graph for second-order gradients)
- Is slow for large query counts
- Can silently return zeros if the computational graph is detached somewhere upstream

**Mitigation:** Consider finite differences as a faster alternative for training (validated by DiGS, Lipman et al.), or use the recently proposed **analytical gradient computation** techniques for implicit networks.

### 🟡 6.5 Smooth-min parameter sensitivity (Medium)

The log-sum-exp atomic union with $\alpha = 10$ creates a smooth approximation where the error at "seams" (equidistant from two atoms) is $\frac{\ln 2}{\alpha} \approx 0.069$ Å. This is small but systematic — the weak prior will consistently push the neural SDF away from the true sharp union, creating a bias.

### 🟢 6.6 Batch padding artifacts (Low)

Masked padding in `collate_fn` zeros out padded atoms, and the `LocalFeatureBuilder` correctly masks them. However, `torch.cdist` still computes distances to padded (zero-coordinate) atoms before masking. For molecules with center-of-mass near the origin, padded atoms at `[0,0,0]` could incorrectly appear as "close" neighbors before the mask is applied.

---

## 7. Concrete Improvements (Prioritized)

### 🔴 P0 — Critical (do these first)

#### 1. Fix the global encoder to be translation-invariant
**Impact:** Correctness  
**Effort:** Trivial

In [global_features.py:30](file:///Users/ruyu/code/nerual-vism/src/biomol_surface_unsup/features/global_features.py#L30), replace absolute coordinates with center-of-mass-relative:

```python
# Before
x = torch.cat([coords, atom_emb, radii.unsqueeze(-1)], dim=-1)

# After
com = (coords * atom_mask.unsqueeze(-1).float()).sum(1, keepdim=True) / atom_mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
rel_coords = coords - com
x = torch.cat([rel_coords, atom_emb, radii.unsqueeze(-1)], dim=-1)
```

#### 2. Increase eikonal weight to 0.5–1.0
**Impact:** Prevents area collapse and ensures $\phi$ is a true SDF  
**Effort:** Config change

#### 3. Replace volume fraction loss with pressure–volume or remove it
**Impact:** Eliminates misleading optimization signal  
**Effort:** Small

```python
# Pressure-volume: penalize exterior volume linearly
def pressure_volume_loss(pred_sdf, mask, pressure=1.0, eps=0.1):
    exterior = smooth_heaviside(pred_sdf, eps)  # H(phi) = 1 outside
    if mask is not None:
        exterior = exterior[mask]
    return pressure * exterior.mean()
```

#### 4. Add interstitial containment points
**Impact:** Prevents bubble-collapse failure mode  
**Effort:** Medium

Sample containment points at bond midpoints and at random positions inside the convex hull of the molecule, not just near atom centers.

### 🟡 P1 — Important (high-impact improvements)

#### 5. Implement the LJ body integral loss
**Impact:** Gives the surface physical meaning beyond geometry  
**Effort:** Medium  

This is the defining term of VISM. Without it, you have a geometric smoother, not a solvation model.

```python
def lj_body_integral(pred_sdf, query_points, coords, epsilon_lj, sigma_lj, atom_mask, eps_h=0.1):
    """∫ U_LJ(x) H_ε(φ(x)) dx estimated via Monte Carlo."""
    # Compute LJ potential at each query point from all atoms
    dists = torch.cdist(query_points, coords)  # [B, Q, N]
    sigma_r6 = (sigma_lj.unsqueeze(1) / dists.clamp_min(0.1)).pow(6)  # [B, Q, N]
    u_lj = (4 * epsilon_lj.unsqueeze(1) * (sigma_r6.pow(2) - sigma_r6))  # [B, Q, N]
    u_lj = (u_lj * atom_mask.unsqueeze(1).float()).sum(-1)  # [B, Q]
    
    # Weight by Heaviside (only count "outside" region)
    h = smooth_heaviside(pred_sdf, eps_h)
    return (u_lj * h).mean()
```

#### 6. Add Fourier / positional encoding to the query decoder
**Impact:** Dramatically improves high-frequency surface detail  
**Effort:** Small

```python
class FourierEncoder(nn.Module):
    def __init__(self, d_in=3, n_freq=6):
        super().__init__()
        self.freqs = 2.0 ** torch.linspace(0, n_freq-1, n_freq)
        self.d_out = d_in * (2 * n_freq + 1)
    
    def forward(self, x):
        encoded = [x]
        for f in self.freqs.to(x.device):
            encoded.append(torch.sin(f * x))
            encoded.append(torch.cos(f * x))
        return torch.cat(encoded, dim=-1)
```

Include this in the decoder input: `z = concat(z_local, z_global, fourier_encode(query_pos))`.

#### 7. Replace DeepSets with SchNet-style message passing
**Impact:** Captures neighbor-neighbor interactions critical for pocket geometry  
**Effort:** Medium

A minimal continuous-filter convolution:

```python
class ContinuousFilterConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, rbf_dim):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )
    
    def forward(self, h, rbf, mask):
        # h: [B, Q, K, F], rbf: [B, Q, K, rbf_dim], mask: [B, Q, K]
        W = self.filter_net(rbf)  # [B, Q, K, F]
        messages = (h * W * mask.unsqueeze(-1).float()).sum(dim=2)  # [B, Q, F]
        return self.update(messages)
```

#### 8. Implement loss weight annealing
**Impact:** Enables the two-phase training strategy described in the design doc  
**Effort:** Small

```python
class LossWeightScheduler:
    def __init__(self, initial_weights, final_weights, warmup_epochs):
        self.initial = initial_weights
        self.final = final_weights
        self.warmup = warmup_epochs
    
    def get_weights(self, epoch):
        t = min(epoch / self.warmup, 1.0)
        return {k: self.initial[k] * (1-t) + self.final[k] * t 
                for k in self.initial}
```

Phase 1: high `weak_prior` (1.0), zero `area`/`volume`  
Phase 2: anneal `weak_prior` → 0, ramp up `area`/`volume`/`lj`

#### 9. Use SIREN activations in the SDF decoder
**Impact:** Much better representation of sharp surface features  
**Effort:** Small

```python
class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega_0=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega_0 = omega_0
        # Sitzmann et al. initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_dim, 1/in_dim)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6/in_dim) / omega_0,
                    math.sqrt(6/in_dim) / omega_0,
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
```

### 🟢 P2 — Nice to have

#### 10. Add probe radius to the surface definition
**Effort:** Trivial — `radii = radii + r_probe` before building features

#### 11. Implement mean curvature loss term
**Effort:** Medium — requires second-order derivatives, computationally expensive

#### 12. Connect the FiLM decoder as the default
**Effort:** Small — wire `FiLMDecoder` into `SurfaceModel` with `z_local`, `z_global` as separate inputs

#### 13. Add adaptive surface-band sampling
**Effort:** Medium — use the current model's SDF prediction to refine sampling during training

#### 14. Center padded atoms at infinity instead of zero
**Effort:** Trivial — `coords.masked_fill(~atom_mask.unsqueeze(-1), 1e6)` before distance computation

---

## Summary: Impact vs. Effort Matrix

```
                     High Impact
                         │
          ┌──────────────┼──────────────┐
          │  P0.1 Fix    │  P1.5 LJ     │
          │  global enc  │  body integral│
          │              │              │
          │  P0.2 Eik    │  P1.7 SchNet │
          │  weight ↑    │  encoder     │
          │              │              │
 Low ─────│  P0.3 Fix    │  P1.6 Fourier│──── High
 Effort   │  vol loss    │  encoding    │  Effort
          │              │              │
          │  P1.9 SIREN  │  P1.8 Weight │
          │  decoder     │  annealing   │
          │              │              │
          │  P0.4 Intrstl│  P2.11 Mean  │
          │  containmnt  │  curvature   │
          └──────────────┼──────────────┘
                         │
                     Low Impact
```

> [!IMPORTANT]
> **The single most impactful change is P0.2 + P0.3**: increase eikonal weight and fix the volume loss. These are config/one-line changes that prevent the most likely failure modes. The single most impactful *architectural* change is P1.6: adding Fourier encoding of query point coordinates to the decoder input.

---

## References

- Gropp et al. (2020). "Implicit Geometric Regularization for Learning Shapes." *ICML*. — IGR, eikonal as primary SDF regularizer
- Sitzmann et al. (2020). "Implicit Neural Representations with Periodic Activation Functions." *NeurIPS*. — SIREN
- Satorras et al. (2021). "E(n) Equivariant Graph Neural Networks." *ICML*. — EGNN
- Schütt et al. (2021). "Equivariant Message Passing for the Prediction of Tensorial Properties and Molecular Spectra." *ICML*. — PaiNN
- Dzubiella, Swanson, McCammon (2006). "Coupling hydrophobicity, dispersion, and electrostatics in continuum solvent models." *PRL*. — Original VISM
- Zhou et al. (2014). "Variational Implicit-Solvent Modeling of Host–Guest Binding." *JCTC*. — VISM with LJ body integral
