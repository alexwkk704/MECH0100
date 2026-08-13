# Parameter dependence — ADMS-DF lattice
*Alex Wong, MECH0100 · 2026-07-13 · answers Week 8 action #2 (Farooq, 10-Jul meeting)*

## TL;DR

The five input parameters split cleanly into **four roles**:

| Rank | Parameter | Fold ΔE at fixed baseline | E* residual (× beyond ρ) | Role |
|---|---|---|---|---|
| 1 | **Thickness** | **9.94×** | 1.20× | Density knob (biggest raw effect) |
| 2 | **Density** (Spherene param) | **7.81×** | 1.21× | Density knob |
| 3 | Inner_Size | 1.26× | **1.60×** | Structural / RVE knob |
| 4 | Size_Multi | 1.23× | 1.11× | Weak boundary-cut noise |
| 5 | Seed | **1.00×** | 1.00× | **INERT** at Alex's Spherene SDK |

**"Fold ΔE"** = max E / min E over that parameter's 1D slice.
**"E* residual"** = max(E*) / min(E*) where E* = (E/E_s) / ρ^1.83 removes the density scaling. Numbers close to 1.0 mean the parameter acts purely through density; larger numbers mean the parameter changes topology on top of density.

## Two families of knobs

**Density-driving (Thickness, Density).** Both give huge fold-changes in E, G and yield onset, but E* residuals collapse to ~1.2× — meaning virtually all of that effect is captured by the Gibson-Ashby ρ^1.83 law. In practice: either D or t can be used to reach a target ρ, and the resulting stiffness is the same. This is why the Wk7 merged plot showed the density-path and thickness-path collapsing onto one curve (E/Es = 0.87·ρ^1.83).

**Structural (Inner_Size).** Small fold ΔE (1.26×) but *big* residual (1.60×) — the parameter changes something beyond ρ. This is the RVE-size convergence effect I saw in Wk7: at 3³ cells the TAI is ~0.21 and stiffness is 30% below the ρ-fit; by 5³ the TAI drops to ~0.05 and stiffness matches the fit. Larger inner sizes are more isotropic and closer to a bulk material property. **Use 5³ (Inner_Size = 15) as the standard RVE.** Going to 6³ costs mesh time; going below 5³ underestimates stiffness.

**Weak (Size_Multi).** ρ is nearly constant (0.194–0.212) so it isn't a density knob. E fluctuates 8.9–11.0 GPa (1.23×) because the outer-box cut passes through different ligaments at different multipliers — mesh-boundary noise, not a physical effect. Recommendation: **fix Size_Multi = 1.6** for the whole surrogate dataset.

**Inert (Seed).** All four seed values I tested (1, 2, 3, 5) at the reference config give **bit-identical** ρ_rel = 0.19430, E = 9.367 GPa, G = 3.706 GPa, TAI = 0.056. The Spherene founder's documented seed-variant behaviour (single-digit-% property spread) is *not* reproduced on my SDK. This is a plugin bug on my end — reported to Chiara / Spherene, and I use density perturbation (D ± 0.2) to sample structural variation instead.

## What this means for the surrogate

- **Density and Thickness are redundant on ρ.** For the ML forward model, ρ_rel alone captures ~all stiffness/strength info from these two. You could drop one from the feature set without losing predictive power.
- **Inner_Size is the only "true" structural feature** among the five inputs. It should be in the feature set (or dropped only if you standardise on 5³).
- **Size_Multi and Seed can be fixed** and removed from the sweep.
- Net: the effective design space is essentially 2D — (ρ_rel, RVE_size).

## Caveats

- Density and Thickness slices at baseline (Inner_Size=15, Size_Multi=1.6, Seed=2) are the largest (N=7 and N=4). Inner_Size, Size_Multi, Seed slices are smaller because I explored fewer values.
- The E* residual metric assumes the Gibson-Ashby law (E ∝ ρ^1.83) is the correct null model. It is — R² = 0.988 on the 68-point deduped fit set (see master Results_summary.xlsx charts sheet).
- Yield-onset numbers only exist for a subset (Density and Thickness slices, plus baseline). Onset residuals are not analysed here — they follow ρ^1.55 with R² = 0.994.

## Data source

`Share/Results/Results_summary.xlsx` → sheet `parameter_effects` (verdict table + per-parameter data blocks + fold-ΔE bar chart) and sheet `charts` (raw scatter plots E/Es, G/Gs, onsets vs ρ_rel). 78 rows across 9 batches.
