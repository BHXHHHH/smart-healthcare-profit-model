# Method-to-code mapping

The implementation follows the numerical specification in the paper.

| Paper component | Implementation |
|---|---|
| Baseline congestion equilibrium, Eqs. (3)-(6) | `reproduce_single.solve_rho`, `doctor_equilibrium`, `baseline_pi_vec` |
| Baseline profit, Eq. (7) | `reproduce_single.baseline_pi_vec` |
| Dual-track costs and thresholds, Eqs. (8)-(13) | `reproduce_single.dual_pi_vec` |
| Physician review equilibrium, Eqs. (14)-(17) | `reproduce_single.doctor_equilibrium` |
| Dual-track profit, Eq. (18) | `reproduce_single.dual_pi_vec` |
| Critical fixed cost and mechanism calculations | `reproduce_single.analysis` |
| Phase diagrams and robustness sweeps | `reproduce_single.analysis` |
| Full-grid profit-difference distribution, Figure 1 | `reproduce_single.figure1` |

The paper's Eq. (13) should use `theta1 = A1 / beta1`, because `theta1` is
the threshold where `u1(theta) = 0`. The current code uses that mathematically
consistent expression. The `theta0` occurrence in the paper is treated as a
notation typo and is not changed in the solver.

The four counterfactual values are identified by the keys in
`expected_outputs/counterfactual.csv`: `eff_gain`, `esc_loss`, `cap_wait_loss`,
and `fixed_cost`. The labels in the paper's Table 4 do not match all four
values; the CSV keys and `reproduce_single.analysis` are the executable record.

| CSV field | Meaning | Reference value |
|---|---|---|
| `eff_gain` | Profit gain relative to e=1 | 0.001120848 |
| `esc_loss` | Profit without referral minus actual profit | 0.019890224 |
| `cap_wait_loss` | Profit at C0=100 minus actual profit | 0.004511232 |
| `fixed_cost` | AI fixed cost K_AI | 0.03 |

For Figure 1, the histogram is built from every `dPi` value in
`core_grid.csv` (36 bins), with a vertical `Delta Pi = 0` reference line and
the positive/negative shares annotated. It is therefore a direct visualization
of the full parameter grid, not a random sample.
