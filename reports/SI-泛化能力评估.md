# Supplementary Note S-x — Generalization along the macrocyclic scaffold and the meso-substituent axes

This note reports how the models behave when the test molecules share no macrocyclic scaffold, or no meso-substituent, with the training molecules. The main text reports the random five-fold split, i.e. the interpolation regime; the two group-wise splits described here delimit where that performance does and does not carry over.

## S-x.1 Protocol

All experiments use the 1,734 molecules described in Section 2 and the same two feature sets: 1,613 Mordred descriptors, and the concatenation of Mordred with four fingerprint families (9,972 columns). Within each training fold, columns that are constant, all-NaN, or redundant at |Pearson r| > 0.95 with a stronger correlate of the target are removed, leaving 510–526 Mordred columns or 3,815–4,055 concatenated columns depending on the fold. Ten regressors are tuned with Optuna (100 trials each, five-fold CV inside the training partition, seed 42) and evaluated on a held-out test partition.

Three split definitions are used:

| Split | Grouping variable | Training | Test | Held-out groups |
|---|---|---|---|---|
| Random (main text) | none (i.i.d. 80/20 split, seed 42) | 1,387 | 347 | — |
| Scaffold (A axis) | `A_number`, 18 macrocyclic scaffolds | 1,383 | 351 | A02, A03, A14, A18 |
| meso (B axis) | `B_number`, 16 meso-substituent groups | 1,398 | 336 | B05, B11, B15 |

Two complementary protocols are reported for each group-wise split. The **held-out split** mirrors the main-text protocol: the test groups are removed once, the models are tuned on the remaining groups, and the reported metrics come from that single test partition. The **leave-one-group-out protocol** rotates the held-out groups over five folds so that every group serves as validation exactly once; it has no fixed test set and is therefore reported as pooled out-of-fold MSE and R² over all 1,734 molecules. Because the fold partitions of the group-wise splits differ in size and in target variance, the pooled statistic is used for cross-axis comparison and the per-fold values are listed alongside.

B00 requires a note. B00 is the unsubstituted parent (hydrogen at all four meso positions) and is used by only two of the 18 scaffolds (A13 and A14); if it were held out, the meso split would simultaneously become a scaffold extrapolation. B00 is therefore kept on the training side in every fold, and its own error is reported separately in S-x.3 as the extreme case of extrapolating back to the parent.

## S-x.2 Generalization across macrocyclic scaffolds (A axis)

Under the scaffold split the best cross-validated model reaches a test R² of only +0.374 (RandomForest, Mordred) and +0.284 (SVR, concatenated), against 0.932 and 0.963 for the same feature sets under the random split. Averaged over all ten regressors the ranking of the two feature sets is reversed: fingerprints lower the mean CV MSE by 0.121 eV² under the random split but raise it by 0.189 eV² under the scaffold split.

**Table S??.1** — Held-out scaffold split: CV MSE and test metrics of the ten regressors, Mordred and concatenated feature sets.

| Model | CV MSE (eV²) [S-M] | Test R² [S-M] | Test RMSE (eV) [S-M] | Test MAE (eV) [S-M] | CV MSE (eV²) [S-MF] | Test R² [S-MF] | Test RMSE (eV) [S-MF] | Test MAE (eV) [S-MF] |
|---|---|---|---|---|---|---|---|---|
| Lasso | 0.6192 | +0.0909 | 0.9047 | 0.6717 | 1.2170 | -0.0014 | 0.9495 | 0.7127 |
| ElasticNet | 0.6223 | +0.0846 | 0.9078 | 0.6733 | 1.1846 | +0.2468 | 0.8234 | 0.6285 |
| SVR | 1.0219 | -0.4093 | 1.1264 | 0.8554 | 1.1748 | +0.2843 | 0.8027 | 0.5943 |
| KernelRidge | 1.0199 | +0.0214 | 0.9386 | 0.7369 | 1.2053 | +0.0982 | 0.9010 | 0.6758 |
| KNeighbors | 1.1882 | -0.0855 | 0.9886 | 0.7016 | 1.2308 | +0.2337 | 0.8306 | 0.6357 |
| DecisionTree | 2.9328 | -0.0208 | 0.9587 | 0.7350 | 2.9328 | -0.0208 | 0.9587 | 0.7350 |
| RandomForest | 2.5315 | +0.3741 | 0.7506 | 0.5494 | 2.8293 | +0.0488 | 0.9254 | 0.6171 |
| XGBoost | 1.2542 | +0.0010 | 0.9484 | 0.7113 | 1.2779 | +0.0005 | 0.9486 | 0.7115 |
| LightGBM | 1.2649 | +0.0530 | 0.9234 | 0.6913 | 1.2733 | +0.0026 | 0.9476 | 0.7040 |
| GaussianProcess | 1.1927 | -1.0759 | 1.3671 | 1.0202 | 1.2136 | +0.2211 | 0.8374 | 0.6366 |

Best CV model: Lasso (0.6192 eV², Mordred) and SVR (1.1748 eV², concatenated). Best test R²: RandomForest (+0.3741) and SVR (+0.2843). Note that the CV-selected model is not the best model on the held-out scaffolds, so the selection criterion itself is unreliable in this regime.

**Table S??.2** — Controlled comparison of the three axes with identical features, preprocessing and hyperparameters (the B-M settings), reported as pooled out-of-fold CV MSE / pooled R² over all 1,734 molecules.

| Model | Random 5-fold | meso leave-one-group-out | Scaffold GroupKFold |
|---|---|---|---|
| RandomForest | 0.1881 / +0.8247 | 0.1696 / +0.8419 | 0.8994 / +0.1614 |
| XGBoost | 0.3063 / +0.7144 | 0.3177 / +0.7038 | 0.9288 / +0.1340 |
| KNeighbors | 0.5790 / +0.4601 | 0.3569 / +0.6673 | 0.9102 / +0.1513 |
| Lasso | 0.4682 / +0.5634 | 0.4712 / +0.5607 | 1.0011 / +0.0666 |

The scaffold column is the quantitative statement of the limitation: RandomForest moves from 0.1881 to 0.8994 eV² (R² +0.825 → +0.161), and every model lands only 7–16% below the constant baseline of 1.0725 eV². Whatever the regressor, an unseen macrocyclic scaffold is not predicted.

## S-x.3 Generalization across meso-substituents (B axis)

Holding the macrocyclic scaffold distribution fixed and instead removing three whole meso-substituent groups leaves performance close to the interpolation regime: best CV MSE 0.4783 eV² (KNeighbors, Mordred) and 0.1767 eV² (KNeighbors, concatenated), best test R² +0.8667 (RandomForest) and +0.9631 (RandomForest). Adding fingerprints helps on this axis (−0.060 eV² on average, 8 of 10 models improved), in contrast to the scaffold axis.

**Table S??.3** — Held-out meso split: CV MSE and test metrics of the ten regressors, Mordred and concatenated feature sets.

| Model | CV MSE (eV²) [B-M] | Test R² [B-M] | Test RMSE (eV) [B-M] | Test MAE (eV) [B-M] | CV MSE (eV²) [B-MF] | Test R² [B-MF] | Test RMSE (eV) [B-MF] | Test MAE (eV) [B-MF] |
|---|---|---|---|---|---|---|---|---|
| Lasso | 0.4878 | +0.4661 | 0.7370 | 0.5588 | 0.4570 | +0.8336 | 0.4114 | 0.3062 |
| ElasticNet | 0.4876 | +0.4664 | 0.7367 | 0.5589 | 0.4597 | +0.8426 | 0.4002 | 0.2973 |
| SVR | 0.5733 | +0.6227 | 0.6195 | 0.4705 | 0.4716 | +0.9049 | 0.3111 | 0.2042 |
| KernelRidge | 0.5569 | +0.6141 | 0.6266 | 0.4922 | 0.4082 | +0.9274 | 0.2717 | 0.1758 |
| KNeighbors | 0.4783 | +0.7369 | 0.5173 | 0.3548 | 0.1767 | +0.9412 | 0.2445 | 0.1354 |
| DecisionTree | 1.4671 | +0.4848 | 0.7239 | 0.5125 | 1.4055 | +0.4848 | 0.7239 | 0.5125 |
| RandomForest | 0.7400 | +0.8667 | 0.3682 | 0.2308 | 1.1031 | +0.9631 | 0.1936 | 0.1240 |
| XGBoost | 0.5465 | +0.7198 | 0.5339 | 0.3569 | 0.4962 | +0.8161 | 0.4325 | 0.2840 |
| LightGBM | 0.4855 | +0.7979 | 0.4534 | 0.3338 | 0.5390 | +0.8267 | 0.4198 | 0.3235 |
| GaussianProcess | 0.7834 | +0.2841 | 0.8534 | 0.6640 | 0.4909 | +0.9317 | 0.2636 | 0.1682 |

**Table S??.4** — meso leave-one-group-out. Each of the 15 substituent groups is held out once, with B00 always on the training side. The five folds hold out B(08,09,10), B(03,11,12), B(01,02,13), B(04,14,15) and B(05,06,07) respectively; "Fold MSE" lists the five fold errors in that order.

| Model | Pooled MSE | Pooled R² | Fold MSE |
|---|---|---|---|
| RandomForest | 0.1696 | +0.8419 | 0.109 / 0.141 / 0.178 / 0.218 / 0.202 |
| XGBoost | 0.3177 | +0.7038 | 0.296 / 0.338 / 0.286 / 0.315 / 0.353 |
| KNeighbors | 0.3569 | +0.6673 | 0.227 / 0.351 / 0.452 / 0.532 / 0.223 |
| Lasso | 0.4712 | +0.5607 | 0.420 / 0.604 / 0.451 / 0.484 / 0.398 |

RandomForest reaches 0.1696 eV² (R² +0.842), i.e. it is as accurate when whole substituent groups are removed as under the random split (0.1881 eV²). The meso axis is therefore an interpolation axis, not an extrapolation axis.

**Table S??.5** — B00 (unsubstituted parent) held out: the single case in which the model must predict the absence of a meso substituent.

| Model | MSE (B00) | R² (B00) | Bias (eV) | Mean prediction (eV) | Constant baseline (eV²) |
|---|---|---|---|---|---|
| RandomForest | 9.7130 | -7.9601 | +2.78 | 8.97 | 1.0945 |
| XGBoost | 2.1998 | -1.0293 | +1.07 | 7.26 | 1.0945 |
| KNeighbors | 0.8232 | +0.2406 | +0.31 | 6.50 | 1.0945 |
| Lasso | 0.9432 | +0.1299 | +0.47 | 6.65 | 1.0945 |

The parent is not predicted by the tree ensembles: RandomForest gives 8.97 eV against a true mean of 6.19 eV, a bias of +2.78 eV and an MSE of 9.7130 eV², i.e. 8.9× worse than the constant baseline of 1.0945 eV². The error persists after the two scaffolds that never appear with a substituted analogue are removed (MSE 10.2869 eV², n = 106), so it is not a scaffold artefact: the model has learned that a meso substituent shifts the gap and cannot represent its removal. Kernel and instance-based models degrade gracefully to the conditional mean instead.

## S-x.4 Interpretation

**(a) The generalization boundary lies on the macrocyclic scaffold, not on the meso-substituent.** Removing three substituent groups costs almost nothing relative to the random split (pooled R² 0.84 vs 0.82 for RandomForest), whereas removing four scaffolds costs essentially the whole signal (pooled R² 0.16). The same estimator, feature set and preprocessing produce the two outcomes, so the difference is a property of the chemistry being held out, not of the learner. A single random-split R² therefore says nothing about whether a model can be used to triage molecules that lie outside the training scaffolds, and we recommend reporting the split axis whenever such a claim is made.

**(b) The macrocyclic scaffold is the governing degree of freedom of this molecular family, and it is precisely the direction in which a learned model cannot help.** The frontier orbitals of a porphyrinoid are set by the size and connectivity of the conjugated macrocycle, and the data show this directly: across the 18 macrocyclic scaffolds the mean gap ranges from 5.18 eV (A14) to 10.44 eV (A04), a 5.3 eV span, whereas within a fixed scaffold the standard deviation over all substituent and metalation combinations is only 0.17–1.43 eV. A model trained on a subset of scaffolds has no basis for the remaining ones, which is exactly what the scaffold split measures. Extending the scaffold coverage is therefore not a convenience but the only way to extend the applicability domain of the predictor, and it is why this work treats the 18 macrocyclic scaffolds as the primary design axis of the database rather than as one more categorical feature. The same result defines the boundary condition for prospective use: the models are reliable screeners inside a known scaffold family, and a new macrocyclic scaffold must be added to the training set (by calculation or measurement) before the model can rank candidates on it.

**(c) Practical consequence for the reported metrics.** The random-split numbers of the main text describe interpolation and are appropriate for ranking candidates within the covered scaffolds. For a prospective campaign that proposes new macrocycles, the relevant error bars are those of Tables S??.1 and S??.2: CV MSE between 0.62 and 2.93 eV² across the ten regressors in the held-out scaffold split, and 0.90–1.00 eV² in the controlled leave-scaffolds-out protocol, rather than the 0.09–0.60 eV² obtained under the random split.

## S-x.5 Terminology

Throughout the manuscript and this SI: **macrocyclic scaffold** denotes the porphyrinoid ring system, i.e. the conjugated macrocycle including its peripheral ring modifications and the central metalation state, indexed A01–A18 (11 of the 18 are reported in metalated forms); **meso-substituent** denotes the substituent at the four meso positions, grouped as B00–B15 in 16 substitution patterns. Where a shorter form is needed after first use, “scaffold” and “substituent” refer to these two objects and to nothing else. The earlier descriptors “macrocyclic topology” and the figure-axis label “molecular backbone index” refer to the same object as “macrocyclic scaffold” and are used interchangeably in the figure files; the text uses the single term defined here.

