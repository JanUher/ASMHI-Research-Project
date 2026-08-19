| step                   | detail                                                                                        |
|:-----------------------|:----------------------------------------------------------------------------------------------|
| Train/test split       | GroupShuffleSplit — 80% train pool / 20% held-out test                                        |
| Group key              | RADAR: participant_id; Androids: file_stem                                                    |
| Cross-validation       | 5-fold GroupKFold on train pool only                                                          |
| Classifier tuning      | GridSearchCV, scoring = ROC-AUC (GBC, RFC, GPBoost, SVC)                                      |
| Regressor tuning       | GridSearchCV, scoring = neg-RMSE (GBR, RFR, SVR)                                              |
| SVM models             | SVC (classification), SVR (regression), MERF-SVR (mixed-effects); RBF kernel, scaled features |
| Binary from regression | RADAR: ŷ ≥ 10; Androids: ŷ ≥ BDI train midpoint                                               |
| MERF                   | Fixed-effects model tuned on X; MERF fit per CV fold with Z covariates                        |
| Wilcoxon baseline      | Classifiers: stratified dummy; Regressors/MERF: train-mean                                    |
| Fairness               | AIF360 disparate impact + statistical parity (gender)                                         |
| Held-out reporting     | test_summary + confusion_matrices sheets in Excel logs                                        |