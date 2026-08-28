| dataset   | category   | model            | display_name              | selection_metric   |   selection_value |
|:----------|:-----------|:-----------------|:--------------------------|:-------------------|------------------:|
| RADAR     | classifier | RADAR SVC        | Support Vector Classifier | roc_auc_mean       |          0.583951 |
| RADAR     | regressor  | RADAR SVR        | Support Vector Regressor  | mae_mean           |          4.9535   |
| RADAR     | gpboost    | RADAR GPBoost    | GPBoost                   | roc_auc_mean       |          0.520961 |
| Androids  | classifier | Androids SVC     | Support Vector Classifier | roc_auc_mean       |          0.812722 |
| Androids  | regressor  | Androids SVR     | Support Vector Regressor  | mae_mean           |         11.6989   |
| Androids  | gpboost    | Androids GPBoost | GPBoost                   | roc_auc_mean       |          0.571429 |