| dataset   | category   | model            | display_name              | selection_metric   |   selection_value |
|:----------|:-----------|:-----------------|:--------------------------|:-------------------|------------------:|
| RADAR     | classifier | RADAR SVC        | Support Vector Classifier | roc_auc_mean       |          0.583951 |
| RADAR     | regressor  | RADAR SVR        | Support Vector Regressor  | mae_mean           |          4.9535   |
| RADAR     | merf       | RADAR MERF-RF    | ME RFRegressor            | roc_auc_mean       |          0.575329 |
| Androids  | classifier | Androids SVC     | Support Vector Classifier | roc_auc_mean       |          0.812722 |
| Androids  | regressor  | Androids SVR     | SVM Regressor             | mae_mean           |         11.6989   |
| Androids  | merf       | Androids MERF-RF | ME RFRegressor            | roc_auc_mean       |          0.697633 |