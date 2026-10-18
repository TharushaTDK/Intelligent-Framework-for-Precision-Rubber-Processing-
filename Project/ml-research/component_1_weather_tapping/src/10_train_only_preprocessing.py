from stage_utils import *
text='Training protocol notes\n\nThe model-comparison scripts fit preprocessing pipelines on each training split. Numeric imputation, categorical imputation, one-hot encoding and any scaling are contained inside each sklearn Pipeline/ColumnTransformer to avoid fitting these transformations on the held-out test rows.\n\nThis project keeps collection-time prediction removed. DRC and price are regression tasks; tapping suitability is a synthetic proxy classification task.\n'
(REPORTS/'10_train_only_preprocessing_protocol.txt').write_text(text,encoding='utf-8')
print('Stage 10 protocol note saved.')
