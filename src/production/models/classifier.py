import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def classifier(features_df):
    
    X = features_df.drop('Target',axis = 1)
    X_train = X.iloc[:-1]
    y_train = features_df['Target'].iloc[:-1]

    # TODO: Make sure this is right
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    lgb_train = lgb.Dataset(X_train, y_train, weight = sample_weights)

    params = {
                          'objective': 'multiclass',
                          'metric': 'multi_logloss',
                          'num_class': 3,
                          'boosting_type': 'gbdt',
                          'num_leaves': 31,
                          'learning_rate': 0.1,
                          'feature_fraction': 1.0,
                          'bagging_fraction': 1.0,
                          'bagging_freq': 0,
                          'min_child_samples': 20,
                          'verbose': -1
                      }


    model = lgb.train(params, lgb_train)
    probs = model.predict(X.tail(1))

    trade = np.argmax(probs, axis=1)

    return trade

