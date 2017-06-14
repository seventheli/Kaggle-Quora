import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def main():
    train = pd.read_csv('./input/train.csv')
    train_cliques = pd.read_csv('./input/train_cliques.csv')
    train_leaky = pd.read_csv('./input/train_leaky.csv')
    train_refeatured = pd.read_csv('./input/train_refeatured.csv')
    train_refeatured_2 = pd.read_csv('./input/train_refeatured_2.csv')
    train_k_core = pd.read_csv('./input/train_kcore.csv')
    X_train = pd.concat((train_cliques, train_refeatured, train_refeatured_2, train_leaky, train_k_core), axis=1)
    y_train = train['is_duplicate'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)

    # UPDownSampling
    pos_train = X_train[y_train == 1]
    neg_train = X_train[y_train == 0]
    X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
    y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = X_valid[y_valid == 1]
    neg_valid = X_valid[y_valid == 0]
    X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.005,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 321,
        'feval': 1,
    }
    # params['scale_pos_weight'] = 0.2
    params['nthread'] = 4
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 8000, watchlist, early_stopping_rounds=100, verbose_eval=50)
    print(log_loss(y_valid, bst.predict(d_valid)))

    print('Building Test Features')
    test = pd.read_csv('./input/test.csv')
    test_cliques = pd.read_csv('./input/test_cliques.csv')
    test_leaky = pd.read_csv('./input/test_leaky.csv')
    test_refeatured = pd.read_csv('./input/test_refeatured.csv')
    test_refeatured_2 = pd.read_csv('./input/test_refeatured_2.csv')
    test_k_core = pd.read_csv('./input/test_kcore.csv')
    x_test = pd.concat((test_cliques, test_refeatured, test_refeatured_2, test_leaky, test_k_core), axis=1)
    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)
    sub = pd.DataFrame()

    sub['test_id'] = test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('./predictions/result.csv', index=False)


if __name__ == '__main__':
    main()
