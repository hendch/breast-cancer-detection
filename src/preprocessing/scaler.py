from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform(scaler, X):
    return scaler.transform(X)
