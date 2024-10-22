def get_baseline(X, model):
    return model.predict(X).mean()
