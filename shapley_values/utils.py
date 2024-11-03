def get_baseline(X, model) -> float:
    """ Returns baseline value for computing ShapleY Values as averaged prediction across trainign set"""
    return model.predict(X).mean()
