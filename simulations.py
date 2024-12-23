import numpy as np

def simulation1(num_samples = 1000):
    np.random.seed(0)
    features, prediction = [], []
    for _ in range(num_samples):
        p = np.random.uniform(0, 1.5)
        f = 2 * (p ** 3) + np.random.normal(loc=0, scale=0.2)
        s = f - p ** 2 + np.random.normal(loc=0, scale=0.2)
        features.append([p, f])
        prediction.append(s)

    return np.array(features), np.array(prediction)

def simulation2(num_samples = 1000):
    np.random.seed(0)
    features, prediction = [], []
    for _ in range(num_samples):
        f = np.random.normal(loc=1, scale=3)
        p = f + np.random.normal(loc=0, scale=1)
        s = f - np.random.normal(loc=0, scale=0.2)
        features.append([p, f])
        prediction.append(s)

    return np.array(features), np.array(prediction)
