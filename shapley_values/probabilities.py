import numpy as np

def get_probability(unique_count, x_hat, indices_baseline, n):
    if len(indices_baseline) == 0:
        return 1

    count = 0
    x_hat_array = np.asarray(x_hat)
    
    for key, occurrences in unique_count.items():
        if all(key[j] == x_hat_array[j] for j in indices_baseline):
            count += occurrences

    return count / n

def conditional_prob(unique_count, x_hat, indices, indices_baseline, n):
    numerator_indices = indices + indices_baseline
    numerator = get_probability(unique_count, x_hat, numerator_indices, n)
    denominator = get_probability(unique_count, x_hat, indices, n)
    return  numerator / (denominator + 1e-7)


