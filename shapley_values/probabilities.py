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
    print(f"numerator_indices: {numerator_indices}")
    numerator = get_probability(unique_count, x_hat, numerator_indices, n)

    print(f"Indices {indices}")
    print(x_hat)
    denominator = get_probability(unique_count, x_hat, indices, n)
    print(f"numerator: {numerator}")
    print(f"Denominator: {denominator}")
    print("----------------")

    return  numerator / (denominator + 1e-7)

def causal_prob(unique_count, x_hat, indices, indices_baseline, causal_struc, n):
    p = 1
    for i in indices_baseline:
        intersect_s, intersect_s_hat = [], []
        intersect_s_hat.append(i)
        if len(causal_struc[str(i)]) > 0:
            for index in causal_struc[str(i)]:
                if index in indices or index in indices_baseline:
                    intersect_s.append(index)
            p *= conditional_prob(unique_count, x_hat, intersect_s, intersect_s_hat, n)
        else:
            p *= get_probability(unique_count, x_hat, intersect_s_hat, n)
    return p


