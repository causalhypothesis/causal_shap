import itertools
import numpy as np
import random
import collections
import math as mt
from shapley_values.probabilities import conditional_prob, get_probability
from shapley_values.utils import get_baseline
from shapley_values.exceptions import CausalGraphException
from enum import Enum
from pydantic import BaseModel
from typing import Any, List, Dict


random.seed(42)


class ShapleyValuesType(Enum):
    MARGINAL = 'MARGINAL'
    CONDITIONAL = 'CONDITIONAL'
    CAUSAL = 'CAUSAL'


class Explainer(BaseModel):
    X: Any
    model: Any
    is_classification: bool = False
    X_counter: collections.Counter = collections.Counter()
    rounding_precision: int = 2

    def __init__(self, **data):
        super().__init__(**data)
        self.X = np.round(self.X, self.rounding_precision)
        self.X_counter = collections.Counter(map(tuple, self.X))

    def compute_shapley_values(self, sample: List, type = ShapleyValuesType.MARGINAL, causal_struct: Dict = None):
        """
        Computes the attribution of each feature for a given sample using Shapley values.

        Args:
            sample (list): A list of feature values representing the input sample for which attributions are computed.
            type (ShapleyValuesType): An enumeration value indicating the type of Shapley values to compute. 
                Options include:
                    - ShapleyValuesType.MARGINAL
                    - ShapleyValuesType.CONDITIONAL
                    - ShapleyValuesType.CAUSAL
            causal_struct (Dict): A dictionary defining the causal structure of the model, which is essential for causal calculations.

        Prints:
            Baseline Value (E[f(X)]):
                The average predicted value across all training samples, serving as a reference point for attributions.
            Predicted Value (f(x)):
                The model's predicted output for the specified sample, reflecting the influence of the features.
            Shapley Values + Baseline Value:
                A combined metric that merges the calculated Shapley value attributions with the expected value of the predicted outcome, providing a comprehensive view of feature contributions.

        Raises:
            CausalGraphMissedException: If the provided causal graph is incorrect or incomplete, indicating a failure to compute attributions properly.

        Returns:
            list: An array containing the attribution values for each feature, ordered according to the input sample.
        """

        if type == ShapleyValuesType.CAUSAL and not causal_struct:
            raise CausalGraphException(
                "Causal graph has to be provided for computing Causal Shapley Values")
        
        sample = np.round(sample, self.rounding_precision)
        n_features = self.X.shape[-1]
        phis = []
        for feature in range(n_features):
            local_shap_score = self.approximate_shapley(feature, sample, type,
                                                        causal_struct)
            phis.append(local_shap_score[0])

        # Check if the sum of the Shapley values and expected value adds up to the prediction
        x = np.reshape(sample, (1, n_features))
        f_x = get_baseline(self.X, self.model)

        print("Baseline Value (E[f(X)]): ", f_x)
        print("Predicted Value (f(x)) ", self.model.predict(x))
        print("Shapley Values + (E[f(X)]): ",
              round(float(sum(phis) + f_x), 3))

        return phis

    def approximate_shapley(self, xi, x, type, causal_struct=None):
        N = self.X.shape[-1]
        m = mt.factorial(N)
        R = list(itertools.permutations(range(N)))
        random.shuffle(R)
        score = 0
        count_negative = 0
        vf1, vf2 = 0, 0
        for i in range(m):
            abs_diff, f1, f2 = self.get_value(type, list(R[i]), x, causal_struct,
                                              xi)

            vf1 += f1
            vf2 += f2
            score += abs_diff
            if vf2 > vf1:
                count_negative -= 1
            else:
                count_negative += 1
        if count_negative < 0:
            score = -1 * score
        return score / m

    def get_value(self, type, permutation, x, causal_struct, xi):
        N = self.X.shape[-1]

        lenX = self.X.shape[0]
        absolute_diff, f1, f2 = 0, 0, 0
        xi_index = permutation.index(xi)
        indices = permutation[:xi_index + 1]
        indices_baseline = permutation[xi_index + 1:]
        x_hat = np.zeros(N)
        x_hat_2 = np.zeros(N)

        for j in indices:
            x_hat[j] = x[j]
            x_hat_2[j] = x[j]

        baseline_check_1, baseline_check_2 = [], []
        f1, f2 = 0, 0
        indices_baseline_2 = indices_baseline[:]
        for i in self.X_counter:
            X = np.asarray(i)
            for j in indices_baseline:
                x_hat[j] = X[j]
                x_hat_2[j] = X[j]

            # No repetition
            # Eg if baseline_indices is null, it'll only run once as x_hat will stay the same over each iteration
            if x_hat.tolist() not in baseline_check_1:
                baseline_check_1.append(x_hat.tolist())
                match type:
                    case ShapleyValuesType.MARGINAL:
                        prob_x_hat = get_probability(
                            self.X_counter, x_hat, indices_baseline, lenX)
                    case ShapleyValuesType.CONDITIONAL:
                        prob_x_hat = conditional_prob(
                            self.X_counter, x_hat, indices, indices_baseline, lenX)
                    case ShapleyValuesType.CAUSAL:
                        prob_x_hat = 0.1  # Implementation with do intervetions

                x_hat = np.reshape(x_hat, (1, N))
                f1 = f1 + (self.model.predict_proba(x_hat)[0][1] * prob_x_hat if self.is_classification else self.model.predict(
                    x_hat) * prob_x_hat)

            # xi index will be given to baseline for f2
            x_hat_2[xi] = X[xi]
            if xi not in indices_baseline_2:
                indices_baseline_2.append(xi)

            # No repetition
            indices_2 = indices[:]
            indices_2.remove(xi)
            if x_hat_2.tolist() not in baseline_check_2:
                baseline_check_2.append(x_hat_2.tolist())
                match type:
                    case ShapleyValuesType.MARGINAL:
                        prob_x_hat_2 = get_probability(
                            self.X_counter, x_hat_2, indices_baseline_2, lenX)
                    case ShapleyValuesType.CONDITIONAL:
                        prob_x_hat_2 = conditional_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX)
                    case ShapleyValuesType.CAUSAL:
                        prob_x_hat_2 = 0.1 # Implementation with do intervetions

                x_hat_2 = np.reshape(x_hat_2, (1, N))
                f2 = f2 + (self.model.predict_proba(x_hat_2)[0][1] * prob_x_hat_2 if self.is_classification else self.model.predict(
                    x_hat_2) * prob_x_hat_2)

            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)
        absolute_diff = abs(f1 - f2)

        return absolute_diff, f1, f2
