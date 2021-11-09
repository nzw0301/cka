import numpy as np
from typing import Union, Callable
from numba import jit

SUPPORTED_KERNELS = {"linear", "rbf"}

class CKA:
    """
    the main implementation comes from
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    """
    def __init__(self, kernel: Union[str, Callable] = "linear", debiased: bool = False, threshold: float = 1.0):

        if not ((type(kernel) == str and kernel in SUPPORTED_KERNELS) or callable(kernel)):
            supported_kernel = "Callable, " + ", ".join(SUPPORTED_KERNELS)
            raise ValueError(f"Unsupported kernel. Kernel should be either {supported_kernel}")

        self.kernel = kernel
        self.debiased = debiased
        self.threshold = threshold

    @staticmethod
    @jit(nopython=True)
    def _gram_linear(x: np.ndarray) -> np.ndarray:
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        return x.dot(x.T)

    @staticmethod
    @jit(nopython=True)
    def _gram_rbf(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """Compute Gram (kernel) matrix for an RBF kernel.

        Args:
            x: A num_examples x num_features matrix of features.
            threshold: Fraction of median Euclidean distance to use as RBF kernel
            bandwidth. (This is the heuristic we use in the paper. There are other
            possible ways to set the bandwidth; we didn't try them.)

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)

        sq_distances = -2. * dot_products + np.expand_dims(sq_norms, axis=1) + np.expand_dims(sq_norms, axis=0)
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2. * threshold ** 2 * sq_median_distance))

    @staticmethod
    @jit(nopython=True)
    def _center_gram(gram: np.ndarray, unbiased: bool = False) -> np.ndarray:
        """Center a symmetric Gram matrix.

        This is equivalent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
            A symmetric matrix with centered columns and rows.
        """

        gram = gram.copy()
        n = gram.shape[0]

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).

            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0) / (n - 2)
            means -= np.sum(means) / (2. * (n - 1))
            gram -= np.expand_dims(means, axis=1)
            gram -= np.expand_dims(means, axis=0)
            np.fill_diagonal(gram, 0)
        else:
            means = np.sum(gram, 0) / n
            means -= np.sum(means) / n / 2.
            gram -= np.expand_dims(means, axis=1)
            gram -= np.expand_dims(means, axis=0)

        return gram
