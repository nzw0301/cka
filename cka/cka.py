import numpy as np
from numba import jit


@staticmethod
@jit(nopython=True)
def gram_linear(x: np.ndarray) -> np.ndarray:
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


@staticmethod
@jit(nopython=True)
def gram_rbf(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
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

@staticmethod
@jit(nopython=True)
def _hsic(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def gram_cka(gram_x: np.ndarray, gram_y: np.ndarray, debiased: bool = False) -> float:
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.

    Returns:
        The value of CKA between X and Y.
    """
    for x in [gram_x, gram_y]:
        if not np.allclose(x, x.T):
            raise ValueError('Input must be a symmetric matrix.')

    gram_x = _center_gram(gram_x, unbiased=debiased)
    gram_y = _center_gram(gram_y, unbiased=debiased)


    return _hsic(gram_x, gram_y)

@staticmethod
@jit(nopython=True)
def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n: int):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))


@staticmethod
@jit(nopython=True)
def feature_space_linear_cka(features_x: np.ndarray, features_y: np.ndarray, biased: bool = False) -> np.ndarray:
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """

    n = features_x.shape[0]

    features_x = features_x - np.expand_dims(np.sum(features_x, 0), axis=0) / n
    features_y = features_y - np.expand_dims(np.sum(features_y, 0), axis=0) / n

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.sum(features_x ** 2, 1)
        sum_squared_rows_y = np.sum(features_y ** 2, 1)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)
