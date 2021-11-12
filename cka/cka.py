"""
The source code comes from
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
by Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey.

The modifications are as follows:

1. Apply `black` & PyCharm's formatter
2. Rename `center_gram` with `_center_gram`

Note that when I apply `numba.jit(nopython=True)` to all functions,
I could not make the code faster, so I decided to use the original code.
"""

import numpy as np


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
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
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def _center_gram(gram, unbiased=False):
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
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = _center_gram(gram_x, unbiased=debiased)
    gram_y = _center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty
            - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
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
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x ** 2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y ** 2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)


class IncrementalCKA:
    def __init__(self, num_layers_0: int, num_layers_1: int) -> None:
        """

        Args:
            num_layers_0: the number of layers in a network 0.
            num_layers_1: the number of layers in a network 1.
        """
        if num_layers_0 < 1:
            raise ValueError(f"`num_layers_0` should be positive: {num_layers_0}")

        if num_layers_1 < 1:
            raise ValueError(f"`num_layers_1` should be positive: {num_layers_1}")

        self.num_layers_0 = num_layers_0
        self.num_layers_1 = num_layers_1
        self._num_mini_batches = np.zeros((num_layers_0, num_layers_1), dtype=int)  # K in the paper
        self._kl = np.zeros((num_layers_0, num_layers_1))
        self._kk = np.zeros(num_layers_0)
        self._ll = np.zeros(num_layers_1)

    @staticmethod
    def _hsic(K: np.ndarray, L: np.ndarray) -> float:
        """
        Eq. 3

        Args:
            K: gram matrix. The shape shape is N \times N, where N is the number of samples in a mini-batches.
            L: gram matrix. The shape shape is N \times N.

        Note that the diagonal elements are zero.

        Returns: HSIC_1 value between K and L.

        """
        n = K.shape[0]
        first = np.trace(np.matmul(K, L))
        second = np.sum(K) * np.sum(L) / (n - 1) / (n - 2)
        third = 2. / (n - 2) * np.sum(K, axis=0).dot(np.sum(L, axis=0))
        denom = n * (n - 3)
        return 1. / denom * (first + second - third)

    def increment_cka_score(self, index_feature_x: int, index_feature_y: int, features_x: np.ndarray,
                            features_y: np.ndarray) -> None:
        """
        Update cka score between `index_feature_x` and `index_feature_y` using a mini-batch.
        This function computes HISC_1 values defined by Eq. 3 and stores them rather than returns them.

        Args:
            index_feature_x: the index for layer in a model 0.
            index_feature_y: the index for layer in a model 1.
            features_x: feature representation extracted by the model 0. The shape is N \times n_features.
            features_y: feature representation extracted by the model 1. The shape is N \times n_features.

            Note that
                - the numbers of samples of `features_x` and `features_y` should be the same.
                - the dimensionalities of `features_x` and `features_y` can differ.

        Returns:
            None.

        """
        assert 0 <= index_feature_x <= self.num_layers_0 - 1
        assert 0 <= index_feature_y <= self.num_layers_1 - 1

        self._num_mini_batches[index_feature_x, index_feature_y] += 1

        gram_x = gram_linear(features_x)
        gram_y = gram_linear(features_y)
        np.fill_diagonal(gram_x, 0)
        np.fill_diagonal(gram_y, 0)

        # since computing terms used in the denom of cka can be skipped,
        # we only compute them when either layer's index is 0.
        if index_feature_x == 0 or index_feature_y == 0:
            if index_feature_x == 0:
                self._ll[index_feature_y] += self._hsic(gram_y, gram_y)
            if index_feature_y == 0:
                self._kk[index_feature_x] += self._hsic(gram_x, gram_x)

        self._kl[index_feature_x, index_feature_y] += self._hsic(gram_x, gram_y)

    def cka(self) -> np.ndarray:
        """
        Compute mini-batch CKA defined by Eq. 2 in the original paper.

        Returns: `np.darray` whose element is cka. The shape is `num_layers_0` \times `num_layers_1`.

        """
        K = np.min(self._num_mini_batches)

        assert K == np.max(self._num_mini_batches)

        cka_score = np.zeros(self._kl.shape)
        for l0, kk in enumerate(self._kk):
            kk = np.sqrt(kk / K)
            for l1, ll in enumerate(self._ll):
                denom = kk * np.sqrt(ll / K)
                cka_score[l0, l1] = self._kl[l0, l1] / K / denom
        return cka_score
