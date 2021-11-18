import numpy as np
import pytest

from cka import IncrementalCKA

num_iter = 3
num_samples = 128
num_features = 32


def test_incremental_cka_trivial() -> None:
    incremental_cka = IncrementalCKA(1, 1)

    for _ in range(num_iter):
        X = np.random.rand(num_samples, num_features)
        incremental_cka.increment_cka_score(index_feature_x=0, index_feature_y=0, features_x=X, features_y=X)
        assert pytest.approx(incremental_cka.cka()[0][0], 1e-5) == 1


def test_invariant_to_orthogonal_transformations() -> None:
    """
    This test case is based on
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=TGrluBcJGtbn
    """

    X = np.random.rand(num_samples, num_features)
    Y = np.random.rand(num_samples, num_features)

    incremental_cka = IncrementalCKA(1, 1)
    incremental_cka.increment_cka_score(index_feature_x=0, index_feature_y=0, features_x=X, features_y=Y)
    cka_score = incremental_cka.cka()[0][0]

    # cka with orthogonal transformed X
    incremental_cka_orthogonal = IncrementalCKA(1, 1)
    transform = np.random.rand(num_features, num_features)
    _, orthogonal_transform = np.linalg.eigh(transform.T.dot(transform))
    incremental_cka_orthogonal.increment_cka_score(index_feature_x=0, index_feature_y=0,
                                                   features_x=X.dot(orthogonal_transform), features_y=Y)
    orthogonal_cka_score = incremental_cka_orthogonal.cka()[0][0]
    assert pytest.approx(cka_score, 1e-5) == orthogonal_cka_score

    # not invariant to non-orthogonal transformed X
    incremental_cka_transform = IncrementalCKA(1, 1)
    incremental_cka_transform.increment_cka_score(index_feature_x=0, index_feature_y=0, features_x=X.dot(transform),
                                                  features_y=Y)
    transform_cka_score = incremental_cka_transform.cka()[0][0]
    assert pytest.approx(transform_cka_score, 1e-5) != orthogonal_cka_score


def test_invariant_to_isotropic_scaling() -> None:
    """
    This test case is based on
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=TGrluBcJGtbn
    """
    
    incremental_cka = IncrementalCKA(1, 1)

    for _ in range(num_iter):
        X = np.random.rand(num_samples, num_features)
        Y = X * 1.337
        incremental_cka.increment_cka_score(index_feature_x=0, index_feature_y=0, features_x=X, features_y=Y)
        assert pytest.approx(incremental_cka.cka()[0][0], 1e-5) == 1
