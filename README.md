# Centered Kernel Alignment (CKA)

This repo contains the numpy implementations of batch CAK [1] and mini-batch CKA [2]. 

### Examples

The mini-batch CKA example is in [`example-minibatch-cka.ipynb`](./example-minibatch-cka.ipynb).


### Tests

Please run the following command.

```bash
pytest tests
```


---
### References

1. Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey  Hinton. [Similarity of Neural Network Representations Revisited](http://proceedings.mlr.press/v97/kornblith19a.html), In _ICML_, 2019. [project site](https://cka-similarity.github.io/).
2. Thao Nguyen, Maithra Raghu, Simon Kornblith. [Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth](https://openreview.net/forum?id=KJNcAkY8tY4), In _ICLR_, 2021.
