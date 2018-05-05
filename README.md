# jzl

Collection of tools and datasets for labwork in jZlab

### Datasets

#### Usage

Just import the dataset library, and call a load method

Default dataset cache is ~/.datasets Directory

```python
import jzl.datasets.emnist as emnist

# [image, class, writer]
(x_train,y_train,w_train),(x_test,y_test,w_test) = emnist.load_letters()
(x_train,y_train,w_train),(x_test,y_test,w_test) = emnist.load_digits()
```

*Supported Datasets*
- (MNIST)[http://yann.lecun.com/exdb/mnist/]
- (EMNIST)[https://www.nist.gov/srd/nist-special-database-19]
- (Kohn V1)[https://crcns.org/data-sets/vc/pvc-8/about]
