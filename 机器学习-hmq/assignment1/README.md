## Assignment1

### Composition

This assignment is to implement KNN algorithm. It mainly contains three parts:
1. [model](./model.py)	This file contains the `KNN`  class ,  `split_rate`  function,  `k_fold`  function and  `score`  function. It is used to implement the algorithm and evaluate the accuracy of the algorithm.
2. [show](./show.py) This file contains the  `show`  function. It is used to visualize the result of the accuracy of the algorithm.
3. [main](./main.py) This file contains the  `__main__`  function. It is used run the algorithm and do some test.

### Requirements

- `numpy`
- `scikit-learn`

Actually, since I use pure Python the implement the algorithm. You do not need to build the environment. (I used sklearn just for the convenience to load the data.)But in that case, you should point out the iris data file(‘iris.data’ or ‘iris.csv’ .etc) path in [data_path.txt](./data_path.txt).

### How to run the code

If you want to run the script with Simple Cross-Validation, you can simply run this script in your shell.

```shell
python ./main.py
```

If you want to run the script with K-Fold Cross-Validation, you can run the script below in your shell.

```shell
python ./main.py k_fold 10(optional, it should be a positive integer)
```

