## Local Minima Shape for Artificial Neural Network
by Adrien Bertaud, Yanis Bendali and Danya Li

The minima accessible by neural networks are studied. The concept of non-uniformity and sharpness are introduced, which characterize the shape of a minimum and hence the feasibility of convergence for a given optimizer. In
particular, this work presents the influence of learning rate and batch size in local minima selection. Our emphasis will be SGD, GD, AdaGrad, and L-BFGS optimizers.


### Training and evaluating sharpness and non-uniformity, results are stored in resutls.csv.
```
python train_and_eval.py
```

### Plot results from data stored in results.csv.
```
python plot_results.py
```

### Run toy examples to illustrate the influence of learning rate on the shape of minima found.
```
python toy_examples.py
```

### Report
```
Report.pdf
```

### Feedbacks from teachers
```
Feedbacks.pdf
```

### Versions used
Code was developed with the following versions:
* **Python** 3.7.6
* **NumPy** 1.18.1
* **matplotlib** 3.3.2
* **torch** 1.5.0
* **torchvision** 0.6.0

