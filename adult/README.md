# Adult Experiment

## Dataset
Place the adult dataset under directory './adult' (adult.data and adult.test).
The data processing code is modified from [IBM/sensitive-subspace-robustness](https://github.com/IBM/sensitive-subspace-robustness).


## Run
Run the experiment for DP or EO:
```
python main.py --method mixup/GapReg/erm --mode dp/eo --lam 0.5
```

## Recommended Lambda
|          | DP | EO |
|----------|:---:|:---:|
|  mixup | [0.1 ~ 0.7] | [0.5 ~ 5] |
|  GapReg |[0.1 ~ 0.7] | [0.5 ~ 5] |
