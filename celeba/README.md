# Celeb-A Experiment

## Dataset
Place the adult dataset under directory ```./celeba``` (adult.data and adult.test) and run ```data_processing.py``` to process the dataset. 

## Run
Run the experiment for DP:
```
python main_dp.py --method mixup/mixup_manifold/GapReg/erm --lam 20
```

Run the experiment for EO:
```
python main_eo.py --method mixup/mixup_manifold/GapReg/erm --lam 20
```

## Recommended Lambda
|          | DP | EO |
|----------|:---:|:---:|
|  mixup | [1, 10, 20] | [10, 20, 30] |
|  mixup_manifold | [25, 50] | [1, 10, 50] |
|  GapReg |[10, 25, 50] | [10, 20, 30] |
