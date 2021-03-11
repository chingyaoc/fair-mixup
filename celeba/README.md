# Celeb-A Experiment

## Dataset
Place the adult dataset under directory './celeba' (adult.data and adult.test) and run 'data_processing.py' to process the dataset. 

## Run
Run the experiment for DP:
```
python main_dp.py --method mixup/mixup_manifold/GapReg/erm --lam 20
```

Run the experiment for EO:
```
python main_eo.py --method mixup/mixup_manifold/GapReg/erm --lam 20
```
