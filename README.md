# Fair Mixup: Fairness via Interpolation

Training classifiers under fairness constraints such as group fairness, regularizes the disparities of predictions between the groups. Nevertheless, even though the constraints are satisfied during training, they might not generalize at evaluation time. To improve the generalizability of fair classifiers, we propose fair mixup, a new data augmentation strategy for imposing the fairness constraint. In particular, we show that fairness can be achieved by regularizing the models on paths of interpolated samples  between the groups. We use mixup, a powerful data augmentation strategy  to generate these interpolates. We analyze fair mixup and empirically show that it ensures a better generalization for both accuracy and fairness measurement in tabular, vision, and language benchmarks.

**Fair Mixup: Fairness via Interpolation** ICLR 2021 [[paper]](https://arxiv.org/abs/2103.06503)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), and
[Youssef Mroueh](https://ymroueh.me/)
<br/>

## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- aif360
- sklearn

## Implementation
The code for Adult and CelebA experiments can be found in the correspoding folders.

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{
chuang2021fair,
title={Fair Mixup: Fairness via Interpolation},
author={Ching-Yao Chuang and Youssef Mroueh},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=DNl5s5BXeBn}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).
