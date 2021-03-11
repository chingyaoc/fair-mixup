# Fair Mixup: Fairness via Interpolation

Training classifiers under fairness constraints such as group fairness, regularizes the disparities of predictions between the groups. Nevertheless, even though the constraints are satisfied during training, they might not generalize at evaluation time. To improve the generalizability of fair classifiers, we propose fair mixup, a new data augmentation strategy for imposing the fairness constraint. In particular, we show that fairness can be achieved by regularizing the models on paths of interpolated samples  between the groups. We use mixup, a powerful data augmentation strategy  to generate these interpolates. We analyze fair mixup and empirically show that it ensures a better generalization for both accuracy and fairness measurement in tabular, vision, and language benchmarks.

**Fair Mixup: Fairness via Interpolation** ICLR 2021 [[paper]](https://openreview.net/forum?id=DNl5s5BXeBn)
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
The code of fair-mixup for Adult and CelebA experiments can be founded in the correspoding folders.

