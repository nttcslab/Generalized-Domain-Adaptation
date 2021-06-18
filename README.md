# Generalized Domain Adaptation (CVPR2021)
This is the official PyTorch impelementation of our paper ["Generalized Domain Adaptation"](https://arxiv.org/abs/2106.01656) (CVPR2021).

This code provides an implementation of our domain adaptation method for problems where the domain label is completely unknown for all samples.

Our method comprises two major steps: 
1. Run domain estimation first to estimate the domain labels for all samples.
2. Run classifier learning next to learn a domain-invariant classifier using the estimated domain labels.

The codes and their details for each step are stored in a sub-directory with the corresponding name.

We use the dataset provided at https://github.com/zjy526223908/BTDA.

Please replace `<sub-directory>/data/<dataset>/imgs_dummy` with the respective `imgs` directory of https://github.com/zjy526223908/BTDA.
