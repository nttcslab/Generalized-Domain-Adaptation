# Generalized Domain Adaptation
This is the official impelementation of the paper "Generalized Domain Adaptation."

1. Run domain estimation program first and get estimated domain labels.
2. Run classifier learning program using the output of the first step.

The details of the programs are described in the respective directories.

In both programs, we use the dataset provided at https://github.com/zjy526223908/BTDA.

Please replace `<program directory>/data/<dataset>/imgs_dummy` with the respective `imgs` directory of https://github.com/zjy526223908/BTDA.