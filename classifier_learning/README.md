# Classifier Learning 

## Before run this program
Run domain estimation program first. Then locate the output of the domain estimation to `clustering` directory and rename it so that it can be loaded in this program.

`clustering/Office31/grid_3/webcam_amazon.csv` is the example of the output.

## Required package
+ PyTorch=1.1.0
+ pandas
+ scikit-learn
+ easydict
+ tensorboardX
+ tqdm
+ opencv

## Run Training

```
python main.py --config=<yaml file in config> --outname=<output name>
```

Result is output at `record/<output name>`.

## Evaluation

```
python assessment.py record/<output name>
```
or 
```
python assessment_inter.py record/<output name>
```

## Acknowledgement
We implemented this code based on https://github.com/CuthbertCai/pytorch_DANN
