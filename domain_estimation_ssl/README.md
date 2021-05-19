# Domain label estimation

## Package install

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
```

## Run program
### Training

```
$ python run.py --config=<yaml file>
```

### Estimate domain labels

```
$ python clustering.py --config=<yaml file>
```

Estimation result is output as `cluster_pca.csv` at the output directory in `record`.

## Acknowledgement
We implemented this program based on https://github.com/sthalles/SimCLR.