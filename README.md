# scFPN
Integrating Single-cell Multi-omics Data through Deep Embedded Fusion Representation

![alt text](https://github.com/biomed-AI/scFPN/blob/main/model_1.pdf "Illustration of scFPN")


## Requirements

Stable version: 

python==3.8.16 

pytorch==1.12.0

scanpy==1.9.3

anndata==0.8.0

episcanpy==0.4.0

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Also you can install the required packages follow there instructions (tested on a linux terminal):

`conda env create -f environment.yaml`


### Statistic of DGI Dataset
|Dataset|Chen et al.|Cao et al.|PBMC 10K-1|PBMC 10K-2|Ma te al.|GSE194122|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|#Cell|1047|1621|10412|11020|32231|69249|
|#CellType|4|3|19|12|22|23|
|#Gene|18666|113153|36601|36601|13428|23296|
|#Peak|136771|189603|116490|344592|108377|120010|
|Protocol|SNARE|sci-CAR|10x|10x|SHARE|10x|

## Usages
For training on GSE194122:
```
CUDA_VISIBLE_DEVICES=0 python train5.py -a GSE194122 -r default -z 32  --combine concat --gene-loss mse -o output
```


For training on other datasets:
```
python train5.py -a DATASET -r default -z 32  --combine concat --gene-loss mse -o output  --count-key X
```

More parameters could be found by:
```
python main.py -h
```

