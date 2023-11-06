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


## Datasets

Please Contact us (raojh6@mail2.sysu.edu.cn) to obtain the Data (from DrugBank and DGIdb) and Splits.

### Statistic of DGI Dataset
|Dataset|DrugBank|DGIdb|
|:-:|:-:|:-:|
|#Drug|425|1185|
|#Gene|11284|1164|
|#Interactions|80924|11266|
|Interaction type|2|14|

## Usages
For training on DrugBank on the transductive scenario:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DrugBank --testing --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
```


For training on DGIdb on the inductive scenario:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DGIdb --testing --mode inductive --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
```

More parameters could be found by:
```
python main.py -h
```

## Reference
If you find the code useful, please cite our paper.
```
@inproceedings{cosmig,
  title     = {Communicative Subgraph Representation Learning for Multi-Relational Inductive Drug-Gene Interaction Prediction},
  author    = {Rao, Jiahua and Zheng, Shuangjia and Mai, Sijie and Yang, Yuedong},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {3919--3925},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/544},
  url       = {https://doi.org/10.24963/ijcai.2022/544},
}
```

## Contact
Jiahua Rao (raojh6@mail2.sysu.edu.cn) and Yuedong Yang (yangyd25@mail.sysu.edu.cn)
