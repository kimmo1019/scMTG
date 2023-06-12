import yaml,time
import argparse
import numpy as np
from scMTG import scMulReg, scMTG, scDEC
from scMTG import util
import scanpy as sc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.load(f)
    model = scMTG(params)
    data_all = np.load('../data/%s/X.npz'%(params['dataset']))
    data = [data_all['arr_0'],data_all['arr_1'],data_all['arr_2']]
    model.train(data=data,n_iter=50000, batches_per_eval=500)
    
    # model = scDEC(params)
    # model.train()

    # model = scMulReg(params)
    # adata_rna = sc.read('../data/pbmc10k/adata_rna.h5')
    # adata_atac = sc.read('../data/pbmc10k/adata_atac.h5')
    # model.train(data=[adata_rna.X.toarray(),adata_atac.X.toarray()])
