import numpy as np
import os
import scipy.sparse as sp 
import scipy.io
import copy
from scipy import pi
import sys
import pandas as pd
from os.path import join
import gzip
import scanpy as sc
from scipy.io import mmwrite,mmread
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler


class Multiome_loader_TI(object):
    def __init__(self, name='multiome',n_components=100,random_seed=1234,label_smooth=True,
                path = '/home/users/liuqiao/work/multiome'):
        #c:cell, g:gene, l:locus
        self.name = name
        self.pca_mat = np.load('%s/datasets/combine-v2/process/pca_combine.npy'%path).astype('float32')
        self.pca_rna_mat = self.pca_mat[:,:50]
        print(np.min(self.pca_rna_mat),np.max(self.pca_rna_mat))
        self.pca_rna_mat = MinMaxScaler().fit_transform(self.pca_rna_mat)
        print(np.min(self.pca_rna_mat),np.max(self.pca_rna_mat))
        self.label_time = np.load('%s/datasets/combine-v2/process/label_time.npy'%path)

        self.tsne_embeds = np.load('%s/datasets/combine-v2/process/tsne_embeds.npy'%path).astype('float32')
        self.tsne_embeds_d2 = self.tsne_embeds[:3006,:]
        self.tsne_embeds_d4 = self.tsne_embeds[3006:(3006+2847),:]
        self.tsne_embeds_d6 = self.tsne_embeds[-5766:,:]

        print(self.pca_rna_mat.shape,len(self.label_time))
        self.data_d2 = self.pca_rna_mat[self.label_time=='D2',:]
        self.data_d4 = self.pca_rna_mat[self.label_time=='D4',:]
        self.data_d6 = self.pca_rna_mat[self.label_time=='D6',:]
        print(self.data_d2.shape, self.data_d4.shape, self.data_d6.shape)

    def get_batch(self, batch_size):
        batch_d2_idx =  np.random.choice(len(self.data_d2), size=batch_size, replace=False)
        batch_d4_idx =  np.random.choice(len(self.data_d4), size=batch_size, replace=False)
        batch_d6_idx =  np.random.choice(len(self.data_d6), size=batch_size, replace=False)
        return self.data_d2[batch_d2_idx,:], self.data_d4[batch_d4_idx,:], self.data_d6[batch_d6_idx,:]

    def load_all(self):
        return self.data_d2, self.data_d4, self.data_d6

class ARC_TS_Sampler(object):
    def __init__(self,name='D2-1',n_components=50,scale=10000,filter_feat=True,filter_cell=False,random_seed=1234,mode=3, \
        min_rna_c=0,max_rna_c=None,min_atac_c=0,max_atac_c=None,start_t=0,prior_dim = 5):
        #c:cell, g:gene, l:locus
        self.name = name
        self.mode = mode
        self.min_rna_c = min_rna_c
        self.max_rna_c = max_rna_c
        self.min_atac_c = min_atac_c
        self.max_atac_c = max_atac_c
        self.start_t = start_t
        self.prior_dim = prior_dim
        #previous kmeans
        if os.path.exists('/home/users/liuqiao/work/multiome/datasets/pca_feats_v2.npz'):
            data = np.load('/home/users/liuqiao/work/multiome/datasets/pca_feats_v2.npz')
            nb_cells = [5400, 3408, 6897]
            self.pca_rna_mat,self.pca_atac_mat = data['arr_0'],data['arr_1']
            #self.pca_combine = np.hstack((self.pca_rna_mat, self.pca_atac_mat))
            self.pca_combine = self.pca_rna_mat
            self.pca_d2 = self.pca_combine[:nb_cells[0],:]
            self.pca_d4 = self.pca_combine[nb_cells[0]:(nb_cells[0]+nb_cells[1]),:]
            self.pca_d6 = self.pca_combine[-nb_cells[2]:,:]
            print(self.pca_d2.shape, self.pca_d4.shape, self.pca_d6.shape)
            self.embeds = MinMaxScaler().fit_transform(self.pca_combine.copy()).astype('float32')
            if start_t==0:
                self.prior = self.embeds[:nb_cells[0],:prior_dim]
                self.data = self.pca_d4
            elif start_t==1:
                self.prior = self.embeds[nb_cells[0]:(nb_cells[0]+nb_cells[1]),:prior_dim]
                self.data = self.pca_d6
            else:
                print('Wrong starting time point %d'%self.start_t)
                sys.exit()

    def get_batch(self, batch_size, sd = 1, weights = None):
        batch_data_idx =  np.random.choice(len(self.data), size=batch_size, replace=False)
        batch_prior_idx =  np.random.choice(len(self.prior), size=batch_size, replace=False)
        return self.data[batch_data_idx,:], self.prior[batch_prior_idx,:]

    def load_all(self):
        batch_prior_idx =  np.random.choice(len(self.prior), size=len(self.data), replace=True)
        return self.data, self.prior[batch_prior_idx,:]


class Gaussian_sampler(object):
    def __init__(self, N, mean, sd=1):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y

#sample continuous (Gaussian) and discrete (Catagory) latent variables together
class Mixture_sampler(object):
    def __init__(self, nb_classes, N, dim, sd, scale=1):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        self.sd = sd 
        self.scale = scale
        np.random.seed(1024)
        self.X_c = self.scale*np.random.normal(0, self.sd**2, (self.total_size,self.dim))
        #self.X_c = self.scale*np.random.uniform(-1, 1, (self.total_size,self.dim))
        self.label_idx = np.random.randint(low = 0 , high = self.nb_classes, size = self.total_size)
        self.X_d = np.eye(self.nb_classes, dtype='float32')[self.label_idx]
        self.X = np.hstack((self.X_c,self.X_d)).astype('float32')
    
    def train(self,batch_size,weights=None):
        X_batch_c = self.scale*np.random.normal(0, 1, (batch_size,self.dim)).astype('float32')
        #X_batch_c = self.scale*np.random.uniform(-1, 1, (batch_size,self.dim))
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        X_batch_d = np.eye(self.nb_classes,dtype='float32')[label_batch_idx]
        return X_batch_c, X_batch_d

    def load_all(self):
        return self.X_c, self.X_d

#sample continuous (Gaussian Mixture) and discrete (Catagory) latent variables together
class Mixture_sampler_v2(object):
    def __init__(self, nb_classes, N, dim, weights=None,sd=0.5):
        self.nb_classes = nb_classes
        self.total_size = N
        self.dim = dim
        np.random.seed(1024)
        if nb_classes<=dim:
            self.mean = np.random.uniform(-5,5,size =(nb_classes, dim))
            #self.mean = np.zeros((nb_classes,dim))
            #self.mean[:,:nb_classes] = np.eye(nb_classes)
        else:
            if dim==2:
                self.mean = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
            else:
                self.mean = np.zeros((nb_classes,dim))
                self.mean[:,:2] = np.array([(np.cos(2*np.pi*idx/float(self.nb_classes)),np.sin(2*np.pi*idx/float(self.nb_classes))) for idx in range(self.nb_classes)])
        self.cov = [sd**2*np.eye(dim) for item in range(nb_classes)]
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        self.Y = np.random.choice(self.nb_classes, size=N, replace=True, p=weights)
        self.X_c = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_d = np.eye(self.nb_classes)[self.Y]
        self.X = np.hstack((self.X_c,self.X_d))

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        if label:
            return self.X_c[indx, :], self.X_d[indx, :], self.Y[indx, :]
        else:
            return self.X_c[indx, :], self.X_d[indx, :]

    def get_batch(self,batch_size,weights=None):
        if weights is None:
            weights = np.ones(self.nb_classes, dtype=np.float64) / float(self.nb_classes)
        label_batch_idx =  np.random.choice(self.nb_classes, size=batch_size, replace=True, p=weights)
        return self.X_c[label_batch_idx, :], self.X_d[label_batch_idx, :]
    def predict_onepoint(self,array):#return component index with max likelyhood
        from scipy.stats import multivariate_normal
        assert len(array) == self.dim
        return np.argmax([multivariate_normal.pdf(array,self.mean[idx],self.cov[idx]) for idx in range(self.nb_classes)])

    def predict_multipoints(self,arrays):
        assert arrays.shape[-1] == self.dim
        return map(self.predict_onepoint,arrays)
    def load_all(self):
        return self.X_c, self.X_d, self.label_idx
    
def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1
    x -= np.max(x, axis = 1, keepdims = True)
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

#get a batch of data from previous 50 batches, add stochastic
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data

if __name__=="__main__":
    from umap import UMAP
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_style("ticks", {"xtick.major.size": 12, "ytick.major.size": 12})
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12) 
    matplotlib.rcParams.update({'font.size': 12})

    legend_params_ = {'loc': 'center left',
                    'bbox_to_anchor':(1.0, 0.45),
                    'fontsize': 20,
                    'ncol': 1,
                    'frameon': False,
                    'markerscale': 1.5
                    }
    s = ARC_TS_Sampler()
    data, label_one_hot = s.load_all()
    label= np.argmax(label_one_hot,axis = 1)
    data_embeds = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation',random_state=42).fit_transform(data)
    data_gen = np.load('/home/users/liuqiao/work/scMTG/src/results/20220921_223552/data_pre_99500.npy')
    data_gen_embeds = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation',random_state=42).fit_transform(data_gen)

    data_embeds_combine = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation',random_state=42).fit_transform(np.concatenate([data, data_gen],axis=0))
    np.savez('umap_embeds_v3', data_embeds, data_gen_embeds,data_embeds_combine)
    #label_combine = np.concatenate([label, label+3])

    # classes = np.unique(label)
    # colors = sns.husl_palette(len(classes), s=.8)
    # plt.figure(figsize=(8,8))
    # for i, c in enumerate(classes):
    #     plt.scatter(data_gen_embeds[label==c, 0], data_gen_embeds[label==c, 1], s=.5, color=colors[i], label=c)
    # plt.legend(**legend_params_)
    # plt.savefig('data_gen_embeds.png')
    #plt.show()

    # def plot_combine(cluster_id):
    #     classes = np.unique(label_combine)
    #     colors = sns.husl_palette(len(classes), s=.8)
    #     plt.figure(figsize=(8,8))
    #     c, i=cluster_id, cluster_id
    #     plt.scatter(data_embeds_combine[label_combine==c, 0], data_embeds_combine[label_combine==c, 1], s=.5, color=colors[i], label=c)
    #     c, i=cluster_id+3, cluster_id+3
    #     plt.scatter(data_embeds_combine[label_combine==c, 0], data_embeds_combine[label_combine==c, 1], s=.5, color=colors[i], label=c)
    #     plt.legend(**legend_params_)
    #     plt.savefig('together_%d.png'%cluster_id)
