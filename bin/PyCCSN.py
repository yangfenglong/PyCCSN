"""
python version of c-csn algorithm
https://github.com/LinLi-0909/c-CSN
"""

import os, argparse, logging, pandas as pd, numpy as np
from scipy import sparse
from scipy import stats

import sys
sys.path.append('.')

import useful_functions as uf

from numba import cuda
from numba import jit
import numpy as np
import math
from time import time



def condition_g(adjmc, kk=50, dlimit=5):
    """return the top kk ρ-statistics genes' index with degree > dlimit"""
    a = np.sum(adjmc, axis=1).flatten()
    id_kk = np.argsort(a)[::-1][0:kk]
    id = id_kk[a[id_kk]>dlimit]
    return id

# 查找元素的一种方法是np.where(a==x)[0][0]，它既不优雅也不快速，因为要查找的项需要从开头遍历数组的所有元素。
# 更快的方式是通过Numba中的next((i[0] for i, v in np.ndenumerate(a) if v==x), -1)来加速。
# 一旦对数组进行排序，情况就会变得更好：v = np.searchsorted(a, x); return v if a[v]==x else -1的复杂度为O(log N)，确实非常快，但是首先需要O(N log N)的排序时间。
# function id = condition_g(adjmc,kk)
# a = sum(adjmc,2);
# b = sum(adjmc~=0,2);
# id1 = b(b>=5);
# [T,INDEX]=sort(a(id1),'descend');
# id2 = INDEX(1:kk);
# id = id1(id2);
# end


def get_data(csv):
    if str(csv).endswith('csv'):
        df = pd.read_csv(csv, index_col=0, header=0)
        cells = df.columns.to_list()
        genes = df.index.to_list()
        df = df.to_numpy()
    else:
        df = pd.read_csv(csv, index_col=0, header=0, sep='\t')
        cells = df.columns.to_list()
        genes = df.index.to_list()
        df = df.to_numpy()
    return df, cells, genes


def caculate_neighborhood(data, nr, nc, upper, lower, boxsize):
    """Define the neighborhood of each plot"""
    for i in range(nr):
        s2 = np.argsort(data[i,:])
        s1 = data[i,:][s2]
        n1 = int(np.sum(np.sign(s1)))
        n0 = nc - n1
        h = int(round(boxsize * np.sqrt(n1)))
        k = 0
        while k < nc:
            s = 0
            while k + s + 1 < nc and s1[k + s + 1] == s1[k]:
                s = s + 1
            if s >= h:
                upper[i, s2[range(k, k+s+1)]] = data[i, s2[k]]
                lower[i, s2[range(k, k+s+1)]] = data[i, s2[k]]
            else:
                upper[i, s2[range(k, k+s+1)]] = data[i, s2[min(nc-1, k+s+h)]]
                lower[i, s2[range(k, k+s+1)]] = data[i, s2[max(n0*(n0>h), k-h)]]
            k = k+s+1
        print('finish caculate the neighborhood of the {} gene'.format(i))
    print('finish caculate the neighborhood of each gene for each cell')
    return upper, lower

    
                


def condition_ndm_nfe(data, nr, nc, upper, lower, NDM, NFE, 
                        p, kk, dlimit, average, normalize, ndm, nfe):
    """
    fcndm = cndm(data, 0.1, 0.1, 1) for test
    Construct the CSN for sepecified cells
    
        Parameters:        
            `cells`   Construct the CSNs for all cells, set cells = None (Default) otherwise input cells.list
            `alpha`   Significant level (eg. 0.001, 0.01, 0.05 ...)
                        larger alpha leads to more edges, Default = 0.01
            `boxsize` Size of neighborhood, the value between 1 to 2 is recommended, Default = 0.1, 
            `edgeW`   1  edge is weighted (statistic pxy(x))
                        0  edge is not weighted (Default)
            `nodeW`   1  node is weighted (gene or otu abundance)
                        0  node is not wieghted (Default)
            `csn`     Cell-specific network, the kth CSN is in csn{k}
                        rows = genes, columns = genes
            `kk`      the number of conditional gene. when kk=0, the method is CSN
            `dlimit`  the min degree limitation of conditional genes.
            `average` whether use the average(adjmc + adjmc1) network or intersection(adjmc.*adjmc1) network.

        Returns: 
            csnet dict 
        Raises:  
            KeyError - raises an exception  
        Notes:
            Too many cells or genes may lead to out of memory.

    学习 dataframe 和array python的矩阵运算。
    np index start from 0 
    每个new cell都要和原来所有的细胞一起计算lower upper边界矩阵，都要排序每个基因来计算。
    如果数据库足够大，可以就用原来的边界矩阵，重新换算出upper和lower矩阵。带入new cell的基因表达数据就可以。
    """
    for k in range(nc): 
        # each cell has a specific B matrix记录其他cell，哪些在对应gene的box里面
        # Construction of conditional cell-specific network (CCSN)
        B = np.zeros((nr, nc))
        # one B matrix for each cell, the value in B matrix is {1: gene is in the box, 0: not}
        # 并行会不会都在修改B，导致混乱？测试

        for j in range(nc):
            # B[:, j] = (data[:, j] <= upper[:, k]) & (data[:, j] >= lower[:, k])
            if average:
                B[:, j] = (data[:, j] <= upper[:, k]) & (data[:, j] >= lower[:, k]) & (data[:, k] > 0)
            else:
                B[:, j] = (data[:, j] <= upper[:, k]) & (data[:, j] >= lower[:, k])

        a = B.sum(axis=1).reshape(-1,1)
        adjmc = (B.dot(B.T)*nc - a*a.T) / \
                np.sqrt( np.multiply(a*a.T, (nc-a)*(nc-a).T)/(nc-1)  +  np.spacing(1) )
        adjmc = (adjmc > p) * 1

        if kk != 0:
            id = [0,1]#condition_g(adjmc, kk=kk, dlimit=dlimit)
            adjmc = np.zeros((nr, nr)) if average else np.ones((nr, nr))
            for m in range(kk):
                B_z = B[id[m], :] * B #x|z, y|z
                idc = np.argwhere(B[id[m], :] != 0).flatten() # in the box
                B_z = B_z[:, idc]
                r = B_z.shape[1]
                a_z = B_z.sum(axis=1).reshape(-1,1)
                c_z = B_z @ B_z.T
                csnk1 = (c_z * r - a_z*a_z.T) / \
                        np.sqrt( np.multiply(a_z*a_z.T, (r-a_z).T * (r-a_z)) / (r-1) + np.spacing(1) )
                adjmc1 = (csnk1 > p) * 1
                adjmc = adjmc + adjmc1 if average else np.multiply(adjmc, adjmc1)
            adjmc = adjmc / kk if average else adjmc

        if nfe:
            datak = data[:, k].reshape(-1,1)
            P = np.multiply(datak * datak.T, adjmc)
            id = P.sum(axis=1) != 0
            x = data[id, k]
            x_n = x / x.sum()
            P1 = P[id][:,id]
            P_n = P1 / P1.sum(axis=1).reshape(-1,1)
            x_p = P_n * x_n.reshape(-1, 1)
            x_p[x_p == 0] = 1
            NFE[k] = -np.sum(np.multiply( x_p, np.log(x_p) ))
        if ndm:
            # Construction of conditional network degree matrix (cndm)
            NDM[:, k] = adjmc.sum(axis=1)
            # if nodeW:
            #     NDM[:, k] = adjmc.sum(axis=1) * data[:, k].reshape(-1,1)
            # else:
            #     NDM[:, k] = adjmc.sum(axis=1)
    # Construction of conditional network degree matrix (cndm)
    if ndm and normalize:
        print('Normalizing network degree matrix ...')
        a = NDM.mean(axis=0)
        NDM = NDM / (a + np.spacing(1)).reshape(1,-1)
        NDM = np.log(1 + NDM)
        print('finish construct network degree matrix ...')

    return NDM, NFE
        


class CCSN:
    """Construction of cell-specific networks
    模型构建过程用所有的样品数据，后续预测用整合有的大表做dm转化但仅输出少量样品(cells.list)的network和degree matrix
    在dblur features水平做矩阵融合
    The function performs the transformation from gene expression matrix to cell-specific network (csn).
    This is a groups style docs.

        Parameters:    
            `data`  Gene expression matrix, rows = genes, columns = cells      \
            # Gene expression matrix (TPM/FPKM/RPKM/count), rows = genes, columns = cells or OTU table

        Returns:   None
        Raises:  KeyError - raises an exception
    """
    
    def __init__(self, datafile, outdir="./"): 
        """
        default values when initialize. set log file
        """
        self.outdir = outdir
        self.tablename=datafile
        uf.create_dir(self.outdir)
        print("Reading data from {}".format(datafile))
        df, self.cells, self.genes = get_data(datafile)

        self.data = df[df.sum(axis=1)!=0][:, df.sum(axis=0)!=0] 
        print("finish reading data from {}".format(datafile))

    def ccsn(self, alpha=0.01, boxsize=0.1, kk=1, dlimit=5, 
                    average=1, normalize=1,ndm=1, nfe=1):
        nr, nc = self.data.shape
        upper = np.zeros((nr, nc))
        lower = np.zeros((nr, nc))
        upper, lower = caculate_neighborhood(self.data, nr, nc, upper, lower, boxsize)

        p = -stats.norm.ppf(q=alpha, loc=0, scale=1) 
                # p: Statistical thresholds under confidence 0.99 (alpha=0.01). 
        NDM = np.zeros((nr, nc))
        NFE = np.zeros((nc, 1))
        self.NDM, self.NFE= condition_ndm_nfe(self.data, nr, nc, upper, lower, NDM, NFE,
                            p, kk, dlimit, 
                            average, #ccsn based on average of ccsns
                            normalize,
                            ndm, nfe) 

        if ndm:
            filename = os.path.join(self.outdir, '{}.alpha{}_boxsize{}_top{}_degree{}.NDM.csv'.format(
                    os.path.basename(self.tablename),alpha, boxsize, kk, dlimit))
            NDM = pd.DataFrame(self.NDM,index=self.genes,columns=self.cells)
            NDM.to_csv(path_or_buf=filename)
            print('Finished network degree matrix, output file: {}'.format(filename))

        # Calculate network flow entropy (NFE)
        if nfe:
            filename = os.path.join(self.outdir, '{}.alpha{}_boxsize{}_top{}_degree{}.NFE.csv'.format(
                    os.path.basename(self.tablename),alpha, boxsize, kk, dlimit))
            NFE = pd.DataFrame(NFE,index=self.cells,columns=['network_flow_entropy'])
            NFE.to_csv(path_or_buf=filename)
            print('Finished network_flow_entropy, output file: {}'.format(filename))
                                       


if __name__ == '__main__':                                                    
    parser = argparse.ArgumentParser(
        description="Cell-specific Network Constructed by Single-cell RNA Sequencing Data",
        usage="%(prog)s --help or -h for detailed help",
        epilog="""
        Example:
           python  %(prog)s --data Otu_table.xls --outdir ./
                            --kk 5 --alpha 0.01 --boxsize 0.1 
                            --normalize --nfe --ndm
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data', '-i', required=True, help='OTU table file (required)') 
    parser.add_argument('--outdir', '-o', default='./', help='a path to store analysis results, default=\'./\'')
    parser.add_argument('--kk', '-k', default=50,  type=int, help='the number of top conditional gene') 
    parser.add_argument('--alpha', '-a', default=0.01,  type=float, help='alpha value cutoff') 
    parser.add_argument('--boxsize', '-b', default=0.1, type=float, help='boxsize for nework construction')
    parser.add_argument('--average', action='store_true', default=False, help='average(adjmc + adjmc1), alternative(adjmc.*adjmc1)') 
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the NDM matrix') 
    parser.add_argument('--ndm', action='store_true', default=False, help='Calculate network flow entropy') 
    parser.add_argument('--nfe', action='store_true', default=False, help=' Construction of conditional network degree matrix') 


    args=parser.parse_args()
    
    t_s = uf.datetime.now() 
    ccsn = CCSN(args.data, outdir=args.outdir) 
    if args.ndm or args.nfe:
        ccsn.ccsn(alpha=args.alpha, boxsize=args.boxsize, kk=args.kk, dlimit=args.dlimit, 
                average=args.average, normalize=args.normalize, ndm=args.ndm, nfe=args.nfe)                                                          
                                                           
    usedtime = t_e - t_s    
    print('Finish constructing network degree matrix, time used: {}'.format(usedtime))


    


        









