#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yangfenglong
# @Date: 2019-05-20

"""
python version of csn algorithm
https://github.com/wys8c764/CSN
"""
    
import os
import argparse
import logging

import pandas as pd
import numpy as np
from scipy import sparse
from scipy import stats


import sys
sys.path.append('.') #表示导入当前文件的目录到搜索路径中
import useful_functions as uf

def condition_g(adjmc,kk=50,dlimit=5):
    '''return the degree >5 and top kk ρ统计量的gene, b可优化参数>degree limit'''
    a = np.sum(adjmc, axis=1) 
    b = np.sum(adjmc!=0, axis=1)
    id1 = np.argwhere(b >= dlimit)
    INDEX = np.argsort(-a[id1.flatten()])
    id2 = INDEX[0:kk]
    id = id1[id2].flatten()
    return id

def get_data(csv):
    if str(csv).endswith("csv"):
        df = pd.read_csv(csv,index_col=0, header=0)
    else:
        df = pd.read_csv(csv, index_col=0, header=0, sep='\t')
    return df

class SSN:
    """Construction of cell-specific networks
    模型构建过程用所有的样品数据，后续预测用整合有的大表做dm转化但仅输出少量样品cells.list的network和degree matrix
    在dblur features水平做矩阵融合
    The function performs the transformation from gene expression matrix to cell-specific network (csn).
    This is a groups style docs.

        Parameters:    
            `data`  Gene expression matrix, rows = genes, columns = cells
        Returns:   None
        Raises:  KeyError - raises an exception
    """
    
    def __init__(self, data, outdir="./", log=None): 
        """
        default values when initialize. set log file
        """
        self.outdir = outdir
        self.tablename=data
        uf.create_dir(self.outdir)
        self.log = os.path.join(self.outdir, log) if log else \
                   os.path.join( self.outdir,"{}_{}.log".format(os.path.basename(data),uf.now()) )
        self.logger = uf.create_logger(self.log)
        self.logger.info("start reading data from {}, log file is {}".format(data, self.log))
        self.data = get_data(data) 
        self.data = self.data.loc[self.data.sum(axis=1)!=0 , self.data.sum(axis=0)!=0] 
        self.csn = None
        # Gene expression matrix (TPM/FPKM/RPKM/count), rows = genes, columns = cells or OTU table
        self.logger.info("finish reading data from {}".format(data))
        
    @uf.robust
    def get_cells(self, cells=None):
        """
        Get cells in list format
        
            Parameters:
                file cells.list
            Returns: 
                cells in list format
            Raises:  
                KeyError - raises an exception
        """
        if not cells:
            cells = list(self.data.columns)
        elif isinstance(cells, list):
            cells = cells
        elif os.access(cells, os.R_OK):
            cells = [cell.strip() for cell in open(cells).readlines()]   
        else:
            print("cells must be list or file with one column")
        return cells
           
    @uf.robust
    def csnet(self, cells=None, alpha=0.01, boxsize=0.1, edgeW=0, kk=0, to_csv=0, *args, **kwargs):
        """
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
                `kk`    the number of conditional gene. when kk=0, the method is CSN

            Returns: 
                csnet dict 
            Raises:  
                KeyError - raises an exception  
            Notes:
                Too many cells or genes may lead to out of memory.
        """
        self.logger.info("start construction cell-specific network ")
        nr,nc=self.data.shape
        data = self.data
 
        #学习 dataframe 和array python的矩阵运算。
        #np index start from 0 
        #每个new cell都要和原来所有的细胞一起计算lower upper边界矩阵，都要排序每个基因来计算。
        #如果数据库足够大，可以就用原来的边界矩阵，重新换算出upper和lower矩阵。带入new cell的基因表达数据就可以。
        
        #Define the neighborhood of each plot 确定box左右边界两个matrix low-up 
        upper=pd.DataFrame(np.zeros((nr,nc)),columns=data.columns,index=data.index)
        lower=pd.DataFrame(np.zeros((nr,nc)),columns=data.columns,index=data.index)
        for i in range(nr):
            sort_gi=data.iloc[i,:].sort_values(axis = 0,ascending = True)
            s1 = sort_gi.values
            s2 = sort_gi.index
            n1 = sum(np.sign(s1))
            n0 = nc-n1  #the number of 0
            h = round(boxsize/np.sqrt(n1)) #方框半径
            k=0
            while k < nc:
                s=0 
                while k+s+1<nc and s1[k+s+1]==s1[k]:
                    #如果下一个cell的genei表达值一样就跳过，统一赋值
                    s = s+1 
                if s >= h:
                    upper.loc[data.index[i],s2[range(k,k+s+1)]] = data.loc[data.index[i],s2[k]] 
                    lower.loc[data.index[i],s2[range(k,k+s+1)]] = data.loc[data.index[i],s2[k]]
                else:
                    upper.loc[data.index[i],s2[range(k,k+s+1)]] = data.loc[data.index[i], s2[int(min(nc-1,k+s+h))]]
                    lower.loc[data.index[i],s2[range(k,k+s+1)]] = data.loc[data.index[i], s2[int(max(n0*(n0>h), k-h))]]
                k = k+s+1
        self.logger.info("finish caculate the neighborhood of each gene for each cell")  
       
        # Construction of CSN 
        # Construction of cell-specific networks for each cell 后续可用网络图论来研究样品间差异
        cells = self.get_cells(cells=cells)
        csn = dict()
        #dict.fromkeys(cells)
        
        B = pd.DataFrame(np.zeros((nr,nc)),columns=data.columns,index=data.index) 
        #每个cell一个B_matrix，基因在box内都是1，不在是0
        p = -stats.norm.ppf(q=alpha,loc=0,scale=1) 
        #0.99置信度下的统计量阈值 Percent point function (inverse of cdf — percentiles). 
        
        """
        cell k has gene j, and the expression value is among lower and upper 
        gene expresion value 决定了upper lower 边界大小，把这个存成dict，供new cell 使用，基因丰度最相近的box作为newcell gene的box边界，根据gene大小对应的up lower键值来，计算new cells 的B矩阵
        根据基因丰度来对cell进行聚类，丰度模式相似的样品，度矩阵也默认相似。
        怎么根据数据库，快速产生度矩阵，然后进行分类预测。数据库里已经有度矩阵，和预测模型，根据new cell的度向量进行判别
        new cell 的度向量，csn快速计算问题，后面再研究
        运算符重载的使用有点不合逻辑：* 不能按元素操作，但 / 确是如此。
        # 虽然我们可以使用sort, 但是sort是全局排序
        # 如果数组非常大, 我们只希望选择最小的10个元素, 直接通过np.partition(arr, 9)即可
        # 然后如果排序的话, 只对这选出来的10个元素排序即可, 而无需对整个大数组进行排序

        """        
        for k in cells: #to update for multi process run
            for j in B.columns:
                B.loc[:,j] = (data.loc[:,j] <= upper.loc[:,k]).astype('int') \
                           * (data.loc[:,j] >= lower.loc[:,k]).astype('int') \
                           * [(i>0)*1 for i in data.loc[:,k]]
   
            a = np.mat(B.sum(axis=1)) #gene丰度和，一行
            # matlab向量是列向量，需要a*a' python 向量a是matrix的一行，需要a.T*a
            csnk = (B.dot(B.T)*nc - a.T*a) \
                 / np.sqrt( (a.T * a) * ((nc-a).T*(nc-a)) / (nc - 1) + np.spacing(1) ) #cell-k's gene-gene network
            # 这个有没有办法利用已经构建好的network？没有的话就不能快速计算新样本的network
            np.fill_diagonal(np.asmatrix(csnk),0) 
            csnlink = (csnk > p)*1    # 1: link exsist, 0: no link 
            
            if csnlink.sum().sum() == 0:  #all genes has no link with each other
                self.logger.info("no genes in Cell {} has a link".format(k))
                continue


            if kk != 0: 
                id = condition_g(csnlink,kk) # 选出top kk ρ越大dependent越高的kk个gene的index 
                adjmc = np.asmatrix(np.zeros([n1,n1]))  # %开始计算kk个c-csn
                for m in range(kk): #从最相关的gene z开始算 c-csn
                    B_z =B*B.iloc[id[m],:]
                    # z存在的情况下B(id(m),:)，每个gene的box-matrix，z和x-y gene都存在为1（三维空间中的盒子），其他情况为0
                    idc = np.argwhere(B.iloc[id[m],:]!=0).flatten() #z不为0的cell index，存在zgene的那些cells
                    B_z = B_z.iloc[:,idc] # z存在的那些cell组成的子矩阵 降维？
                    r = B_z.shape[1] # r: cell numbers
                    a_z = np.mat(B_z.sum(axis=1)) # gene degree sum 在box里的数量Nxy
                    c_z = B_z@B_z.T 

                    csnk1 = (c_z*r - a_z.T*a_z) \
                            / np.sqrt( (a_z.T * a_z) * ((r-a_z).T*(r-a_z)) / (r - 1) + np.spacing(1) ) 
                    np.fill_diagonal(np.asmatrix(csnk1),0) 
                    csnlink1 = (csnk1 > p)*1 
                    csnlink = csnlink + csnlink1               
            else: 
                kk=1 

            csnlink = csnlink/kk

            # if edgeW:
            #     csn[k]=np.multiply(csnk, csnlink)
            # else:
            #     csn[k]= csnlink
            csn[k]= csnlink            
            
            if to_csv:                
                filename = os.path.join(self.outdir,"cellnws", "{}.nw.csv".format(k))
                uf.create_dir(self.outdir + "/cellnws")
                csn[k].to_csv(path_or_buf = filename)
            self.logger.info('Cell {} specific network is completed'.format(k))
        self.logger.info('Finished all {} cell specific networks'.format(len(cells)))
        self.csn = csn
    
    @uf.robust
    def csndm(self, cells=None, normalize=1, to_csv=1, nodeW=0, *args, **kwargs):
        """Construction of network degree matrix
        The function performs the transformation from gene expression matrix to network degree matrix (ndm).
        
            Parameters:        
                `data`     Gene expression matrix (TPM/RPKM/FPKM/count), rows = genes, columns = cells. otu_even.table    
                `alpha`    Significant level (eg. 0.001, 0.01, 0.05 ...), Default = 0.01
                `boxsize`  Size of neighborhood, Default = 0.1 (nx(k) = ny(k) = 0.1*n)
                `normalize`1  result is normalized (Default); 
                           0  result is not normalized
            Note:
                If gene expression matrix is sparse, use the sparse matrix will accelerate the calculation and reduce memory footprint
                data = sparse(data); upper = sparse(upper); lower = sparse(lower);
                可用于机器学习，样品分类预测等
                只输出指定 cells 的degree matrix ，不指定就输出所有cell的全部gene's dm
        """
        data = self.data
        self.logger.info("Constructing network degree matrix ...")
        cells = self.get_cells(cells=cells)
        ndm = pd.DataFrame()
        csn = self.csn
        #degree矩阵要不要加权丰度矩阵？可以设置权重比例的大小 nodeW*percent? or np.log1p(nodeW) 设计优化参数
        #csn[k]=sparse.coo_matrix((csnk > 0)*1) #存稀疏矩阵，不能diag运算，稀疏矩阵另一套算法
        celln=0
        for k in cells:
            if not k in csn:
                self.logger.info("Cell {} has no network yet".format(k))
                continue
            if nodeW:
                ndm.loc[:,k] = (csn[k].sum(axis=1) - np.diag(csn[k])) \
                             * data.loc[:,k] #gene_i和其他所有基因的连接数 - 自连的情况
            else:
                ndm.loc[:,k] = csn[k].sum(axis=1) - np.diag(csn[k])
            celln += 1
            self.logger.info("Network degree vector of cell {} is complete".format(k))
 
        if normalize:
            # normalization of network degree matrix
            self.logger.info("Normalizing network degree matrix ...")
            a = ndm.mean(axis=0)
            ndm = ndm.div(a+np.spacing(1) ,axis=1)
            ndm = np.log(1+ndm)




            # dsum = ndm.sum(axis=0)
            # ndm = ndm.div(dsum,axis=1) \
            #     * dsum.mean()**2/2000
            
            
            #按照cell平均的度大小来均一化 ,axis=1按列索引，按行广播
            #2000是个常数，可以优化，数值模拟出来的值。根据每个cell的平均基因数和度的关系，不同的肠道或其他样本类型算出一个特定的参数来bsxfun
                                                                                                             
        if to_csv:
            filename = os.path.join(self.outdir, 
                       "{}.{}cells.nwdm.csv".format(os.path.basename(self.tablename),celln))
            ndm.to_csv(path_or_buf = filename)
            self.logger.info("Finish generate network degree matrix file {}".format(filename))

if __name__ == '__main__':                                                    
    """https://www.jianshu.com/p/516f009c0875 nextflow流程後續加入scikitlearn 機器學習"""
    parser = argparse.ArgumentParser(
        description="Cell-specific Network Constructed by Single-cell RNA Sequencing Data",
        usage="%(prog)s --help or -h for detailed help",
        epilog="""
        Example:
           python  %(prog)s --data Otu_table.xls --netcells cells.list --dmcells cells.list
                            --outdir ./ --logfile ntwk.log --dm2csv --net2csv
                            --normalize
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data', '-i', required=True, help='OTU table file (required)') 
    parser.add_argument('--outdir', '-o', default='./', help='a path to store analysis results, default=\'./\'')
    parser.add_argument('--logfile', default=None, help='the output log file name')    
    parser.add_argument('--netcells', default=None, help='one column list of cell_names or caculate network for all cells [default]')
    parser.add_argument('--dmcells', default=None, help='one column list of cell_names or caculate degree matrix for all cells [default]')
    parser.add_argument('--edgeW', action='store_true', default=False, help='weight the network links with link strength [default:no weight]') 
    parser.add_argument('--nodeW', action='store_true', default=False, help='weight the network nodes with abundance [default: no weight]')                                                
    parser.add_argument('--net2csv', action='store_true', default=False, help='print out the net file for each cell [default: no print]') 
    parser.add_argument('--dm2csv', action='store_true', default=False, help='print out the degree matrix file of the selected cells [default: print]')                                                                                                               
    parser.add_argument('--kk', '-k', default=50,  type=int, help='the number of top conditional gene') 
    parser.add_argument('--alpha', '-a', default=0.01,  type=float, help='alpha value cutoff') 
    parser.add_argument('--boxsize', '-b', default=0.1, type=float, help='boxsize for nework construction')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the matrix') 

    args=parser.parse_args()
    
    t_s = uf.datetime.now()    
    csn = SSN(args.data, outdir=args.outdir, log=args.logfile) 
    csnet = csn.csnet(cells=args.netcells, kk=args.kk,
                      alpha=args.alpha, boxsize=args.boxsize, 
                      edgeW=args.edgeW, to_csv=args.net2csv)                                                          
    csndm = csn.csndm(cells=args.dmcells, 
                      normalize=args.normalize, to_csv=args.dm2csv, nodeW=args.nodeW)
    t_e = uf.datetime.now()
                                                            
    usedtime = t_e - t_s
    
    csn.logger.info('Finish constructing network degree matrix, time used: {}'.format(usedtime))
