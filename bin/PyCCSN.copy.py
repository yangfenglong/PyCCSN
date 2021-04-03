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

def condition_g(adjmc, kk=50, dlimit=5):
    """return the top kk ρ-statistics genes' index with degree > dlimit"""
    a = np.sum(adjmc, axis=1)
    id1 = np.argwhere(a >= dlimit)
    INDEX = np.argsort(a[id1.flatten()])[::-1] #descend
    id = INDEX[0:kk]
    return id.tolist()


def get_data(csv):
    if str(csv).endswith('csv'):
        df = pd.read_csv(csv, index_col=0, header=0)
    else:
        df = pd.read_csv(csv, index_col=0, header=0, sep='\t')
    return df


def caculate_neighborhood(data, boxsize=0.1):
    """Define the neighborhood of each plot"""
    nr, nc = data.shape
    upper = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
    lower = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
    for i in range(nr):
        sort_gi = data.iloc[i, :].sort_values(axis=0, ascending=True)
        s1 = sort_gi.values
        s2 = sort_gi.index
        n1 = sum(np.sign(s1))
        n0 = nc - n1
        h = round(boxsize * np.sqrt(n1))
        k = 0
        while k < nc:
            s = 0
            while k + s + 1 < nc and s1[(k + s + 1)] == s1[k]:
                s = s + 1

            if s >= h:
                upper.loc[(data.index[i], s2[range(k, k + s + 1)])] = data.loc[(data.index[i], s2[k])]
                lower.loc[(data.index[i], s2[range(k, k + s + 1)])] = data.loc[(data.index[i], s2[k])]
            else:
                upper.loc[(data.index[i], s2[range(k, k + s + 1)])] = data.loc[(data.index[i], s2[int(min(nc - 1, k + s + h))])]
                lower.loc[(data.index[i], s2[range(k, k + s + 1)])] = data.loc[(data.index[i], s2[int(max(n0 * (n0 > h), k - h))])]
            k = k + s + 1
    return upper, lower
    


def condition_ndm_nfe(data, upper, lower, alpha=0.01, kk=1, 
                      dlimit=5, average=1, ndm=1, nfe=1, to_csv=1):
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

    # Construction of conditional cell-specific network (CCSN)
    B = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
    # one B matrix for each cell, the value in B matrix is {1: gene is in the box, 0: not}
    # 并行会不会都在修改B，导致混乱？测试
    p = -stats.norm.ppf(q=alpha, loc=0, scale=1) 
    # p: Statistical thresholds under confidence 0.99. 

    NDM = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
    NFE = pd.DataFrame(np.zeros((nc, 1)), columns=['network_flow_entropy'], index=data.columns)

    for k in cells: 
        # each cell has a specific B matrix记录其他cell，哪些在对应gene的box里面
        for j in B.columns:
            if average:
                B.loc[:, j] = (data.loc[:, j] <= upper.loc[:, k]) & (data.loc[:, j] >= lower.loc[:, k]) & (data.loc[:, k] > 0)
            else:
                B.loc[:, j] = (data.loc[:, j] <= upper.loc[:, k]) & (data.loc[:, j] >= lower.loc[:, k])

        B = B * 1
        a = np.matrix(B.sum(axis=1))
        adjmc = (B.dot(B.T) * nc - a.T * a) / np.sqrt(np.multiply(a.T * a, (nc - a).T * (nc - a)) / (nc - 1) + np.spacing(1))
        adjmc = (adjmc > p) * 1

        if adjmc.sum().sum() == 0:
            self.logger.info('no genes in Cell {} has a link'.format(k))
            continue
        if kk != 0:
            id = condition_g(adjmc, kk=kk, dlimit=dlimit)
            adjmc = pd.DataFrame(np.zeros([nr, nr])) if average else pd.DataFrame(np.ones([nr, nr]))
            for m in range(kk):
                B_z = B.iloc[id[m], :] * B
                idc = np.argwhere(B.iloc[id[m], :] != 0).flatten()
                B_z = B_z.iloc[:, idc]
                r = B_z.shape[1]
                a_z = np.mat(B_z.sum(axis=1))
                c_z = B_z @ B_z.T
                csnk1 = (c_z * r - a_z.T * a_z) / np.sqrt(np.multiply(a_z.T * a_z, (r - a_z).T * (r - a_z)) / (r - 1) + np.spacing(1))
                adjmc1 = (csnk1 > p) * 1
                adjmc = adjmc + adjmc1 if average else adjmc * adjmc1

        else:
            kk = 1
        adjmc = adjmc / kk if average else adjmc

        
        if ndm:
            # Construction of conditional network degree matrix (cndm)
            if nodeW:
                NDM.loc[:, k] = adjmc.sum(axis=1) * data.loc[:, k]
            else:
                NDM.loc[:, k] = adjmc.sum(axis=1)
        elif nfe:
            datak = np.mat(data.loc[:, k])
            P = np.multiply(datak.T * datak, np.mat(csn[k]))
            cc = P.sum(axis=1) != 0
            idc = np.array(cc)[:, 0]
            id = data.index[idc]
            x = data.loc[(id, k)]
            x_n = x / x.sum()
            P1 = P[[id]][:, id]
            P_n = P1 / P1.sum(axis=1)
            x_p = pd.DataFrame(P_n) * np.array(x_n).reshape(-1, 1)
            x_p[x_p == 0] = 1
            NFE.loc[k] = -np.sum(np.sum(x_p * np.log(x_p)))

    # Construction of conditional network degree matrix (cndm)
    if ndm:
        if normalize:
            self.logger.info('Normalizing network degree matrix ...')
            a = NDM.mean(axis=0)
            NDM = NDM.div(a + np.spacing(1), axis=1)
            NDM = np.log(1 + NDM)
        if to_csv:
            filename = os.path.join(self.outdir, '{}.{}cells.nwdm.csv'.format(os.path.basename(self.tablename), celln))
            NDM.to_csv(path_or_buf=filename)
            self.logger.info('Finished network degree matrix, file: {}'.format(filename))

    # Calculate network flow entropy (NFE)
    if nfe and to_csv:
        filename = os.path.join(self.outdir, '{}.{}cells.NFE.csv'.format(os.path.basename(self.tablename), celln))
        NFE.to_csv(path_or_buf=filename)
        self.logger.info('Finished network_flow_entropy, output file: {}'.format(filename))
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
    
    def __init__(self, datafile, outdir="./", log=None): 
        """
        default values when initialize. set log file
        """
        self.outdir = outdir
        self.tablename=datafile
        uf.create_dir(self.outdir)
        self.log = os.path.join(self.outdir, log) if log else \
                   os.path.join( self.outdir,"{}_{}.log".format(os.path.basename(datafile),uf.now()) )
        self.logger = uf.create_logger(self.log)
        self.logger.info("start reading data from {}, log file is {}".format(data, self.log))
        df = get_data(datafile) 
        self.data = df.loc[df.sum(axis=1)!=0 , df.sum(axis=0)!=0] 
        self.logger.info("finish reading data from {}".format(data))

    def ccsn(self, alpha=0.01, boxsize=0.1, kk=1, dlimit=5, average=1, ndm=1, nfe=1, to_csv=1):
        upper, lower = caculate_neighborhood(self.data, boxsize=0.1, logger=self.logger)
        self.NDM, self.NFE = condition_ndm_nfe(self.data, upper, lower, logger=self.logger,
                                    alpha=alpha, kk=kk, dlimit=dlimit, 
                                    average=average, #ccsn based on average of ccsns
                                    ndm=1, nfe=1, to_csv=1)                                    


if __name__ == '__main__':                                                    
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
    parser.add_argument('--nfe2csv', action='store_true', default=False, help='print out the net file for each cell [default: no print]') 
    parser.add_argument('--ndm2csv', action='store_true', default=False, help='print out the degree matrix file of the selected cells [default: print]')                                                                                                               
    parser.add_argument('--kk', '-k', default=50,  type=int, help='the number of top conditional gene') 
    parser.add_argument('--alpha', '-a', default=0.01,  type=float, help='alpha value cutoff') 
    parser.add_argument('--boxsize', '-b', default=0.1, type=float, help='boxsize for nework construction')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the matrix') 
    parser.add_argument('--nfe', action='store_true', default=False, help=' Construction of conditional network degree matrix') 
    parser.add_argument('--ndm', action='store_true', default=False, help='Calculate network flow entropy') 


    args=parser.parse_args()
    
    t_s = uf.datetime.now()   


    csn = CCSN(args.data, outdir=args.outdir, log=args.logfile) 
    if args.ndm or args.nfe:
        csnet = csn.csnet(cells=args.netcells, kk=args.kk,
                      alpha=args.alpha, boxsize=args.boxsize, 
                      edgeW=args.edgeW, to_csv=args.net2csv)                                                          
        csndm = csn.csndm(cells=args.dmcells, 
                      normalize=args.normalize, to_csv=args.dm2csv, nodeW=args.nodeW)
    t_e = uf.datetime.now()
                                                            
    usedtime = t_e - t_s
    
    csn.logger.info('Finish constructing network degree matrix, time used: {}'.format(usedtime))


    


        









