# uncompyle6 version 3.7.4
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.8.5 (default, Jan 27 2021, 15:41:15) 
# [GCC 9.3.0]
# Embedded file name: /home/docker/CSN/bin/gpu_ccsn.py
# Compiled at: 2021-03-31 16:15:26
# Size of source mod 2**32: 19699 bytes
"""
python version of csn algorithm
https://github.com/wys8c764/CSN
"""
import os, argparse, logging, pandas as pd, numpy as np
from scipy import sparse
from scipy import stats
import sys
sys.path.append('.')
import useful_functions as uf

def condition_g(adjmc, kk=50, dlimit=5):
    """return the degree >5 and top kk ρ统计量的gene, b可优化参数>degree limit"""
    a = np.sum(adjmc, axis=1)
    id1 = np.argwhere(a >= dlimit)
    INDEX = np.argsort(a[id1.flatten()])[::-1]
    id2 = INDEX[0:kk]
    return id2.tolist()


def get_data(csv):
    if str(csv).endswith('csv'):
        df = pd.read_csv(csv, index_col=0, header=0)
    else:
        df = pd.read_csv(csv, index_col=0, header=0, sep='\t')
    return df


class SSN:
    """Construction of cell-specific networks
    模型构建过程用所有的样品数据，后续预测用整合有的大表做dm转化但仅输出少量样品(cells.list)的network和degree matrix
    在dblur features水平做矩阵融合
    The function performs the transformation from gene expression matrix to cell-specific network (csn).
    This is a groups style docs.
    
    Parameters:    
        `data`  Gene expression matrix, rows = genes, columns = cells
        Returns:   None
        Raises:  KeyError - raises an exception
    """

    def __init__(self, data, outdir='./', log=None):
        """
        default values when initialize. set log file
        """
        self.outdir = outdir
        self.tablename = data
        uf.create_dir(self.outdir)
        self.log = os.path.join(self.outdir, log) if log else os.path.join(self.outdir, '{}_{}.log'.format(os.path.basename(data), uf.now()))
        self.logger = uf.create_logger(self.log)
        self.logger.info('start reading data from {}, log file is {}'.format(data, self.log))
        df = get_data(data)
        self.data = df.loc[(df.sum(axis=1) != 0, df.sum(axis=0) != 0)]
        self.csn = None
        self.logger.info('finish reading data from {}'.format(data))

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
        else:
            if isinstance(cells, list):
                cells = cells
            else:
                if os.access(cells, os.R_OK):
                    cells = [cell.strip() for cell in open(cells).readlines()]
                else:
                    print('cells must be list or file with one column')
        return cells

    @uf.robust
    def csnet(self, cells=None, alpha=0.01, boxsize=0.1, edgeW=0, kk=0, dlimit=5, to_csv=0, average=1, *args, **kwargs):
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
        self.logger.info('start construction cell-specific network ')
        nr, nc = self.data.shape
        data = self.data
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

        self.logger.info('finish caculate the neighborhood of each gene for each cell')
        cells = self.get_cells(cells=cells)
        csn = dict()
        B = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
        p = -stats.norm.ppf(q=alpha, loc=0, scale=1)
        for k in cells:
            for j in B.columns:
                if average:
                    B.loc[:, j] = (data.loc[:, j] <= upper.loc[:, k]) & (data.loc[:, j] >= lower.loc[:, k]) & (data.loc[:, k] > 0)
                else:
                    B.loc[:, j] = (data.loc[:, j] <= upper.loc[:, k]) & (data.loc[:, j] >= lower.loc[:, k])

            B = B * 1
            a = np.matrix(B.sum(axis=1))
            csnk = (B.dot(B.T) * nc - a.T * a) / np.sqrt(np.multiply(a.T * a, (nc - a).T * (nc - a)) / (nc - 1) + np.spacing(1))
            csnlink = (csnk > p) * 1
            if csnlink.sum().sum() == 0:
                self.logger.info('no genes in Cell {} has a link'.format(k))
                continue
                if kk != 0:
                    id = condition_g(csnlink, kk=kk, dlimit=dlimit)
                    csnlink = pd.DataFrame(np.zeros([nr, nr])) if average else pd.DataFrame(np.ones([nr, nr]))
                    for m in range(kk):
                        B_z = B.iloc[id[m], :] * B
                        idc = np.argwhere(B.iloc[id[m], :] != 0).flatten()
                        B_z = B_z.iloc[:, idc]
                        r = B_z.shape[1]
                        a_z = np.mat(B_z.sum(axis=1))
                        c_z = B_z @ B_z.T
                        csnk1 = (c_z * r - a_z.T * a_z) / np.sqrt(np.multiply(a_z.T * a_z, (r - a_z).T * (r - a_z)) / (r - 1) + np.spacing(1))
                        csnlink1 = (csnk1 > p) * 1
                        csnlink = csnlink + csnlink1 if average else csnlink * csnlink1

                else:
                    kk = 1
                csnlink = csnlink / kk if average else csnlink
                csn[k] = csnlink
                if to_csv:
                    filename = os.path.join(self.outdir, 'cellnws', '{}.nw.csv'.format(k))
                    uf.create_dir(self.outdir + '/cellnws')
                    csn[k].to_csv(path_or_buf=filename)
                self.logger.info('Cell {} specific network is completed'.format(k))

        self.logger.info('Finished constructing all {} cell specific networks'.format(len(cells)))
        self.upper = upper
        self.lower = lower
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
        self.logger.info('Constructing network degree matrix ...')
        cells = self.get_cells(cells=cells)
        nr, nc = self.data.shape
        ndm = pd.DataFrame(np.zeros((nr, nc)), columns=data.columns, index=data.index)
        csn = self.csn
        celln = 0
        for k in cells:
            if k not in csn:
                self.logger.info('Cell {} has no network'.format(k))
                continue
                if nodeW:
                    ndm.loc[:, k] = csn[k].sum(axis=1) * data.loc[:, k]
                else:
                    ndm.loc[:, k] = csn[k].sum(axis=1)
                celln += 1
                self.logger.info('Network degree vector of cell {} is complete'.format(k))

        if normalize:
            self.logger.info('Normalizing network degree matrix ...')
            a = ndm.mean(axis=0)
            ndm = ndm.div(a + np.spacing(1), axis=1)
            ndm = np.log(1 + ndm)
        self.ndm = ndm
        if to_csv:
            filename = os.path.join(self.outdir, '{}.{}cells.nwdm.csv'.format(os.path.basename(self.tablename), celln))
            ndm.to_csv(path_or_buf=filename)
            self.logger.info('Finished network degree matrix, file: {}'.format(filename))

    @uf.robust
    def nfe(self, cells=None, to_csv=1, *args, **kwargs):
        data = self.data
        csn = self.csn
        self.logger.info('caculate network_flow_entropy ...')
        cells = self.get_cells(cells=cells)
        nr, nc = data.shape
        NFE = pd.DataFrame(np.zeros((nc, 1)), columns=['network_flow_entropy'], index=data.columns)
        celln = 0
        for k in cells:
            if k not in csn:
                self.logger.info('Cell {} has no network'.format(k))
                NFE.loc[k] = None
                continue
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
                NFE.loc[k]
                celln += 1
                self.logger.info('network_flow_entropy of cell {} is {}'.format(k, NFE.loc[k][0]))

        self.NFE = NFE
        if to_csv:
            filename = os.path.join(self.outdir, '{}.{}cells.NFE.csv'.format(os.path.basename(self.tablename), celln))
            NFE.to_csv(path_or_buf=filename)
            self.logger.info('Finished network_flow_entropy, output file: {}'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell-specific Network Constructed by Single-cell RNA Sequencing Data', usage='%(prog)s --help or -h for detailed help', epilog='\n        Example:\n           python  %(prog)s --data Otu_table.xls --netcells cells.list --dmcells cells.list\n                            --outdir ./ --logfile ntwk.log --dm2csv --net2csv\n                            --normalize\n        ', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', '-i', required=True, help='OTU table file (required)')
    parser.add_argument('--outdir', '-o', default='./', help="a path to store analysis results, default='./'")
    parser.add_argument('--logfile', default=None, help='the output log file name')
    parser.add_argument('--netcells', default=None, help='one column list of cell_names or caculate network for all cells [default]')
    parser.add_argument('--dmcells', default=None, help='one column list of cell_names or caculate degree matrix for all cells [default]')
    parser.add_argument('--edgeW', action='store_true', default=False, help='weight the network links with link strength [default:no weight]')
    parser.add_argument('--nodeW', action='store_true', default=False, help='weight the network nodes with abundance [default: no weight]')
    parser.add_argument('--net2csv', action='store_true', default=False, help='print out the net file for each cell [default: no print]')
    parser.add_argument('--dm2csv', action='store_true', default=False, help='print out the degree matrix file of the selected cells [default: print]')
    parser.add_argument('--kk', '-k', default=50, type=int, help='the number of top conditional gene')
    parser.add_argument('--alpha', '-a', default=0.01, type=float, help='alpha value cutoff')
    parser.add_argument('--boxsize', '-b', default=0.1, type=float, help='boxsize for nework construction')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the matrix')
    args = parser.parse_args()
    t_s = uf.datetime.now()
    csn = SSN(args.data, outdir=args.outdir, log=args.logfile)
    csnet = csn.csnet(cells=args.netcells, kk=args.kk, alpha=args.alpha, boxsize=args.boxsize, edgeW=args.edgeW, to_csv=args.net2csv)
    csndm = csn.csndm(cells=args.dmcells, normalize=args.normalize, to_csv=args.dm2csv, nodeW=args.nodeW)
    t_e = uf.datetime.now()
    usedtime = t_e - t_s
    csn.logger.info('Finish constructing network degree matrix, time used: {}'.format(usedtime))
# okay decompiling __pycache__/gpu_ccsn.cpython-35.pyc
