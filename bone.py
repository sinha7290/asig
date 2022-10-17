
# coding: utf-8

# In[1]:
import cv2
import re
import numpy as np
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.axes_grid1 import SubplotDivider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.transforms import *
import PIL
import math
#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import seaborn as sns
import json
from sklearn.metrics import *
from scipy.stats import fisher_exact, ttest_ind
import scipy.stats
from pprint import pprint
import os
import pickle
import sys
sys.path.append("/booleanfs2/sahoo/Hegemon/")
import StepMiner as smn
import HegemonUtil as hu
import random
random.seed(20)

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

acolor = ["#00CC00", "#D8A03D","#EC008C",
        'cyan', "#B741DC", "#808285",
        'blue', 'black', 'green', 'red',
        'orange', 'brown', 'pink', 'purple']

def getLangelier26Gene():
    l1 = [[k] for k in getEntries("covid/langelier-26.txt", 0)]
    wt1 = [float(k) for k in getEntries("covid/langelier-26.txt", 1)]
    return wt1, l1

def getLangelier10Gene():
    l1 = [["PCSK5"], ["IL1R2"], ["IL1B"], ["IFI6"], ["WDR74"],
            ["FAM83A"], ["ADM"], ["IFI27"], ["KRT13"], ["DCUN1D3"]]
    wt1 = [0.033, -0.057, -0.048, 0.458, 0.116, 0.016,
            -0.079, 0.079, -0.009, -0.047]
    return wt1, l1

def getLangelier3Gene():
    l1 = [["IL1R2"], ["IL1B"], ["IFI6"]]
    wt1 = [-0.038, -0.056, 0.388]
    return wt1, l1

def getSViP():
    l1 = [readList("covid/iav-list-1.txt")[0:20]] # 20 gene signature
    wt1 = [1]
    return wt1, l1

def getViP():
    l1 = [readList("covid/list-2.txt")] # 166 gene signature
    wt1 = [1]
    return wt1, l1

def getKD13():
    l1 = [['CACNA1E', 'DDIAS', 'KLHL2', 'PYROXD2', 'SMOX', 'ZNF185',
        'LINC02035', 'CLIC3', 'S100P', 'IFI27', 'HS.553068', 'CD163', 'RTN1']]
    wt1 = [1]
    return wt1, l1

def getDS9():
    wt1, l1 = [1], [['DMGDH', 'SLC31A1', 'PNPO', 'GNE', 'IVD', 'DCAF11',
        'ALDH9A1', 'F11', 'ABAT']]
    return wt1, l1

def getMSigDB(gs):
    url = "https://www.gsea-msigdb.org/gsea/msigdb/download_geneset.jsp?geneSetName=" + gs + "&fileType=txt"
    df = pd.read_csv(url, sep="\t")
    df.columns.values[0] = 'ID'
    l1 = [list(df.ID[1:])]
    wt1 = [1]
    return wt1, l1

def getCls13a14a3():
    order = [13, 14, 3]
    wt1 = [-1, 1, 2]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    return wt1, l1

def getCls13():
    order = [13]
    wt1 = [-1]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    return wt1, l1

def getCls14a3():
    order = [14, 3]
    wt1 = [1, 2]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    return wt1, l1

def getCls14a3v2():
    l1 = [['RPS16', 'RPS21', 'RPL6', 'CCDC88A', 'RPL3', 'RPS15A', 'RPL14',
        'PCBP2', 'RPL23', 'RPL13', 'METTL7A'],
        ['CLEC10A', 'ANXA4', 'UBL3', 'RPS21', 'RPL6', 'RPS15A', 'ARL4C',
            'RPL14', 'HEXA', 'METTL7A', 'RPS16']]
    l1 = [['RPL24', 'NACA', 'RPS16', 'RPS21', 'RPL6', 'CCDC88A',
        'RPL3', 'RPS15A', 'EEF1B2', 'RPL14', 'PCBP2', 'RPL23',
        'RPL13', 'METTL7A', 'EEF2'],
        ['CLEC10A', 'INPP5A', 'NACA', 'ANXA4', 'UBL3', 'RPS21',
            'ITSN1', 'RPL24', 'RPS15A', 'ARL4C', 'RPL14', 'RPL6',
            'HEXA', 'METTL7A', 'EEF2', 'RPS16']]
    wt1 = [1, 2]
    return wt1, l1

def getCls13a14a3v2():
    wt1, l1 = getCls13()
    wt2, l2 = getCls14a3v2()
    return wt1+wt2, l1+l2

def asciiNorm(ah):
    if sys.version_info[0] >= 3:
        keys = list(ah.keys())
        for k in keys:
            ah[bytes(k, encoding='latin-1').decode('utf-8')] = ah[k]
    return ah

#https://pubmed.ncbi.nlm.nih.gov/29885947/
#https://pubmed.ncbi.nlm.nih.gov/28419265/
#https://pubmed.ncbi.nlm.nih.gov/32859206/
#https://pubmed.ncbi.nlm.nih.gov/32629654/
#https://pubmed.ncbi.nlm.nih.gov/29449546/
def sepsis11gene(ana):
    h = ana.h
    hhash = {}
    for i in range(len(h.headers)):
        hhash[h.headers[i]] = i
    #hdr = ["GSM1914807", "GSM1914808", "GSM1914809", "GSM1914810", "GSM1914811"]
    #order = [hhash[k] for k in hdr]

    l1 = [["CEACAM1", "ZDHHC19", "C9orf95", "GNA15", "BATF", "C3AR1"],
          ["KIAA1370", "TGFBI", "MTCH1", "RPGRIP1", "HLA-DPB1"]]
    geoM = []
    lenM = []
    for s in l1:
        expr = []
        genes = []
        for gn in s:
            idl1 = h.getIDs(gn, research=0)
            for id1 in idl1:
                e = h.getExprData(id1);
                v = np.array([float(e[i]) if e[i] != "" else 0 for i in h.aRange()])
                if len([i for i in h.aRange() if e[i] == ""]) > 0:
                    continue
                expr.append(v)
                genes.append(id1)
        res = []
        for i in h.aRange():
            v = [k[i-h.start] for k in expr]
            a = np.array(v)
            gm = a.prod()**(1.0/len(a))
            res.append(gm)
        geoM.append(res)
        lenM.append(len(expr))
    score = [geoM[0][i - h.start] - geoM[1][i - h.start] * lenM[1] / lenM[0]
             for i in h.aRange()]
    #print([score[k - h.start] for k in order])
    ana.f_ranks = score
    print(score)
    actual = [1 if ana.aval[i] >= 1 else 0 for i in h.aRange()]
    fpr, tpr, thrs = roc_curve(actual, score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    arr = [ana.f_ranks[i - ana.h.start] for i in ana.order]
    i1 = [ana.order[i] for i in np.argsort(arr)]
    index = np.array([i - ana.h.start for i in i1])
    ana.cval = np.array([[ana.aval[i] for i in i1]])
    ana.ranks = geoM
    ana.i1 = i1
    ana.index = index
    ana.otype = 1

def processGeneGroupsDf(ana, l1, wt1, debug = 0, fthr = None):
    ana.orderDataDf(l1, wt1); print("ROC-AUC", ana.getMetrics())
    actual = [1 if ana.aval[i] >= 1 else 0 for i in ana.i1]
    score = [ana.f_ranks[i - ana.start] for i in ana.i1]
    thr = hu.getThrData(score)
    nm = (np.max(ana.f_ranks) - np.min(ana.f_ranks))/16
    if fthr is None:
        fthr = thr[0]
    if fthr == "thr0":
        fthr = thr[0] - nm
    if fthr == "thr2":
        fthr = thr[0] + nm
    if fthr == "thr3":
        fthr = thr[0] + 3 * nm
    print(thr)
    print(nm, fthr)
    predicted = [1 if ana.f_ranks[i - ana.start] >= fthr else 0 for i in ana.i1]
    c_dict = {}
    for i in ana.order:
        c_dict[i] = ana.f_ranks[i - ana.start]
        c_dict[i] = 0
        if ana.f_ranks[i - ana.start] >= fthr:
            c_dict[i] = 1
    fpr, tpr, thrs = roc_curve(actual, score)
    roc_auc = auc(fpr, tpr)
    if debug == 1:
        print(actual)
        print(predicted)
        print(score)
        print(thr[0], thr, nm)
        print("ROC-AUC", roc_auc)
        print('Accuracy', accuracy_score(actual, predicted))
        print(classification_report(actual, predicted, target_names=ana.atypes))
    return c_dict, fpr, tpr, roc_auc    
    
def reactome(idlist):
    import requests
    reactomeURI = 'http://www.reactome.org/AnalysisService/identifiers/projection?pageSize=100&page=1';
    response = requests.post(reactomeURI, data = idlist, \
                headers = { "Content-Type": "text/plain",  "dataType" : "json" })
    obj = json.loads(response.text)
    df = pd.DataFrame()
    df['name'] = [p["name"] for  p in obj["pathways"]]
    df['pValue'] = [p["entities"]["pValue"] for  p in obj["pathways"]]
    df['fdr'] = [p["entities"]["fdr"] for  p in obj["pathways"]]
    return df

def getCode(p):
    if p <= 0:
        return '0'
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    if p <= 0.1:
        return '.'
    return ''

def printOLS(fm, df1):
    import statsmodels.formula.api as smf
    lm1 = smf.ols(formula=fm, data=df1).fit()
    print(lm1.summary())
    idx = lm1.params.index
    ci = lm1.conf_int()
    ci_1 = [ ci[0][i] for i in range(len(idx))]
    ci_2 = [ ci[1][i] for i in range(len(idx))]
    c_1 = [ getCode(p) for p in lm1.pvalues]
    df = pd.DataFrame({'Name': idx, 
	'coeff' : lm1.params, 'lower 0.95' : ci_1,
        'upper 0.95' : ci_2, 'pvalues' : lm1.pvalues, 'codes': c_1},
        columns=['Name', 'coeff', 'lower 0.95',
	    'upper 0.95', 'pvalues', 'codes'])
    for i in range(len(idx)):
        print('%s\t%.2f\t(%0.2f - %0.2f)\t%0.3f' % \
                (idx[i], lm1.params[i], ci[0][i], ci[1][i], lm1.pvalues[i]))
    print(df.to_string(formatters={'coeff':'{:,.2f}'.format,
        'lower 0.95':'{:,.2f}'.format, 'upper 0.95':'{:,.2f}'.format,
        'pvalues': '{:,.3f}'.format}))
    return df

def getBoolean(cfile, sthr, pthr, code):
    res = []
    with open(cfile, "r") as bFile:
        for ln in bFile:
            ll = ln.strip().split("\t")
            bs = [ [int(ll[i]) for i in range(2, 6)] ]
            bs += [ [int(ll[i]) for i in range(2, 6)] ]
            bs += [ [float(ll[i]) for i in range(6, 10)] ]
            bs += [ [float(ll[i]) for i in range(10, 14)] ]
            rel, stats = hu.getBooleanRelationType(bs, sthr, pthr)
            if rel == code:
                res.append(ll[1])
    return res

def processOne(ana, order, wt1, ax1, ax2, id1 = None, l1=None, violin=1):
    genes = []
    if (l1 is None):
        nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
        sel = 2
        genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    if id1 is None:
        params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
                'genes': genes, 'ax': ax1, 'acolor': acolor}
        ax = ana.printTitleBar(params)
        res = ana.getROCAUC()
        ax.text(len(ana.cval[0]), 4, res)
        if (violin == 1):
            params['ax'] = ax2
            params['vert'] = 0
            ax = ana.printViolin(None, params)
        else:
            ax = ana.densityPlot(ax2, acolor)
        return ana
    else:
        params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
                'genes': genes, 'acolor': acolor}
        ax = ana.printTitleBar(params)
        if (violin == 1):
            ax = ana.printViolin(None, {'vert':0})
        else:
            ax = ana.densityPlot()
    expr = ana.h.getExprData(id1)
    c = [acolor[ana.aval[i]] for i in ana.order]
    a = [ana.aval[i] for i in ana.order]
    x = [float(expr[i]) for i in ana.order]
    y = [ana.f_ranks[i - ana.h.start] for i in ana.order]
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['c'] = c
    df['a'] = a
    ax = df.plot('x', 'y', c=c, kind = 'scatter')
    ana.addAxes(ax)
    for i in range(len(ana.atypes)):
        cond1 = (df['a'] == i)
        if (sum(cond1) > 0):
            s1 = np.max(df[cond1]['y']) - np.min(df[cond1]['y'])
            s2 = np.max(df[cond1]['x']) - np.min(df[cond1]['x'])
            df.loc[cond1, 'y'] += (np.mean(df[cond1]['x']) - df.loc[cond1, 'x']) * (s1+1) / (s2+1)
            df.loc[cond1, 'x'] -= (np.mean(df[cond1]['y']) - df.loc[cond1, 'y']) * (s2+1) / (s1+1)
    ax = df.plot('x', 'y', c=c, kind = 'scatter')
    ana.addAxes(ax)
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression(normalize=True)
    linreg.fit(np.array(df['x']).reshape(-1, 1),df['y'])
    y_pred = linreg.predict(np.array(df['x']).reshape(-1, 1))
    df['y1'] = (df['y'] - y_pred)
    ax = df.plot('x', 'y1', c=c, kind = 'scatter')
    ana.addAxes(ax)
    ana.f_ranks = df['y1']
    ana.i1 = [ana.order[i] for i in np.argsort(ana.f_ranks)]
    ana.f_ranks = [0 for i in ana.h.aRange()]
    for i in range(len(ana.order)):
        ana.f_ranks[ana.order[i] - ana.h.start] = df['y1'][i]
    index = np.array([i - ana.h.start for i in ana.i1])
    ana.cval = np.array([[ana.aval[i] for i in ana.i1]])
    ana.data = np.array([ana.expr[i] for i in ana.ind_r])[:,index]

    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
                  'genes': genes, 'ax': ax1, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    if (violin == 1):
        params['ax'] = ax2
        params['vert'] = 0
        ax = ana.printViolin(None, params)
    else:
        ax = ana.densityPlot(ax2, acolor)
    
    return ana

def processWithNorm(ana, l1, wt1, id1, desc, ax):
    processOne(ana, None, wt1, None, None, id1, l1=l1)
    params = {'spaceAnn': len(ana.order)/len(ana.atypes),
              'tAnn': 1, 'widthAnn':1, 'acolor': acolor,
              'w': 5, 'h': 0.8, 'atypes': ana.atypes ,'cval': ana.cval, 'ax': ax}
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    ax.text(-1, 2, desc, horizontalalignment='right',
                verticalalignment='center')
    return ax

def processComp(ana, id1):
    fig = plt.figure(figsize=(5,5), dpi=100)
    n1 = 7
    axlist = []
    for i in range(n1):
        ax = plt.subplot2grid((n1, 1), (i, 0))
        axlist.extend([ax])

    order = [13]
    wt1 = [-1]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    processWithNorm(ana, l1, wt1, id1, "Cluster \\#13", axlist[0])

    order = [14, 3]
    wt1 = [1, 2]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    processWithNorm(ana, l1, wt1, id1, "Cluster \\#14-3", axlist[1])

    order = [13, 14, 3]
    wt1 = [-1, 1, 2]
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    processWithNorm(ana, l1, wt1, id1, "Cluster \\#13-14-3", axlist[2])

    l1 = getBecker()
    wt1 = [-1, 1]
    processWithNorm(ana, l1, wt1, id1, "PMID: 26302899", axlist[3])

    l1 = getBell2016()
    wt1 = [-1, -1, 1]
    processWithNorm(ana, l1, wt1, id1, "PMID: 26986567", axlist[4])

    l1 = getCoates()
    wt1 = [-1, 1]
    processWithNorm(ana, l1, wt1, id1, "PMID: 18199539", axlist[5])

    l1 = getMartinez()
    wt1 = [-1, 1]
    processWithNorm(ana, l1, wt1, id1, "PMID: 17082649", axlist[6])

    return fig

def processDensityWide(ana, l1, wt1, desc=None):
    fig = plt.figure(figsize=(10,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ana.orderData(l1, wt1)
    ax = ana.densityPlot(ax1, acolor)
    if desc is not None:
        ax.set_title(desc)
    return fig

def plotDensityBar(ana, desc=None):
    fig = plt.figure(figsize=(4,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': [], 'ax': ax1, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if desc is not None:
        ax.text(-1, 2, desc, horizontalalignment='right',
                    verticalalignment='center')
    ax = ana.densityPlot(ax2, acolor)
    return fig

def plotViolinBar(ana, desc=None):
    fig = plt.figure(figsize=(4,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': [], 'ax': ax1, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getROCAUC()
    ax.text(len(ana.cval[0]), 4, res)
    if desc is not None:
        ax.text(-1, 2, desc, horizontalalignment='right',
                    verticalalignment='center')
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
            'genes': [], 'ax': ax2, 'acolor': acolor, 'vert': 0}
    ax = ana.printViolin(None, params)
    return fig

def plotBarViolin(actual, atypes, score, source = None, desc = None):
    i1 = [i for i in np.argsort(score)]
    cval = np.array([[actual[i] for i in i1]])
    sns.set()
    sns.set_style("white")
    sns.set_style({'text.color': '.5',
            'xtick.color':'.5', 'ytick.color':'.5', 'axes.labelcolor': '.5'})
    sns.set_context("notebook")
    sns.set_palette([adj_light(c, 1.5, 1) for c in acolor])
    fig = plt.figure(figsize=(4,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    params = {'spaceAnn': len(actual)/len(atypes), 'tAnn': 1, 'widthAnn':1,
                      'genes': [], 'ax': ax1, 'acolor': acolor}
    ax = plotTitleBar(cval, atypes, params)
    if source is not None:
        ax.text(len(cval[0]), 0, source)
    import sklearn.metrics
    res = []
    for k in range(1, len(atypes)):
        a = [actual[i] for i in i1 if actual[i] == 0 or actual[i] == k]
        s = [score[i] for i in i1 if actual[i] == 0 or actual[i] == k]
        fpr, tpr, thrs = sklearn.metrics.roc_curve(a, s, pos_label=k)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        res += ["%.2f" % roc_auc]
    ax.text(len(cval[0]), 4, ",".join(res))
    if desc is not None:
        ax.text(-1, 2, desc, horizontalalignment='right',
                            verticalalignment='center')
    lval = [[] for i in atypes]
    for i in range(len(actual)):
        lval[actual[i]] += [score[i]]
    atypes = [str(atypes[i]) + "("+str(len(lval[i]))+")"
            for i in range(len(atypes))]
    params['ax'] = ax2
    params['vert'] = 0
    ax = plotViolin(lval, atypes, params)
    return fig

def processData(ana, l1, wt1, desc=None, violin=1):
    ana.orderData(l1, wt1)
    if (violin == 1):
        return plotViolinBar(ana, desc)
    return plotDensityBar(ana, desc)

def processDataMm(ana, l1, wt1, desc=None, violin=1):
    genes = []
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    if (violin == 1):
        return plotViolinBar(ana, desc)
    return plotDensityBar(ana, desc)

def processDataVnorm(ana, id1 = None):
    fig = plt.figure(figsize=(5,9), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((8, 1), (0, 0))
    ax2 = plt.subplot2grid((8, 1), (1, 0), rowspan=3)
    ax3 = plt.subplot2grid((8, 1), (4, 0))
    ax4 = plt.subplot2grid((8, 1), (5, 0), rowspan=3)
    order = [13]
    wt1 = [-1]
    processOne(ana, order, wt1, ax1, ax2, id1)
    order = [14, 3]
    wt1 = [1, 2]
    processOne(ana, order, wt1, ax3, ax4, id1)
    return fig

def processDataHnorm(ana, id1 = None):
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 3), (0, 0))
    ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax3 = plt.subplot2grid((4, 3), (0, 1))
    ax4 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    ax5 = plt.subplot2grid((4, 3), (0, 2))
    ax6 = plt.subplot2grid((4, 3), (1, 2), rowspan=3)
    order = [13]
    wt1 = [-1]
    processOne(ana, order, wt1, ax1, ax2, id1)
    order = [14, 3]
    wt1 = [1, 2]
    processOne(ana, order, wt1, ax3, ax4, id1)
    order = [13, 14, 3]
    wt1 = [-1, 1, 2]
    processOne(ana, order, wt1, ax5, ax6, id1)
    return fig

def processDataVMm(ana, violin=1):
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    sel = 2
    order = [13]
    wt1 = [-1]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    fig = plt.figure(figsize=(5,9), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = plt.subplot2grid((8, 1), (0, 0))
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': genes, 'ax': ax, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    ax = plt.subplot2grid((8, 1), (1, 0), rowspan=3)
    ax2 = plt.subplot2grid((8, 1), (4, 0))
    ax3 = plt.subplot2grid((8, 1), (5, 0), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13")
    else:
        ax = ana.densityPlot(ax, acolor)
        ax.set_ylabel("Density - Cluster \\#13")
    #plt.tight_layout()
    order = [14, 3]
    wt1 = [1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#14-3")
    return fig

def processDataHMm(ana, violin=1):
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    sel = 2
    order = [13]
    wt1 = [-1]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = plt.subplot2grid((4, 3), (0, 0))
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': genes, 'ax': ax, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    ax = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 3), (0, 1))
    ax3 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13")
    else:
        ax = ana.densityPlot(ax, acolor)
        ax.set_ylabel("Density - Cluster \\#13")
    #plt.tight_layout()
    order = [14, 3]
    wt1 = [1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#14-3")
    ax2 = plt.subplot2grid((4, 3), (0, 2))
    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=3)
    order = [13, 14, 3]
    wt1 = [-1, 1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13-14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#13-14-3")
    return fig





def processDataHMmV2(ana, violin=1):
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    sel = 2
    order = [13]
    wt1 = [-1]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = plt.subplot2grid((4, 3), (0, 0))
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': genes, 'ax': ax, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    ax = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 3), (0, 1))
    ax3 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13")
    else:
        ax = ana.densityPlot(ax, acolor)
        ax.set_ylabel("Density - Cluster \\#13")
    #plt.tight_layout()
    wt1, l1 = getCls14a3v2()
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#14-3")
    ax2 = plt.subplot2grid((4, 3), (0, 2))
    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=3)
    wt1, l1 = getCls13a14a3v2()
    ana.convertMm(l1, genes)
    ana.orderData(ana.gene_groups, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13-14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#13-14-3")
    return fig

def processDataH(ana, violin=1):
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    sel = 2
    order = [13]
    wt1 = [-1]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = plt.subplot2grid((4, 3), (0, 0))
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': genes, 'ax': ax, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    ax = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 3), (0, 1))
    ax3 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13")
    else:
        ax = ana.densityPlot(ax, acolor)
        ax.set_ylabel("Density - Cluster \\#13")
    #plt.tight_layout()
    order = [14, 3]
    wt1 = [1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#14-3")
    ax2 = plt.subplot2grid((4, 3), (0, 2))
    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=3)
    order = [13, 14, 3]
    wt1 = [-1, 1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    print(" ".join([str(ana.atype[i]) for i in ana.i1]))
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    res = ana.getMetrics(ana.cval[0])
    ax.text(len(ana.cval[0]), 4, ",".join(res))
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13-14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#13-14-3")
    return fig

def processDataV(ana, violin=1):
    nx = [0, 1, 4, 5, 6, 8, 9, 10, 16, 17, 19, 20, 21, 25, 28]
    sel = 2
    order = [13]
    wt1 = [-1]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    fig = plt.figure(figsize=(5,9), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = plt.subplot2grid((8, 1), (0, 0))
    params = {'spaceAnn': len(ana.order)/len(ana.atypes), 'tAnn': 1, 'widthAnn':1,
              'genes': genes, 'ax': ax, 'acolor': acolor}
    ax = ana.printTitleBar(params)
    ax = plt.subplot2grid((8, 1), (1, 0), rowspan=3)
    ax2 = plt.subplot2grid((8, 1), (4, 0))
    ax3 = plt.subplot2grid((8, 1), (5, 0), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#13")
    else:
        ax = ana.densityPlot(ax, acolor)
        ax.set_ylabel("Density - Cluster \\#13")
    #plt.tight_layout()
    order = [14, 3]
    wt1 = [1, 2]
    genes, wt1, l1 = getGeneGroups([nx[i] for i in order], wt1, 0)
    ana.orderData(l1, wt1)
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":3, "tval":6.5, "select":sel})
    print(len(ana.col_labels), len(ana.row_ids))
    params.update({'ax':ax2})
    ax = ana.printTitleBar(params)
    if (violin == 1):
        params['ax'] = ax3
        params['vert'] = 0
        ax = ana.printViolin(None, params)
        ax.set_ylabel("Violin - Cluster \\#14-3")
    else:
        ax = ana.densityPlot(ax3, acolor)
        ax.set_ylabel("Density - Cluster \\#14-3")
    return fig

def processDataHViP(ana, id1 = None):
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 3), (0, 0))
    ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax3 = plt.subplot2grid((4, 3), (0, 1))
    ax4 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    wt1, l1 = getViP()
    processOne(ana, None, wt1, ax1, ax2, id1, l1)
    ax1.text(-0.5, 4, 'ViP', horizontalalignment='right',
                            verticalalignment='center')
    wt1, l1 = getSViP()
    processOne(ana, None, wt1, ax3, ax4, id1, l1)
    ax3.text(-0.5, 4, 'sViP', horizontalalignment='right',
                            verticalalignment='center')
    return fig

def processDataHViPMm(ana, id1 = None):
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    ax1 = plt.subplot2grid((4, 3), (0, 0))
    ax2 = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax3 = plt.subplot2grid((4, 3), (0, 1))
    ax4 = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    wt1, l1 = getViP()
    l1 = getGroupsMm(l1)
    processOne(ana, None, wt1, ax1, ax2, id1, l1)
    ax1.text(-0.5, 4, 'ViP', horizontalalignment='right',
                            verticalalignment='center')
    wt1, l1 = getSViP()
    l1 = getGroupsMm(l1)
    processOne(ana, None, wt1, ax3, ax4, id1, l1)
    ax3.text(-0.5, 4, 'sViP', horizontalalignment='right',
                            verticalalignment='center')
    return fig

def getFisher(predicted, actual):
    data_list = {'x' : predicted}
    df = pd.DataFrame(data_list)
    df['y'] = pd.Series(np.array(actual))
    target_names = ['N', 'C']
    tab = pd.crosstab(df.y > 0, df.x > 0)
    print(tab)
    if tab.shape == (2,2):
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return fisher_exact(tab)[1];
    return (1, 1)

def processGeneGroups(ana, l1, wt1, debug = 0, fthr = None, tn=1):
    ana.orderData(l1, wt1); print("ROC-AUC", ana.getMetrics3())
    actual = [1 if ana.aval[i] >= 1 else 0 for i in ana.i1]
    score = [ana.f_ranks[i - ana.h.start] for i in ana.i1]
    thr = hu.getThrData(score)
    nm = (np.max(ana.f_ranks) - np.min(ana.f_ranks))/16
    if fthr is None:
        fthr = thr[0]
    if fthr == "thr0":
        fthr = thr[0] - nm
    if fthr == "thr2":
        fthr = thr[0] + nm
    if fthr == "thr3":
        fthr = thr[0] + 3 * nm
    print(thr)
    print(nm, fthr)
    predicted = [1 if ana.f_ranks[i - ana.h.start] >= fthr else 0 for i in ana.i1]
    c_dict = {}
    for i in ana.order:
        c_dict[i] = ana.f_ranks[i - ana.h.start]
        if (tn == 2):
            c_dict[i] = 0
            if ana.f_ranks[i - ana.h.start] >= fthr:
                c_dict[i] = 1
    fpr, tpr, thrs = roc_curve(actual, score)
    roc_auc = auc(fpr, tpr)
    pval_fe = getFisher(predicted, actual);
    if debug == 1:
        print(actual)
        print(predicted)
        print(score)
        print(thr[0], thr, nm)
        print("ROC-AUC", roc_auc)
        print('Accuracy', accuracy_score(actual, predicted))
        print(classification_report(actual, predicted, target_names=ana.atypes))
    return c_dict, fpr, tpr, roc_auc

def processOneDataset(ana, l1, wt1, prms = None, violin=1):
    acolor = ["#40B549", "#FCEE23", "#D8A03D","#EC008C",
            'cyan', "#B741DC", "#808285",
            'blue', 'black', 'green', 'red']
    sel = 2
    ana.orderData(l1, wt1)
    #ana.normMacrophageGene("TYROBP", {"thr":1})
    #ana.normMacrophageGene("FCER1G", {"thr":4, "tval":-4, "select":sel})
    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = plt.subplot2grid((4, 1), (0, 0))
    params = {'spaceAnn': 10, 'tAnn': 3, 'widthAnn':3, 'acolor': acolor,
              'w': 5, 'h': 0.8, 'atypes': ana.atypes ,'cval': ana.cval,
              'ax': ax}
    if prms is not None:
        params.update(prms)
    ax = ana.printTitleBar(params)
    roc = ana.getMetrics()
    print("ROC-AUC", roc)
    ax.text(len(ana.cval[0]), 4, str(roc))
    ax = plt.subplot2grid((4, 1), (1, 0), rowspan=3)
    if (violin == 1):
        params['ax'] = ax
        params['vert'] = 0
        ax = ana.printViolin(None, params)
    else:
        ax = ana.densityPlot(ax, acolor)
    plt.tight_layout()
    plt.show()

def printStats(cfile, thr):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile));
        exit()
    fp = open(cfile, "r")
    numhigh = 0
    numlow = 0
    total = 0
    for line in fp:
        line = line.strip();
        ll = re.split("[\t]", line);
        if float(ll[2]) >= 0 and float(ll[3]) < thr:
            numhigh += 1
        if float(ll[2]) < 0  and float(ll[3]) < thr:
            numlow += 1
        total += 1
    fp.close();
    print(cfile, numhigh, numlow, total)

def getStats(cfile, thr, index):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    high = set()
    low = set()
    for line in fp:
        line = line.strip();
        ll = re.split("[\t]", line);
        if float(ll[2]) >= 0 and float(ll[3]) < thr:
            high.add(ll[index])
        if float(ll[2]) < 0  and float(ll[3]) < thr:
            low.add(ll[index])
    fp.close();
    return high, low

def getEntries(cfile, index):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    res = []
    for line in fp:
        line = line.strip();
        ll = re.split("[\t]", line);
        res += [ll[index]]
    fp.close();
    return res

def getPVal(cfile):
    return [float(i) for i in getEntries(cfile, 3)]

def getFdrStats(cfile, thr, index):
    pval = getPVal(cfile)
    pval = [1.0 if np.isnan(k) else k for k in pval]
    ids = getEntries(cfile, index)
    stat = getEntries(cfile, 2)
    from statsmodels.stats import multitest
    mstat = multitest.multipletests(pval, thr, 'fdr_bh')
    high = set()
    low = set()
    for i in range(len(pval)):
        if mstat[0][i] and float(stat[i]) >= 0:
            high.add(ids[i])
        if mstat[0][i] and float(stat[i]) < 0:
            low.add(ids[i])
    return high, low

def saveFdrStats(ofile, cfile, thr):
    pval = getPVal(cfile)
    pval = [1.0 if np.isnan(k) else k for k in pval]
    ids = getEntries(cfile, 0)
    name = getEntries(cfile, 1)
    stat = getEntries(cfile, 2)
    diff = getEntries(cfile, 3)
    from statsmodels.stats import multitest
    mstat = multitest.multipletests(pval, thr, 'fdr_bh')
    df = pd.DataFrame()
    df['ProbeID'] = ids
    df['Name'] = name
    df['Stat'] = stat
    df['pval'] = pval
    df['Diff'] = diff
    df['pval_adj'] = mstat[1]
    df['sig'] = mstat[0]
    df.to_csv(ofile, sep="\t", index=False)

def readIBDGenes(cfile):
    genes = "PRKAB1 PPARG PPP1CA PPARGC1A SIRT6 OCLN PARD3 IL23A IL6 S1PR5 ACTA2     CCDC88A COL1A1 CXCL10 ELMO1 HIF1A IL10 IL33 ITGA4 ITGB1 ITGB7 JAK1 MMP14 MMP2     MMP9 MRC1 NOD2 PML PRKCQ RIPK2 S1PR1 SNAI2 SPHK1 TGFB1 TIMP2 TLR2 TLR4 VIM CLDN2     IL11 MMP1 MMP3 CEMIP KIAA1199 PRKAA2 IL8 CXCL8 LGR5"
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    nodelist = re.split("[\[\]()\s]", genes)
    for line in fp:
        line = line.strip();
        ll = re.split("[\[\]()\s]", line);
        nodelist += ll
    fp.close();
    return [i for i in hu.uniq(nodelist) if i != '']

def readGenes(cfile):
    genes = ""
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    nodelist = re.split("[\[\]()\s]", genes)
    for line in fp:
        line = line.strip();
        ll = re.split("[\[\]()\s]", line);
        nodelist += ll
    fp.close();
    return [i for i in hu.uniq(nodelist) if i != '']

def plotSizes(csizes):
    w,h, dpi = (6, 4, 100)
    fig = plt.figure(figsize=(w,h), dpi=dpi)
    ax = fig.add_axes([70.0/w/dpi, 54.0/h/dpi, 1-2*70.0/w/dpi, 1-2*54.0/h/dpi])
    ax.loglog(range(len(csizes)), csizes, "r-", clip_on=False);
    ax.grid(False)
    ax.set_axis_bgcolor("white")
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('black')
            child.set_linewidth(0.5)
    ax.tick_params(direction='out', length=4, width=1, colors='k', top=False,
            right=False)
    ax.tick_params(which="minor", direction='out', length=2, width=0.5, 
                   colors='k', top=False, right=False)
    ax.set_xlabel("Clusters ranked by size")
    ax.set_ylabel("Cluster sizes")
    fig.savefig("paper/cluster-sizes-1.pdf", dpi=200)

def printReport(actual, predicted, score, target_names):
    print(classification_report(actual, predicted, target_names=target_names))
    fpr, tpr, _ = roc_curve(actual, score)
    roc_auc = auc(fpr, tpr)
    print('ROC AUC', roc_auc)
    print('ROC AUC', roc_auc_score(actual, score))
    print('Accuracy', accuracy_score(actual, predicted))
    wi,hi, dpi = (4, 4, 100)
    fig = plt.figure(figsize=(wi,hi), dpi=dpi)
    ax = fig.add_axes([70.0/wi/dpi, 54.0/hi/dpi, 1-2*70.0/wi/dpi, 1-2*54.0/hi/dpi])
    ax.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    return ax

def convertScore(mylist):
    print(mylist)
    hs = dict()
    for x in mylist:
        if (x not in hs):
            hs[x] = 1
        else:
            hs[x] += 1
    keys =list(hs.keys())
    values = [0] + list(np.cumsum(list(hs.values())))
    for i in range(len(keys)):
        hs[keys[i]] = values[i]
    hh = dict()
    res = []
    for x in mylist:
        if (x not in hh):
            hh[x] = 0
        else:
            hh[x] += 1
        res += [hh[x] + hs[x]]
    return res

def mergeRanks(group, start, exp, weight):
    X = np.array([[e[k-start] for e in exp] for k in group])
    arr = np.dot(X, np.array(weight))
    return arr

def mergeRanks2(group, exp, weight):
    X = np.array([[e[k] for e in exp] for k in range(len(group))])
    arr = np.dot(X, np.array(weight))
    return arr

def getOrder(group, start, exp, weight):
    arr = mergeRanks(group, start, exp, weight)
    return [group[i] for i in np.argsort(arr)]

def getRanks(gene_groups, h):
    expr = []
    row_labels = []
    ranks = []
    counts = []
    for s in gene_groups:
        print(len(s), s)
        count = 0
        avgrank = [0 for i in h.aRange()]
        for gn in s:
          for id in h.getIDs(gn):
            e = h.getExprData(id);
            t = h.getThrData(id);
            if e[-1] == "":
                continue
            v = np.array([float(e[i]) for i in h.aRange()])
            te = []
            for i in h.aRange():
                v1 = (float(e[i]) - t[3]) / 3;
                if np.std(v) > 0:
                    v1 = v1 / np.std(v)
                avgrank[i-h.start] += v1
                te.append(v1)
            expr.append(te)
            #row_labels.append(h.getSimpleName(id))
            row_labels.append(gn)
            count += 1
            #if count > 100:
            #    break
        ranks.append(avgrank)
        counts += [count]
    print(counts)
    return ranks, row_labels, expr

def getRanks2(gene_groups, h):
    expr = []
    row_labels = []
    row_ids = []
    row_numhi = []
    ranks = []
    g_ind = 0
    counts = []
    for s in gene_groups:
        count = 0
        avgrank = [0 for i in h.aRange()]
        for gn in s:
          for id in h.getIDs(gn):
            e = h.getExprData(id);
            t = h.getThrData(id);
            if t is None or len(e) != (h.getEnd()+1):
                continue
            v = np.array([float(e[i]) if e[i] != "" else 0 for i in h.aRange()])
            te = []
            sd = np.std(v)
            if sd == np.NaN or sd <= 0:
                continue
            for i in h.aRange():
                if (e[i] != ""):
                    v1 = (float(e[i]) - t[3]) / 3;
                    if sd > 0:
                        v1 = v1 / sd
                else:
                    v1 = -t[3]/3/sd
                avgrank[i-h.start] += v1
                te.append(v1)
            expr.append(te)
            row_labels.append(h.getSimpleName(id))
            row_ids.append(id)
            v1 = [g_ind, sum(v > t[3])]
            if g_ind > 3:
                v1 = [g_ind, sum(v <= t[3])]
            else:
                v1 = [g_ind, sum(v > t[3])]
            row_numhi.append(v1)
            count += 1
            #if count > 200:
            #    break
        ranks.append(avgrank)
        g_ind += 1
        counts += [count]
    print(counts)
    return ranks, row_labels, row_ids, row_numhi, expr

def getRanks3(gene_groups, h, order):
    expr = []
    row_labels = []
    row_ids = []
    row_numhi = []
    ranks = []
    g_ind = 0
    counts = []
    for s in gene_groups:
        count = 0
        avgrank = [0 for i in order]
        for gn in s:
          for id in h.getIDs(gn):
            e = h.getExprData(id);
            if len(e) != (h.getEnd()+1):
                continue
            v = np.array([float(e[i]) if e[i] != "" else 0 for i in order])
            t = hu.getThrData(v)
            te = []
            for i in range(len(order)):
                if e[order[i]] == "":
                    v1 = - t[3] / 3;
                else:
                    v1 = (float(e[order[i]]) - t[3]) / 3;
                if np.std(v) > 0:
                    v1 = v1 / np.std(v)
                avgrank[i] += v1
                te.append(v1)
            expr.append(te)
            row_labels.append(h.getSimpleName(id))
            row_ids.append(id)
            v1 = [g_ind, sum(v > t[3])]
            if g_ind > 3:
                v1 = [g_ind, sum(v <= t[3])]
            else:
                v1 = [g_ind, sum(v > t[3])]
            row_numhi.append(v1)
            count += 1
            #if count > 200:
            #    break
        ranks.append(avgrank)
        g_ind += 1
        counts += [count]
    print(counts)
    return ranks, row_labels, row_ids, row_numhi, expr

def getGeneGroups2(order = None, weight = None, debug = 1):
    reload(hu)
    db = hu.Database("/booleanfs2/sahoo/Hegemon/explore.conf")
    h = hu.Hegemon(db.getDataset("PLP11"))
    h.init()
    h.initPlatform()
    h.initSurv()
    data_item = []
    with open('CD/info2/path-1.json') as data_file:
        data_item += json.load(data_file)
    with open('CD/info2/path-2.json') as data_file:
        data_item += json.load(data_file)
    cfile = "CD/cd-network-2-cls.txt"
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    nodelist = {}
    nhash = {}
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        nodelist[ll[0]] = ll[2:]
        for i in ll[2:]:
            nhash[i] = ll[0];
    fp.close();
    gene_groups = []
    for i in range(len(data_item)):
        gene_groups.append(set())
        gn = data_item[i][2][0][0]
        for g in data_item[i][2]:
            gene_groups[i].add(g[0])
            if g[0] in nodelist:
                for k in nodelist[g[0]]:
                    gene_groups[i].add(k)
        for g in data_item[i][3]:
            gene_groups[i].add(g)
            if g in nodelist:
                for k in nodelist[g]:
                    gene_groups[i].add(k)
        if debug == 1:
            print(i, gn, h.getSimpleName(gn), data_item[i][0], len(gene_groups[i]))
    print([len(s) for s in gene_groups])
    if order is None:
        order = [7, 6, 5, 1];
        order = [7, 6, 5];
    gene_groups = [gene_groups[i] for i in order]
    print([len(s) for s in gene_groups])
    gene_groups = getSimpleName(gene_groups, h)
    print([len(s) for s in gene_groups])
    if weight is None:
        weight = [-3, -2, -1]
    print(weight)
    genes = readGenes("cluster-names.txt")
    return genes, weight, gene_groups

def getRanks4(gene_groups, h, m0, m1, order):
    expr = []
    row_labels = []
    row_ids = []
    row_numhi = []
    ranks = []
    g_ind = 0
    counts = []
    for s in gene_groups:
        count = 0
        avgrank = [0 for i in order]
        for gn in s:
          for id in h.getIDs(gn):
            e = h.getExprData(id);
            if len(e) != (h.getEnd()+1):
                continue
            v = np.array([float(e[i]) if e[i] != "" else 0 for i in order])
            if (np.max(v) - np.min(v)) <= 0:
                continue
            v1 = np.mean([float(e[i]) if e[i] != "" else 0 for i in m1])
            v0 = np.mean([float(e[i]) if e[i] != "" else 0 for i in m0])
            t = [v1, v1, v1, (v0 + v1)/2]
            te = []
            for i in range(len(order)):
                if e[order[i]] == "":
                    v1 = - t[3] / 3;
                else:
                    v1 = (float(e[order[i]]) - t[3]) / 3;
                if np.std(v) > 0:
                    v1 = v1 / np.std(v)
                avgrank[i] += v1
                te.append(v1)
            expr.append(te)
            row_labels.append(h.getSimpleName(id))
            row_ids.append(id)
            v1 = [g_ind, sum(v > t[3])]
            if g_ind > 3:
                v1 = [g_ind, sum(v <= t[3])]
            else:
                v1 = [g_ind, sum(v > t[3])]
            row_numhi.append(v1)
            count += 1
            #if count > 200:
            #    break
        ranks.append(avgrank)
        g_ind += 1
        counts += [count]
    print(counts)
    return ranks, row_labels, row_ids, row_numhi, expr

def saveList(ofile, l1):
    of = open(ofile, "w")
    for i in l1:
        of.write("\t".join([i]) + "\n")
    of.close()

def readList(cfile):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    f_order = []
    for line in fp:
        line = line.strip();
        ll = re.split("[\s]", line);
        f_order += [ll[0]]
    fp.close();
    return f_order

def saveCData(ofile, h, i1, f_ranks):
    f_order = dict()
    for i in h.aRange():
        f_order[i] = ""
    for i in range(len(i1)):
        f_order[i1[i]] = str(i)
    of = open(ofile, "w")
    for i in h.aRange():
        id1 = h.headers[i]
        of.write("\t".join([id1, f_order[i], str(f_ranks[i - h.start])]) + "\n")
    of.close()

def saveHeatmapData(ofile, row_labels, row_numhi, row_ids, index, expr):
    ind_r = np.array(sorted(range(len(row_labels)), key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
    of = open(ofile, "w")
    for i in ind_r:
        id1 = row_ids[i]
        of.write("\t".join([id1, row_labels[i], str(row_numhi[i][0]),                             str(row_numhi[i][1])] + [str(expr[i][j]) for j in index]) + "\n")
    of.close()

def readCData(cfile):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    f_order = []
    f_ranks = []
    for line in fp:
        line = line.strip();
        ll = re.split("[\s]", line);
        f_order += [ll[1]]
        f_ranks += [float(ll[2])]
    fp.close();
    return f_order, f_ranks

def readHeatmapData(cfile):
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    row_labels, row_numhi, row_ids, expr = [], [], [], []
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        row_ids += [ll[0]]
        row_labels += [ll[1]]
        row_numhi += [ [int(ll[2]), int(ll[3])] ]
        expr += [[float(k) for k in ll[4:]]]
    fp.close();
    return row_labels, row_numhi, row_ids, expr

def barTop(tax, atypes, color_sch1, params):
    spaceAnn = 70
    widthAnn = 3
    tAnn = 1
    if 'spaceAnn' in params:
        spaceAnn = params['spaceAnn']
    if 'widthAnn' in params:
        widthAnn = params['widthAnn']
    if 'tAnn' in params:
        tAnn = params['tAnn']
    for i in range(len(atypes)):
        tax.add_patch(patches.Rectangle( (i *spaceAnn, 0), widthAnn, 3,
                                        facecolor=color_sch1[i % len(color_sch1)], edgecolor="none", alpha=1.0))
        tax.text(i * spaceAnn + widthAnn + tAnn, 1, atypes[i], rotation='horizontal',
                 ha='left', va='center', fontsize=12)

def plotHeatmap(ofile, data, col_labels, row_labels, params):
    
    genes = []
    atypes = []
    cval = []
    dpi, tl, tw, ts, tsi, lfs = (100, 3, 0.25, 0.5, 0, 8)
    if 'genes' in params:
        genes = params['genes']
    if 'atypes' in params:
        atypes = params['atypes']
    if 'cval' in params:
        cval = params['cval']
    if 'dpi' in params:
        dpi = params['dpi']
    if 'tl' in params:
        tl = params['tl']
    if 'tw' in params:
        tw = params['tw']
    if 'ts' in params:
        ts = params['ts']
    if 'tsi' in params:
        tsi = params['tsi']
    if 'lfs' in params:
        lfs = params['lfs']
    
    w,h = (12, 12)
    dx, dy = (10, 10)
    if 'dx' in params:
        dx = params['dx']
    if 'dy' in params:
        dy = params['dy']
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
        
    nAt, nGt = (len(col_labels), len(row_labels))
    if 'ax' not in params:
        fig = plt.figure(figsize=(w,h), dpi=dpi)
        ax = fig.add_axes([70.0/w/dpi, 54.0/h/dpi, 1-2*70.0/w/dpi, 1-2*54.0/h/dpi])
    else:
        ax = params['ax']
    extent = [0, nAt*dx, 0, nGt*dy]

    cvals  = [-1, -0.7, -0.4, 0, 0, 0.2, 1]
    clrs = ["#210B61","#0B614B","#04B45F", "#D8F781", "#F2F5A9", "red", "#DF0101"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), clrs))
    cmap = colors.LinearSegmentedColormap.from_list("BGYR1", tuples)
    plt.register_cmap(cmap=cmap)

    cvals  = [-1, -0.7, -0.4, 0, 0, 0.8, 1]
    clrs = ["#210B61","#0B614B","#04B45F", "#D8F781", "#F2F5A9", "red", "#DF0101"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), clrs))
    cmap = colors.LinearSegmentedColormap.from_list("BGYR2", tuples)
    plt.register_cmap(cmap=cmap)

    im = ax.imshow(data, cmap="bwr", interpolation='nearest', vmin=-2.0, vmax=2.0, extent = extent)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticklabels([])
    
    yticks = []
    ylabels = []
    for g in genes:
        if g in row_labels:
            i = row_labels.index(g)
            yticks += [-dy/2 + (len(row_labels) - i) * dy]
            ylabels += [ row_labels[i] ]
    si = np.argsort(np.array(yticks))
    yiticks = np.array(yticks)[si]
    yoticks = np.array(yticks)[si]
    ylabels = np.array(ylabels)[si]
    sy = 5
    if 'sy' in params:
        sy = params['sy']
    for i in range(1, len(yoticks)):
        diff = yoticks[i] - yoticks[i - 1]
        if diff < sy*dy:
            yoticks[i] = yoticks[i - 1] + sy*dy
    for i in range(len(yoticks)):
        yoticks[i] = yoticks[i] + tsi
    ax.set_yticks(yiticks)
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.grid(False)
    ax.tick_params(top=False, left=True, bottom=False, right=False,
            length=tl, width=tw)
    plt.xlim(xmin=0)
    trans = blended_transform_factory(ax.transData, ax.transData)
    fx, fy =  ax.transData.transform((1, 1)) - ax.transData.transform((0, 0))
    fx = dpi/fx/72
    fy = dpi/fy/72
    print(fx, fy)
    fx = max(fx, fy)
    fy = max(fx, fy)
    oo = 2
    for i in range(len(yoticks)):
        ax.annotate(str(ylabels[i]), xy=(0.0, yoticks[i]),
                xycoords=trans,
                xytext=(-(2*tl+ts), 0), textcoords='offset points', color="black",
                fontsize=lfs, ha="right")
        ax.plot((-(2*tl+ts)*fx, -(tl+ts+oo)*fx, -(tl+oo)*fx, -tl*fx),
                (yoticks[i]+4*fy, yoticks[i]+4*fy, yiticks[i], yiticks[i]),
                transform=trans,
                linestyle='solid', linewidth=tw, color='black', clip_on=False)
        oo += 0.5
        if (oo > 2):
            oo = 0

    # Create colorbar
    aspect = 20
    pad_fraction = 0.5
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Expression", rotation=-90, va="bottom")

    color_sch1 = ["#3B449C", "#B2509E","#EA4824"]
    color_sch1 = ["#00CC00", "#EFF51A","#EC008C", "#F7941D", "#808285",
            'cyan', 'blue', 'black', 'green', 'red']
    if 'acolor' in params:
        color_sch1 = params['acolor']

    if len(cval) > 0:
        width = axes_size.AxesX(ax, aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        tax = divider.append_axes("top", size=width, pad=pad)
        extent = [0, nAt, 0, 5]
        tax.axis(extent)
        cmap = colors.ListedColormap(color_sch1)
        boundaries = range(len(color_sch1) + 1)
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        tax.imshow(cval, interpolation='nearest', cmap=cmap, norm=norm, extent=extent, aspect="auto")
        tax.set_xticklabels([])
        tax.set_yticklabels([])
        tax.tick_params(top=False, left=False, bottom=False, right=False)
        if 'tline' in params and params['tline'] == 1:
            tax.set_xticks(np.arange(0, nAt, 1))
            tax.grid(which='major', alpha=1, linestyle='-', linewidth='1',
                    color='black')
        else:
            tax.grid(False)

    pad = axes_size.Fraction(0.2, width)
    lax = divider.append_axes("top", size=width, pad=pad, frame_on=False)
    lax.axison = False
    lax.axis(extent)
    lax.set_xticklabels([])
    lax.set_yticklabels([])
    lax.grid(False)
    lax.tick_params(top=False, left=False, bottom=False, right=False)
    barTop(lax, atypes, color_sch1, params)

    ax.get_figure().savefig(ofile, dpi=dpi)
    return ax

def plotTitleBar(cval, atypes, params):
    dpi = 100
    if 'dpi' in params:
        dpi = params['dpi']
    w,h = (5, 0.8)
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
    color_sch1 = ["#3B449C", "#B2509E","#EA4824"]
    color_sch1 = ["#00CC00", "#EFF51A","#EC008C", "#F7941D", "#808285",
            'cyan', 'blue', 'black', 'green', 'red']
    if 'acolor' in params:
        color_sch1 = params['acolor']
    if 'cval' in params:
        cval = params['cval']

    ax = None
    if 'ax' in params:
        ax = params['ax']
    if ax is None:
        fig = plt.figure(figsize=(w,h), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    nAt = len(cval[0])
    extent = [0, nAt, 0, 5]
    ax.axis(extent)
    cmap = colors.ListedColormap(color_sch1)
    boundaries = range(len(color_sch1) + 1)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(cval, interpolation='nearest', cmap=cmap, \
                      norm=norm, extent=extent, aspect="auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(top=False, left=False, bottom=False, right=False)
    ax.set_xticks(np.arange(0, nAt, 1))
    ax.grid(which='major', alpha=0.2, linestyle='-', linewidth=0.5,
            color='black')
    for edge, spine in ax.spines.items():
                spine.set_visible(False)
    divider = make_axes_locatable(ax)
    width = axes_size.AxesX(ax, aspect=1./20)
    spaceAnn = 70
    widthAnn = 3
    tAnn = 1
    if 'spaceAnn' in params:
        spaceAnn = params['spaceAnn']
    if 'widthAnn' in params:
        widthAnn = params['widthAnn']
    if 'tAnn' in params:
        tAnn = params['tAnn']
    pad = axes_size.Fraction(0.1, width)
    lax = divider.append_axes("top", size="100%", pad="20%", frame_on=False)
    lax.axison = False
    lax.axis(extent)
    lax.set_xticklabels([])
    lax.set_yticklabels([])
    lax.grid(False)
    lax.tick_params(top=False, left=False, bottom=False, right=False)
    if 'atypes' in params:
        atypes = params['atypes']
    barTop(lax, atypes, color_sch1, params)
    return ax

def plotDensity(x, atypes, ax = None, color = None):
    if color is None:
        color = acolor
    df = pd.Series(x)
    for i in range(len(atypes)):
        idx = df[df == i].index
        n = len(idx)
        l = str(atypes[i]) + "(" + str(n) + ")"
        df1 = pd.DataFrame(pd.Series(idx),
                columns=[l])
        if n <= 1:
            continue
        if ax is None:
            #ax = df1.plot.kde(bw_method=1.0, c=color[i], label=l)
            ax = sns.kdeplot(df1[l], bw = n, cut = 2, color=color[i], label=l)
        else:
            #ax = df1.plot.kde(bw_method=1.0, ax = ax, c=color[i], label=l)
            ax = sns.kdeplot(df1[l], bw = n, cut =2, ax = ax, color=color[i], label=l)
    for i in range(len(atypes)):
        idx = df[df == i].index
        n = len(idx)
        l = str(atypes[i]) + "(" + str(n) + ")"
        df1 = pd.DataFrame(pd.Series(idx),
                columns=[l])
        if n != 1:
            continue
        df1['y'] = 1
        if ax is None:
            #ax = df1.plot.line(x=l, y='y', c=color[i], label=l)
            ax = plt.axvline(x=idx[0], c=color[i])
        else:
            #ax = df1.plot.line(x=l, y='y', ax = ax, c=color[i], label=l)
            ax.axvline(x=idx[0], c=color[i])

    ax.set_title("Density plot")
    ax.set_xlabel("Sample rank")
    return ax

def adj_light(color, l=1, s=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, l * c[1])), 
                 max(0, min(1, s * c[2])))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def cAllPvals(lval, atypes):
    for i in range(len(lval)):
        for j in range(i +1, len(lval)):
            if len(lval[i]) <= 0:
                continue
            if len(lval[j]) <= 0:
                continue
            #print(lval[i])
            #print(lval[j])
            t, p = ttest_ind(lval[i],lval[j], equal_var=False)
            desc = "%s vs %s %.3g, %.3g" % (atypes[i], atypes[j], t, p)
            print(desc)

def plotScores(data, atypes, params):
    dpi = 100
    if 'dpi' in params:
        dpi = params['dpi']
    vert = 0
    if 'vert' in params:
        vert = params['vert']
    if vert == 0:
        w,h = (2, 0.25 * len(atypes))
    else:
        w,h = (0.75 * len(atypes), 2)
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
    color_sch1 = ["#3B449C", "#B2509E","#EA4824"]
    color_sch1 = ["#00CC00", "#EFF51A","#EC008C", "#F7941D", "#808285",
            'cyan', 'blue', 'black', 'green', 'red']
    if 'acolor' in params:
        color_sch1 = params['acolor']
    if 'cval' in params:
        cval = params['cval']

    ax = None
    if 'ax' in params:
        ax = params['ax']
    if ax is None:
        fig = plt.figure(figsize=(w,h), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)

    meanpointprops = dict(marker='D', markerfacecolor='none', markersize=5,
                              linestyle='none')
    meanlineprops = dict(linestyle='--', linewidth=1, color='purple')
    cols = [ color_sch1[i % len(color_sch1)] for i in range(len(data)) ]
    bp = ax.boxplot(data, notch=0, sym='+', vert=vert, whis=1.5,
            showmeans=1, meanline=0, meanprops=meanpointprops, widths=0.7)
    if vert == 0:
        ax.set_yticklabels(atypes)
        ax.set_xticklabels([])
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
    else:
        ax.set_xticklabels(atypes)
    ax.grid(False)
    ax.tick_params(top=False, left=True, bottom=False, right=False)

    from matplotlib.patches import Polygon,Arrow
    for i in range(len(data)):
        col1 = adj_light(cols[i], 1, 0.5)
        col2 = 'blue'
        col3 = adj_light(cols[i], 0.5, 1)
        plt.setp(bp['medians'][i], color='red')
        plt.setp(bp['boxes'][i], color=col1)
        plt.setp(bp['means'][i], markeredgecolor=col2)
        plt.setp(bp['whiskers'][2*i], color=col1)
        plt.setp(bp['whiskers'][2*i+1], color=col1)
        plt.setp(bp['caps'][2*i], color=col1)
        plt.setp(bp['caps'][2*i+1], color=col1)
        plt.setp(bp['fliers'][i], markeredgecolor=cols[i], marker='o')
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        ax.add_patch(Polygon(box_coords, facecolor=cols[i], alpha=0.2))
        m, h = mean_confidence_interval(data[i])
        if vert == 0:
            ax.add_patch(Arrow(m, i + 1,
                h, 0, width=1, facecolor=col3, alpha=0.4))
            ax.add_patch(Arrow(m, i + 1,
                -h, 0, width=1, facecolor=col3, alpha=0.4))
        else:
            ax.add_patch(Arrow(i + 1, m,
                0, h, width=1, facecolor=col3, alpha=0.4))
            ax.add_patch(Arrow(i + 1, m,
                0, -h, width=1, facecolor=col3, alpha=0.4))

    return ax, bp

def plotViolin(data, atypes, params):
    df = pd.DataFrame()
    df['score'] = [k for i in range(len(data)) for k in data[i]]
    df['category'] = [atypes[i] for i in range(len(data)) for k in data[i]]
    m1 = []
    pvals = []
    for i in range(1, len(data)):
        if len(data[i]) <= 0:
            m1 += [0]
            pvals += [""]
            continue
        m1 += [max(data[i]) + (max(data[i]) - min(data[i])) * 0.1]
        t, p = ttest_ind(data[0],data[i], equal_var=False)
        if (p < 0.05):
            pvals += ["p=%.3g" % p]
        else:
            pvals += [""]
    dpi = 100
    if 'dpi' in params:
        dpi = params['dpi']
    w,h = (1.5 * len(atypes), 4)
    if 'w' in params:
        w = params['w']
    if 'h' in params:
        h = params['h']
    color_sch1 = acolor
    if 'acolor' in params:
        color_sch1 = params['acolor']
    sns.set()
    sns.set_style("white")
    sns.set_style({'text.color': '.5', 
        'xtick.color':'.5', 'ytick.color':'.5', 'axes.labelcolor': '.5'})
    sns.set_context("notebook")
    sns.set_palette([adj_light(c, 1.5, 1) for c in color_sch1])
    ax = None
    if 'ax' in params:
        ax = params['ax']
    if ax is None:
        fig,ax = plt.subplots(figsize=(w,h), dpi=dpi)
    width = 1
    height = 1
    if 'width' in params:
        width = params['width']
    if 'vert' in params and params['vert'] == 1:
        ax = sns.violinplot(x="category", y="score", inner='quartile',
                linewidth=0.5, width=width, ax = ax, data=df,
                order = atypes)
        ax = sns.swarmplot(x="category", y="score", color = 'blue', alpha=0.2,
                ax=ax, data=df, order = atypes)
        ax.set_xlabel("")
        pos = range(len(atypes))
        for tick,label in zip(pos[1:],ax.get_xticklabels()[1:]):
            ax.text(pos[tick], m1[tick - 1], pvals[tick - 1],
                    horizontalalignment='center', size=12,
                    color='0.3')
        ax.yaxis.grid(True, clip_on=False)
    else:
        ax = sns.violinplot(x="score", y="category", inner='quartile',
                linewidth=0.5, width=width, ax = ax, data=df,
                order = atypes)
        ax = sns.swarmplot(x="score", y="category", color = 'blue', alpha=0.2,
                ax=ax, data=df, order = atypes)
        ax.set_ylabel("")
        pos = range(len(atypes))
        for tick,label in zip(pos[1:],ax.get_yticklabels()[1:]):
            ax.text(m1[tick - 1], pos[tick]-0.5, pvals[tick - 1],
                    horizontalalignment='center', size=12,
                    color='0.3')
        ax.xaxis.grid(True, clip_on=False)
    return ax

def getGroupsMmv1(gene_groups):
    cfile = "/booleanfs2/sahoo/Data/SeqData/genome/Homo_sapiens.GRCh38.95.chr_patch_hapl_scaff.len.txt"
    fp = open(cfile, "r")
    hsdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        hsdict[ll[0]] = ll[1]
    fp.close();

    cfile = "/booleanfs2/sahoo/Data/Sarah/CellCycle/database/mart-export-hs-mm.txt"
    fp = open(cfile, "r")
    mmdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        if len(ll) > 3 and ll[0] in hsdict:
            g = hsdict[ll[0]]
            if g not in mmdict:
                mmdict[g] = []
            mmdict[g] += [ll[3]]
    fp.close();

    gene_groups_mm = []
    for s in gene_groups:
        s1 = set()
        for g in s:
            if g in mmdict:
                for k in mmdict[g]:
                    s1.add(k)
        gene_groups_mm.append(s1)
    return gene_groups_mm

def getGroupsMm(gene_groups):
    cfile = "/booleanfs2/sahoo/Data/Sarah/CellCycle/database/ensembl-GRCh38.p13-100-hs-mm.txt"
    fp = open(cfile, "r")
    mmdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        if len(ll) > 3 and ll[2] != '' and ll[3] != '':
            g = ll[3]
            if g not in mmdict:
                mmdict[g] = []
            mmdict[g] += [ll[2]]
    fp.close();

    gene_groups_mm = []
    for s in gene_groups:
        s1 = set()
        for g in s:
            if g in mmdict:
                for k in mmdict[g]:
                    s1.add(k)
        gene_groups_mm.append(s1)
    return gene_groups_mm

def getGroupsHsv1(gene_groups):
    cfile = "/booleanfs2/sahoo/Data/SeqData/genome/Homo_sapiens.GRCh38.95.chr_patch_hapl_scaff.len.txt"
    fp = open(cfile, "r")
    hsdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        hsdict[ll[0]] = ll[1]
    fp.close();

    cfile = "/booleanfs2/sahoo/Data/Sarah/CellCycle/database/mart-export-hs-mm.txt"
    fp = open(cfile, "r")
    mmdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        if len(ll) > 3 and ll[0] in hsdict:
            g = hsdict[ll[0]]
            if ll[3] not in mmdict:
                mmdict[ll[3]] = []
            mmdict[ll[3]] += [g]
    fp.close();

    gene_groups_hs = []
    for s in gene_groups:
        s1 = set()
        for g in s:
            if g in mmdict:
                for k in mmdict[g]:
                    s1.add(k)
        gene_groups_hs.append(s1)
    return gene_groups_hs

def getGroupsHs(gene_groups):
    cfile = "/booleanfs2/sahoo/Data/Sarah/CellCycle/database/ensembl-GRCm38.p6-100-mm-hs.txt"
    fp = open(cfile, "r")
    mmdict = {}
    for line in fp:
        line = line.strip();
        ll = re.split("\t", line);
        if len(ll) > 3 and ll[1] != '' and ll[2] != '':
            g = ll[1]
            if g not in mmdict:
                mmdict[g] = []
            mmdict[g] += [ll[2]]
    fp.close();

    gene_groups_hs = []
    for s in gene_groups:
        s1 = set()
        for g in s:
            if g in mmdict:
                for k in mmdict[g]:
                    s1.add(k)
        gene_groups_hs.append(s1)
    return gene_groups_hs

def getGeneGroups(order = None, weight = None, debug = 1):
    reload(hu)
    db = hu.Database("/booleanfs2/sahoo/Hegemon/explore.conf")
    h = hu.Hegemon(db.getDataset("GL4"))
    h.init()
    h.initPlatform()
    h.initSurv()
    data_item = []
    with open('figures/path-1.json') as data_file:
        data_item += json.load(data_file)
    with open('figures/path-0.json') as data_file:
        l1 = json.load(data_file)
        data_item[5] = l1[5]
        data_item[6] = l1[6]
    with open('figures/path-2.json') as data_file:
        data_item += json.load(data_file)
    with open('figures/path-3.json') as data_file:
        data_item += json.load(data_file)
    with open('figures/path-4.json') as data_file:
        data_item += json.load(data_file)
    cfile = "figures/mac-net-cls.txt"
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    nodelist = {}
    nhash = {}
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        nodelist[ll[0]] = ll[2:]
        for i in ll[2:]:
            nhash[i] = ll[0];
    fp.close();
    gene_groups = []
    for i in range(len(data_item)):
        gene_groups.append(set())
        gn = data_item[i][2][0][0]
        for g in data_item[i][2]:
            gene_groups[i].add(g[0])
            if g[0] in nodelist:
                for k in nodelist[g[0]]:
                    gene_groups[i].add(k)
        for g in data_item[i][3]:
            gene_groups[i].add(g)
            if g in nodelist:
                for k in nodelist[g]:
                    gene_groups[i].add(k)
        if debug == 1:
            print(i, gn, h.getSimpleName(gn), data_item[i][0], len(gene_groups[i]))
    print([len(s) for s in gene_groups])
    if order is None:
        order = [1, 3, 4, 5];
        order = [35]
        order = [43, 44, 45];
        order = [8, 9, 10]
    gene_groups = [gene_groups[i] for i in order]
    print([len(s) for s in gene_groups])
    gene_groups = getSimpleName(gene_groups, h)
    print([len(s) for s in gene_groups])
    if weight is None:
        weight = [-1, 1, 2, 3]
        weight = [-1, -2, -3]
        weight = [-1]
        weight = [-1, -2, -3]
    print(weight)
    genes = readGenes("figures/cluster-names.txt")
    return genes, weight, gene_groups
def getSName(name):
    l1 = re.split(": ", name)
    l2 = re.split(" /// ", l1[0])
    return l2[0]

def getRanksDf(df_e, df_t):
    expr = []
    row_labels = []
    row_ids = []
    row_numhi = []
    ranks = []
    g_ind = 0
    counts = []
    for k in range(len(df_e)):
        count = 0
        order = range(2, df_e[k].shape[1])
        avgrank = [0 for i in order]
        for j in range(df_e[k].shape[0]):
            e = df_e[k].iloc[j,:]
            t = df_t[k]['thr2'][j]
            if e[-1] == "":
                continue
            v = np.array([float(e[i]) if e[i] != "" else 0 for i in order])
            te = []
            sd = np.std(v)
            for i in order:
                if (e[i] != ""):
                    v1 = (float(e[i]) - t) / 3;
                    if sd > 0:
                        v1 = v1 / sd
                else:
                    v1 = -t/3/sd
                avgrank[i-2] += v1
                te.append(v1)
            expr.append(te)
            nm = getSName(e[1])
            row_labels.append(nm)
            row_ids.append(e[0])
            v1 = [g_ind, sum(v > t)]
            if g_ind > 3:
                v1 = [g_ind, sum(v <= t)]
            else:
                v1 = [g_ind, sum(v > t)]
            row_numhi.append(v1)
            count += 1
            #if count > 200:
            #    break
        ranks.append(avgrank)
        g_ind += 1
        counts += [count]
    print(counts)
    return ranks, row_labels, row_ids, row_numhi, expr

def getIBDGeneGroups(order = None, weight = None, debug = 1):
    reload(hu)
    db = hu.Database("/booleanfs2/sahoo/Hegemon/explore.conf")
    h = hu.Hegemon(db.getDataset("PLP7"))
    h.init()
    h.initPlatform()
    h.initSurv()
    data_item = []
    dir1 = "/booleanfs2/sahoo/Data/Pradipta/"
    with open(dir1 + 'Supplementary/path-1.json') as data_file:
        data_item += json.load(data_file)
    with open(dir1 + 'Supplementary/path-2.json') as data_file:
        data_item += json.load(data_file)
    cfile = dir1 + "Supplementary/ibd-network-clusters.txt"
    if not os.path.isfile(cfile):
        print("Can't open file {0} <br>".format(cfile))
        exit()
    fp = open(cfile, "r")
    nodelist = {}
    nhash = {}
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        nodelist[ll[0]] = ll[2:]
        for i in ll[2:]:
            nhash[i] = ll[0];
    fp.close();
    gene_groups = []
    for i in range(len(data_item)):
        gene_groups.append(set())
        gn = data_item[i][2][0][0]
        for g in data_item[i][2]:
            gene_groups[i].add(g[0])
            if g[0] in nodelist:
                for k in nodelist[g[0]]:
                    gene_groups[i].add(k)
        for g in data_item[i][3]:
            gene_groups[i].add(g)
            if g in nodelist:
                for k in nodelist[g]:
                    gene_groups[i].add(k)
        if debug == 1:
            print(i, gn, h.getSimpleName(gn), data_item[i][0], len(gene_groups[i]))
    print([len(s) for s in gene_groups])
    if order is None:
        order = [7, 6, 5, 1];
        order = [7, 6, 5];
    gene_groups = [gene_groups[i] for i in order]
    print([len(s) for s in gene_groups])
    gene_groups = getSimpleName(gene_groups, h)
    print([len(s) for s in gene_groups])
    if weight is None:
        weight = [-3, -2, -1]
    print(weight)
    genes = readGenes(dir1 + "cluster-names.txt")
    return genes, weight, gene_groups

def getSimpleName(gene_groups, h):
    res = []
    for s in gene_groups:
        s1 = set()
        for g in s:
            for id1 in h.getIDs(g):
                name = h.getSimpleName(id1)
                if name != "" and name != "---":
                    s1.add(name)
        res.append(s1)
    return res

def getCoates():
    #PMID: 18199539 DOI: 10.1158/0008-5472.CAN-07-3050
    cfile = "database/c2.all.v6.2.symbols.gmt"
    fp = open(cfile, "r")
    l1 = l2 = None
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        if ll[0] == "COATES_MACROPHAGE_M1_VS_M2_UP":
           l1 = ll[2:]
        if ll[0] == "COATES_MACROPHAGE_M1_VS_M2_DN":
           l2 = ll[2:]
    fp.close();
    res = [set(l1), set(l2)]
    return res

def getMartinez():
    #PMID: 17082649 DOI: 10.4049/jimmunol.177.10.7303
    cfile = "database/c7.all.v6.2.symbols.gmt"
    fp = open(cfile, "r")
    l1 = l2 = None
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        if ll[0] == "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_UP":
           l2 = ll[2:]
        if ll[0] == "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_DN":
           l1 = ll[2:]
    fp.close();
    res = [set(l1), set(l2)]
    return res

def getMartinezM2hsmm():
    cfile = "database/Martinez-Table3.txt"
    res = getEntries(cfile, 0)
    return [res[1:]]

def getMartinezM0hsmm():
    cfile = "database/Martinez-TableS1.txt"
    res = getEntries(cfile, 0)
    return [res[1:]]

def getBell2016():
# Citation: Bell LCK, Pollara G, Pascoe M, Tomlinson GS, Lehloenya RJ, Roe J, et
# al. (2016) In Vivo Molecular Dissection of the Effects of HIV-1 in Active
# Tuberculosis. PLoS Pathog 12(3): e1005469.
# https://doi.org/10.1371/journal.ppat.1005469
#PMID: 26986567 PMCID: PMC4795555 DOI: 10.1371/journal.ppat.1005469
    cfile = "database/journal.ppat.1005469.s018.txt"
    fp = open(cfile, "r")
    l1 = l2 = l3 = None
    for line in fp:
        line = line.strip();
        ll = line.split("\t");
        if ll[0] == "LPS":
           l1 = ll[2].replace('"', "").split(", ")
        if ll[0] == "IFNg":
           l2 = ll[2].replace('"', "").split(", ")
        if ll[0] == "IL-4 and IL-13":
           l3 = ll[2].replace('"', "").split(", ")
    fp.close();
    res = [set(l1), set(l2), set(l3)]
    return res

def getBecker():
    #Citation:
    #Becker M, De Bastiani MA, Parisi MM, Guma FT, Markoski MM, Castro
    #MA, Kaplan MH, Barbé-Tuana FM, Klamt F. Integrated Transcriptomics 
    #Establish Macrophage Polarization Signatures and have Potential 
    #Applications for Clinical Health and Disease. Sci Rep. 2015 Aug 25;
    #5:13351. doi: 10.1038/srep13351.
    #PMID: 26302899 PMCID: PMC4548187 DOI: 10.1038/srep13351
    cfile = "database/srep13351-s1-1.txt"
    l1 = getEntries(cfile, 0)
    cfile = "database/srep13351-s1-2.txt"
    l2 = getEntries(cfile, 0)
    return [l1[1:], l2[2:]]


class BIGraph:
    def readEqGraph(cfile):
        edges = {}
        nodes = {}
        count = 0
        with open(cfile, "r") as netFile:
            for ln in netFile:
                if (not ln.startswith("Found : 5")):
                    continue
                pln = ln.strip().split("\t")
                id1, id2 = pln[3], pln[4]
                nodes[id1] = 1
                nodes[id2] = 1
                if (id1 not in edges):
                    edges[id1] = {}
                count += 1
                edges[id1][id2] = 1
        print(str(count) + " edges Processed")
        return nodes, edges
    def gamma(u, edges, hash1):
        if (hash1 and u in hash1):
            return hash1[u]
        res = [u] + list(edges[u].keys())
        if hash1:
            hash1[u] = res
        return res
    def rho(u, v, edges, hash1, shash):
        if (shash and u in shash and v in shash[u]):
            return shash[u][v]
        gu = BIGraph.gamma(u, edges, hash1)
        gv = BIGraph.gamma(v, edges, hash1)
        g_union = set(gu).union(gv)
        g_int = set(gu).intersection(gv)
        res = 0
        if len(g_union) == 0:
            res = 0
        else:
            res = len(g_int) / len(g_union)
        if shash:
            shash[u][v] = res
        return res
    def pruneEqGraph(edges):
        from networkx.utils.union_find import UnionFind
        uf = UnionFind()
        hash1 = {}
        shash = {}
        num = len(edges)
        count = 0
        eqscores = []
        for u in edges:
            print(count, num, end='\r', flush=True)
            data = []
            scores = []
            for v in edges[u]:
                r = BIGraph.rho(u, v, edges, hash1, shash)
                data += [r]
                scores += [[r, v]]
                uf[u], uf[v]
            filter1 =  sorted(scores, reverse=True)
            for s in filter1:
                if (uf[u] != uf[s[1]]):
                    uf.union(u, s[1])
                    eqscores.append([u, s[0], s[1]])
                    #print(u, s[0], s[1])
                    break
            count += 1
        return pd.DataFrame(eqscores)

    def getClusters(df, thr=0.5):
        from networkx.utils.union_find import UnionFind
        uf = UnionFind()
        edges = {}
        for i in df.index:
            id1 = df[0][i]
            id2 = df[2][i]
            if id1 not in edges:
                edges[id1] = {}
            edges[id1][id1] = df[1][i] 
            if id2 not in edges:
                edges[id2] = {}
            edges[id2][id2] = df[1][i] 
            uf[id1], uf[id2]
            if (df[1][i] > thr):
                uf.union(id1, id2)
                edges[id1][id2] = df[1][i] 
                edges[id2][id1] = df[1][i]
        rank = {}
        for k in edges:
            rank[k] = len(edges[k])
        cls = {}
        for k in uf.to_sets():
            l = sorted(k, key=lambda x: rank[x], reverse=True)
            cls[l[0]] = [len(l), l]
        return cls

    def getNCodes(net, l):
        relations = {}
        for u in l:
            res = net.readBlockID(u)
            i = net.balancedhash[u]
            for j in range(len(net.balanced)):
                v = net.balanced[j]
                code = res[j]
                if code <= 0:
                    continue
                if code not in relations:
                    relations[code] = {}
                if v not in relations[code]:
                    relations[code][v] = 0
                relations[code][v] += 1
        return relations

    def getClustersGraph(net, cls):
        nodes = cls
        ids = [k for k in cls]
        eqgraph = []
        count = 0
        for u in ids:
            if (count % 100) == 0:
                print(count, end='\r', flush=True)
            nl = nodes[u][0]
            mid = int(nl / 2)
            l = [u]
            if (nl > 2):
                l = [u, nodes[u][1][1], nodes[u][1][mid]]
            if (nl > 10):
                l += [nodes[u][1][i] for i in [int(mid/4), int(mid/2), mid - 1]]
            ru = BIGraph.getNCodes(net, l)
            for c in range(1, 7):
                if c not in ru:
                    continue
                for v in ru[c]:
                    if v not in nodes:
                        continue
                    if (nl > 10 and ru[c][v] < 3):
                        continue
                    eqgraph.append([u, str(c), v, ru[c][v]])
            count += 1
        return pd.DataFrame(eqgraph)

    def readClustersGraph(cls, cg):
        edges = {}
        clusters = {}
        nodep = {}
        for u in cls:
            clusters[u] = cls[u][1]
            for k in cls[u][1]:
                nodep[k] = u
        print(str(len(clusters)) + " Processed")
        print("Processing edges...")
        count = 0
        for i in cg.index:
            if (count % 100) == 0:
                print(count, end='\r', flush=True)
            u, c, v, ru = cg.loc[i, :]
            if(u not in clusters or v not in clusters):
                continue
            if (u not in edges):
                edges[u] = {}
            if c not in edges[u]:
                edges[u][c] = {}
            count += 1
            edges[u][c][v] = 1
        print(str(count) + " edges processed")
        return edges, clusters, nodep

    def readClustersGraphFile(netprm):
        edges = {}
        clusters = {}
        nodep = {}
        with open(netprm + "-cls.txt", "r") as clusterFile:
            for ln in clusterFile:
                pln = ln.strip().split("\t")
                if(int(pln[1]) > 0): # min size threshold
                    clusters[pln[0]] = pln[2:]
                    for k in pln[2:]:
                        nodep[k] = pln[0]
        print(str(len(clusters)) + " Processed")
        print("Processing edges...")
        count = 0
        with open(netprm + "-eq-g.txt", "r") as edgeFile:
            for ln in edgeFile:
                if (count % 1000) == 0:
                    print(count, end='\r', flush=True)
                pln = ln.strip().split("\t")
                if(pln[0] not in clusters or
                   pln[2] not in clusters):
                    continue
                if (pln[0] not in edges):
                    edges[pln[0]] = {}
                if pln[1] not in edges[pln[0]]:
                    edges[pln[0]][pln[1]] = {}
                count += 1
                edges[pln[0]][pln[1]][pln[2]] = 1
        print(str(count) + " edges processed")
        return edges, clusters, nodep

    def getPath(G, s, p):
        visited = {}
        visited[s] = 1
        res = [s]
        l1 = list(G.neighbors(s))
        l1 = list(set(p).intersection(l1)) + list(set(l1).difference(p))
        while (len(l1) > 0):
            if l1[0] in visited:
                break
            visited[l1[0]] = 1
            res += [l1[0]]
            l1 = list(G.neighbors(l1[0]))
            l1 = list(set(p).intersection(l1)) + list(set(l1).difference(p))
        return res

    def getBINGraph(ana, edges, clusters, dr):
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout

        sclusters = {k:len(clusters[k]) for k in clusters}
        dclusters = {k:dr.loc[k] for k in clusters}
        def getW(k):
            return (-sclusters[k], -dclusters[k])
        def getS(k):
            return np.log(sclusters[k]+1)/np.log(1.5) + 2
        keys = sorted(sclusters, key=lambda k:getW(k))

        net = nx.DiGraph()
        for id1 in edges:
            if '2' in edges[id1]:
                l1 = sorted(edges[id1]['2'], key=lambda k:getW(k))
                for id2 in l1:
                    net.add_edge(id1, id2, rel='2')
            if 2 in edges[id1]:
                l1 = sorted(edges[id1][2], key=lambda k:getW(k))
                for id2 in l1:
                    net.add_edge(id1, id2, rel='2')

        G = nx.DiGraph()
        for id1 in keys[0:10]:
            l1 = []
            if id1 in edges and '4' in edges[id1]:
                l1 += list(edges[id1]['4'].keys())
            if id1 in edges and '6' in edges[id1]:
                l1 += list(edges[id1]['6'].keys())
            if id1 in edges and 4 in edges[id1]:
                l1 += list(edges[id1][4].keys())
            if id1 in edges and 6 in edges[id1]:
                l1 += list(edges[id1][6].keys())
            l1 = list(set(l1))
            l2 = sorted(l1, key=lambda k:getW(k))[0:2]
            print (sclusters[id1], dclusters[id1], l2)
            for id2 in l2:
                G.add_node(id1, label=ana.h.getSimpleName(id1), 
                           size=getS(id1), title='hilo', group=1)
                G.add_node(id2, label=ana.h.getSimpleName(id2),
                           size=getS(id2), title='hilo', group=1)
                G.add_edge(id1, id2, rel='4', color='red')

        l1 = list(G)
        for id1 in l1:
            l2 = BIGraph.getPath(net, id1, l1)
            print(l2)
            t1 = id1
            for id2 in l2[1:]:
                G.add_node(t1, label=ana.h.getSimpleName(t1), 
                           size=getS(t1), title='lolo', group=2)
                G.add_node(id2, label=ana.h.getSimpleName(id2),
                           size=getS(id2), title='lolo', group=2)
                G.add_edge(t1, id2, rel='2', color='blue',
                          arrows={'to':{'enabled':True, 'type':'arrow'}})
                t1 = id2
                
        return G

    def visualizeNetwork(G):
        from pyvis.network import Network
        nt = Network("500px", "500px", heading="BoNE", notebook=True)
        nt.from_nx(G)
        return nt.show("network/nx.html")

    def getBINGraphGML(ana, edges, clusters, dr):
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout

        sclusters = {k:len(clusters[k]) for k in clusters}
        dclusters = {k:dr.loc[k] for k in clusters}
        def getW(k):
            return (-sclusters[k], -dclusters[k])
        def getS(k):
            return np.log(sclusters[k]+1)/np.log(1.5) + 2
        keys = sorted(sclusters, key=lambda k:getW(k))

        net = nx.DiGraph()
        for id1 in edges:
            if '2' in edges[id1]:
                l1 = sorted(edges[id1]['2'], key=lambda k:getW(k))
                for id2 in l1:
                    net.add_edge(id1, id2, rel='2')
            if 2 in edges[id1]:
                l1 = sorted(edges[id1][2], key=lambda k:getW(k))
                for id2 in l1:
                    net.add_edge(id1, id2, rel='2')

        G = nx.DiGraph()
        for id1 in keys[0:10]:
            l1 = []
            if id1 in edges and '4' in edges[id1]:
                l1 += list(edges[id1]['4'].keys())
            if id1 in edges and '6' in edges[id1]:
                l1 += list(edges[id1]['6'].keys())
            if id1 in edges and 4 in edges[id1]:
                l1 += list(edges[id1][4].keys())
            if id1 in edges and 6 in edges[id1]:
                l1 += list(edges[id1][6].keys())
            l1 = list(set(l1))
            l2 = sorted(l1, key=lambda k:getW(k))[0:2]
            print (sclusters[id1], dclusters[id1], l2)
            for id2 in l2:
                G.add_node(id1, name=ana.h.getSimpleName(id1),
                           graphics = {'x': 0, 'y': 0, 'w': getS(id1), 'h': getS(id1),
                                       'type': 'ellipse', 'fill': '#889999', 'outline': '#666666',
                                       'outline_width': 1.0})
                G.add_node(id2, name=ana.h.getSimpleName(id2),
                           graphics = {'x': 0, 'y': 0, 'w': getS(id2), 'h': getS(id2),
                                       'type': 'ellipse', 'fill': '#889999', 'outline': '#666666',
                                       'outline_width': 1.0})
                G.add_edge(id1, id2, rel='4', 
                           graphics={'width': 1.0, 'fill': '#ff0000', 'type': 'line',
                                    'source_arrow': 0, 'target_arrow': 0})

        l1 = list(G)
        for id1 in l1:
            l2 = BIGraph.getPath(net, id1, l1)
            print(l2)
            t1 = id1
            for id2 in l2[1:]:
                G.add_node(t1, name=ana.h.getSimpleName(t1), 
                           graphics = {'x': 0, 'y': 0, 'w': getS(t1), 'h': getS(t1),
                                       'type': 'ellipse', 'fill': '#889999', 'outline': '#666666',
                                       'outline_width': 1.0})
                G.add_node(id2, name=ana.h.getSimpleName(id2),
                           graphics = {'x': 0, 'y': 0, 'w': getS(id2), 'h': getS(id2),
                                       'type': 'ellipse', 'fill': '#889999', 'outline': '#666666',
                                       'outline_width': 1.0})
                G.add_edge(t1, id2, rel='2', 
                          graphics={'width': 1.0, 'fill': '#0000ff', 'type': 'line',
                                    'source_arrow': 0, 'target_arrow': 1})
                t1 = id2
        return G

    def writeGML(G, ofile="network/nx.gml"):
        import networkx as nx
        nx.write_gml(G, ofile)

    def getDiff(ana, l1):
        res = []
        for k in l1:
            for u in ana.h.getIDs(k):
                expr = ana.h.getExprData(u)
                v1 = [float(expr[i]) for i in ana.state[0]]
                v2 = [float(expr[i]) for i in ana.state[1]]
                t, p = ttest_ind(v1,v2, equal_var=False)
                m1 = np.mean(v1)
                m2 = np.mean(v2)
                diff = m2 - m1
                res.append([k, u, m1, m2, diff, t, p])
        cl = ['Name', 'ID', 'm1', 'm2', 'diff', 't', 'p']
        return pd.DataFrame(res, columns=cl)

    def saveClusters(ofile, cls):
        fp = open(ofile, "w")
        for k in sorted(cls, key=lambda x: cls[x][0], reverse=True):
            l1 = [k, cls[k][0]] + cls[k][1]
            l1 = [str(k) for k in l1]
            fp.write("\t".join(l1) + "\n")
        fp.close()
        return
    def readClusters(cfile):
        cls = {}
        fp = open(cfile, "r")
        for line in fp:
            l1 = line.strip().split("\t")
            cls[l1[0]] = [int(l1[1]), l1[2:]]
        fp.close()
        return cls
    def getVolcano(ana, cfile, genes):
        h = ana.h
        list1 = []
        for g in genes:
            id1 = h.getBestID(h.getIDs(g).keys())
            if id1 is None:
                print (g)
                continue
            list1 += [id1]
        idlist = bone.getEntries(cfile, 0)
        idhash = {}
        for i in range(len(idlist)):
            idhash[idlist[i]] = i
        order = [idhash[k] for k in list1]
        pval = [float(i) for i in bone.getEntries(cfile, 3)]
        fc = [float(i) for i in bone.getEntries(cfile, 4)]
        c = ["black" for i in fc]
        df = pd.DataFrame()
        df["-log10(p)"] = -np.log10(pval)
        df["log(FC)"] = fc
        for i in range(len(pval)):
            if df["-log10(p)"][i] >= -np.log10(0.05) and abs(df["log(FC)"][i]) >= 1:
                c[i] = "green"
            if df["-log10(p)"][i] >= -np.log10(0.05) and abs(df["log(FC)"][i]) < 1:
                c[i] = "red"
            if df["-log10(p)"][i] < -np.log10(0.05) and abs(df["log(FC)"][i]) >= 1:
                c[i] = "orange"
        df["color"] = c
        df.dropna(inplace = True)
        ax = df.plot.scatter("log(FC)", "-log10(p)", c = df["color"],
                s=8, alpha=0.2, figsize=(6, 4))
        ax.set_xlim([-9, 9])
        ax.set_ylim([0, 9])
        for i in order:
            ax.text(fc[i], -np.log10(pval[i]),  h.getSimpleName(idlist[i]),
                        horizontalalignment='left', verticalalignment='top')
            ax.plot([fc[i]], [-np.log10(pval[i])], "bo", ms=4)
        return ax

class BINetwork:

    def __init__(self, filename):
        if not os.path.isfile(filename):
            print("Can't open file {0} <br>".format(filename));
            exit()
        self.fp = open(filename, "rb")
        self.file = filename
        self.magic = -1;
        self.major = -1;
        self.minor = -1;
        self.num = -1;
        self.numBits = -1;
        self.numBytes = -1;
        self.startPtr = -1;
        self.name = None;
        self.low = []
        self.high = []
        self.balanced = []
        self.balancedhash = {}

    def getCounts(res):
        count = [0, 0, 0, 0, 0, 0, 0] #count each relationships
        for k in res:
            count[k] += 1
        return count

    def getCountsString(res):
        count = BINetwork.getCounts(res)
        return ", ".join([str(k) for k in count])

    def readVal(buffer, i, nBits, debug=0):
        index = int(i * nBits/8)
        v1 = 0
        v2 = 0
        if index < len(buffer): 
            v1 = buffer[index]
        if (index + 1) < len(buffer): 
            v2 = buffer[index + 1]
        shift = (i * nBits) % 8
        mask = (1 << nBits) - 1
        val = ((v1 | v2 << 8) >> shift) & mask
        if (debug == 1):
            print(v1, v2, shift, mask, val)
        return val

    #
    # Codes for Boolean Relationships:
    # 0 - No relation
    # 1 - X low -> i high
    # 2 - X low -> i low
    # 3 - X high -> i high
    # 4 - X high -> i low
    # 5 - Equivalent
    # 6 - Opposite
    #
    # X - buffer1 and buffer2 corresponds to the two lines for probe X
    # i - probe i
    def readCode(buffer1, buffer2, i, numBits, debug=0):
        index = 2 * i * numBits;
        v1 = BINetwork.readVal(buffer1, index, numBits);
        v2 = BINetwork.readVal(buffer1, index + numBits, numBits);
        v3 = BINetwork.readVal(buffer2, index, numBits);
        v4 = BINetwork.readVal(buffer2, index + numBits, numBits);
        if (debug == 1):
            print (index, index + numBits, numBits, v1, v2, v3, v4)
        total = v1 + v2 + v3 + v4;
        if (total == 1):
            if (v1 == 1):
                return 1
            if (v2 == 1):
                return 2
            if (v3 == 1):
                return 3
            if (v4 == 1):
                return 4
        if (total == 2):
            if (v2 == 1 and v3 == 1):
                return 5
            if (v1 == 1 and v4 == 1):
                return 6
        return 0;

    def getLow(self):
        return self.readList(0)
    def getHigh(self):
        return self.readList(1)
    def getBalanced(self):
        return self.readList(2)

    def readList(self, num):
        fh = self.fp
        fh.seek(3 + num * 4)
        buffer = fh.read(4)
        ptr = array.array("I", buffer)[0]
        fh.seek(ptr)
        buffer = fh.read(4);
        length = array.array("I", buffer)[0]
        buffer = fh.read(length)
        name = buffer.decode('utf-8')
        buffer = fh.read(4);
        size = array.array("I", buffer)[0]
        res = []
        for i in range(size):
            buffer = fh.read(4);
            length = array.array("I", buffer)[0]
            buffer = fh.read(length)
            res += [buffer.decode('utf-8')]
        return res
    
    def init(self):
        fh = self.fp
        fh.seek(0)
        self.magic = array.array("B", fh.read(1))[0]
        self.major = array.array("B", fh.read(1))[0]
        self.minor = array.array("B", fh.read(1))[0]
        self.low = self.getLow()
        self.high = self.getHigh();
        self.balanced = self.getBalanced();
        self.balancedhash = {}
        for i in range(len(self.balanced)):
            self.balancedhash[self.balanced[i]] = i
        fh.seek(3 + 3 * 4)
        buffer = fh.read(4)
        ptr = array.array("I", buffer)[0]
        buffer = fh.read(4)
        self.num = array.array("I", buffer)[0]
        buffer = fh.read(4)
        self.numBits = array.array("I", buffer)[0]
        self.numBytes = int(self.num * self.numBits/8) + 1
        fh.seek(ptr)
        self.startPtr = ptr
        return

    def readBlock(self):
        fh = self.fp
        buffer1 = fh.read(self.numBytes)
        buffer2 = fh.read(self.numBytes)
        res = []
        for i in range(int(self.num/2)):
            code = BINetwork.readCode(buffer1, buffer2, i, self.numBits);
            res += [code]
        return res

    def readBlockIndex(self, a):
        ptr = self.startPtr + 2 * a * self.numBytes;
        self.fp.seek(ptr)
        return self.readBlock()

    def readBlockID(self, id1):
        if id1 in self.balancedhash:
            a = self.balancedhash[id1]
            return self.readBlockIndex(a)
        return None

    def printDetails(self):
        print("Network: ", self.file)
        print(self.magic, self.major, self.minor)
        print(self.num, self.numBits, self.numBytes)
        print ("Low(", len(self.low), "):")
        #print (" ".join(self.low))
        print ("High(", len(self.high), "):")
        #print (" ".join(self.high))
        print ("Balanced(", len(self.balanced), "):")
        #print (" ".join(self.balanced))
        for i in range(10):
            count = [0, 0, 0, 0, 0, 0, 0] #count each relationships
            res = self.readBlock(); #get all relationships code for one probe
            for k in res:
                count[k] += 1
            #print the counts
            print (self.balanced[i], ", ".join([str(k) for k in count]))    
            
class IBDAnalysis:

    def __init__(self, cf="/booleanfs2/sahoo/Hegemon/explore.conf"):
        self.db = hu.Database(cf);
        self.state = []
        self.params = {}
        self.otype = 0
        self.start = 2
        self.end = 2
        self.axes = []

    def addAxes(self, ax):
        self.axes += [ax]

    def aRange(self):
        return range(self.start, self.end + 1)

    def getTitle(self):
        title = self.name + " (" + self.source + "; n = " + str(self.num) + ")"
        return title

    def printInfo(self):
        print(self.name + " (n = " + str(self.num) + ")")
        url = "http://hegemon.ucsd.edu/Tools/explore.php?key=polyps&id="
        if self.dbid.startswith("NB"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=nb&id="
        if self.dbid.startswith("PLP"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=polyps&id="
        if self.dbid.startswith("CRC"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=colon&id="
        if self.dbid.startswith("MAC"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=mac&id="
        if self.dbid.startswith("MACV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=macv&id="
        if self.dbid.startswith("LIV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=liver&id="
        if self.dbid.startswith("G16"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=gbm&id="
        if self.dbid.startswith("GL"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=global&id="
        if self.dbid.startswith("GS"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=gastric&id="
        if self.dbid.startswith("AD"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=ad&id="
        if self.dbid.startswith("COV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=covid&id="
        if self.dbid.startswith("LU"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=lung&id="
        if self.dbid.startswith("HRT"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=heart&id="
        print(self.source + " " + url + self.dbid)
        print(len(self.order), [len(i) for i in self.state], \
                self.source, url + self.dbid, self.dbid)

    def prepareDataDf(self, dbid):
        self.dbid = dbid
        self.dataset = hu.getHegemonDataset(self.dbid)
        self.num = self.dataset[2]
        self.name = self.dataset[1]
        self.source = self.dataset[3]
        obj = hu.getHegemonPatientData(self.dbid, 'time')
        self.headers = obj[0]
        self.hhash = {}
        self.start = 2;
        self.end = len(self.headers) - 1
        for i in range(len(self.headers)):
            self.hhash[self.headers[i]] = i

    def prepareData(self, dbid, cfile =
            "/booleanfs2/sahoo/Hegemon/explore.conf"):
        self.db = hu.Database(cfile)
        self.dbid = dbid
        self.h = hu.Hegemon(self.db.getDataset(self.dbid))
        self.h.init()
        self.h.initPlatform()
        self.h.initSurv()
        self.num = self.h.getNum()
        self.start = self.h.getStart()
        self.end = self.h.getEnd()
        self.name = self.h.rdataset.getName()
        self.source = self.h.getSource()
        self.headers = self.h.headers
        self.hhash = {}
        for i in range(len(self.headers)):
            self.hhash[self.headers[i]] = i

    def initData(self, atype, atypes, ahash):
        for i in range(len(atypes)):
            ahash[atypes[i]] = i
        aval = [ahash[i] if i in ahash else None for i in atype]
        expg = [i for i in self.aRange() if aval[i] is not None]
        self.state = [[i for i in range(len(atype)) if aval[i] == k] 
                for k in range(len(atypes))]
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = expg
        self.printInfo()

    def getSurvName(self, name):
        return hu.getHegemonPatientData(self.dbid, name)[1]

    def convertMm(self, gene_groups, genes):
        self.gene_groups = getGroupsMm(gene_groups)
        self.genes = getGroupsMm([genes])[0]

    def orderData(self, gene_groups, weight):
        self.col_labels = [self.h.headers[i] for i in self.order]
        #ranks, row_labels, expr = getRanks(gene_groups, h)
        ranks, row_labels, row_ids, row_numhi, expr = getRanks2(gene_groups,
                self.h)
        self.f_ranks = mergeRanks(self.h.aRange(), self.h.start, ranks, weight)
        #i1 = getOrder(self.order, self.h.start, ranks, weight)
        arr = [self.f_ranks[i - self.h.start] for i in self.order]
        i1 = [self.order[i] for i in np.argsort(arr)]
        index = np.array([i - self.h.start for i in i1])
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        if len(self.ind_r) > 0:
            self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        else:
            self.data = np.array([expr[i] for i in self.ind_r])
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        self.otype = 1


    def orderData2(self, gene_groups, weight):
        self.col_labels = [self.h.headers[i] for i in self.order]
        #ranks, row_labels, expr = getRanks(gene_groups, h)
        ranks, row_labels, row_ids, row_numhi, expr = getRanks3(gene_groups,
                self.h, self.order)
        self.f_ranks = mergeRanks2(self.order, ranks, weight)
        i1 = [self.order[i] for i in np.argsort(self.f_ranks)]
        index = np.argsort(self.f_ranks)
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        self.otype = 2

    def orderDataDf(self, gene_groups, weight):
        data_e = []
        data_t = []
        for k in gene_groups:
            df_e = hu.getHegemonDataFrame(self.dbid, k, None)
            df_t = hu.getHegemonThrFrame(self.dbid, k)
            rhash = {}
            for i in range(df_t.shape[0]):
                rhash[df_t.iloc[i,0]] = i
            order = [rhash[df_e.iloc[i,0]] for i in range(df_e.shape[0])]
            df_t = df_t.reindex(order)
            df_t.reset_index(inplace=True)
            data_e.append(df_e)
            data_t.append(df_t)
        self.col_labels = self.headers[self.start:]
        if len(gene_groups) > 0:
            self.col_labels = data_e[0].columns[self.start:]
        self.chash = {}
        for i in range(len(self.col_labels)):
            self.chash[self.col_labels[i]] = i
        ranks, row_labels, row_ids, row_numhi, expr = getRanksDf(data_e, data_t)
        i1 = getOrder(self.order, self.start, ranks, weight)
        index = np.array([i - self.start for i in i1])
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        self.f_ranks = mergeRanks(range(self.start, len(self.headers)),
                self.start, ranks, weight)
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        self.otype = 1

    def saveData(self, ofile1, ofile2):
        saveHeatmapData(ofile1, self.row_labels, self.row_numhi,
                self.row_ids, self.index, self.expr)
        saveCData(ofile2, self.h, self.i1, self.f_ranks)

    def readData(self, file1, file2):
        row_labels, row_numhi, row_ids, data = readHeatmapData(file1)
        i1, f_ranks = readCData(file2)
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.f_ranks = f_ranks

    def printHeatmap(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax, divider = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        predicted = [1 if f_ranks[i - self.start] >= thr[0] - nm else 0 for i in i1]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i - self.start] >= thr[0] else 0 for i in i1]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i - self.start] >= thr[0] + nm else 0 for i in i1]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1, divider

    def printHeatmap2(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax, divider = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData([f_ranks[i - self.start] for i in i1])
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        predicted = [1 if f_ranks[i - self.start] >= thr[0] - nm else 0 for i in i1]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i - self.start] >= thr[0] else 0 for i in i1]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i - self.start] >= thr[0] + nm else 0 for i in i1]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1, divider

    def printHeatmap3(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax, divider = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        i2 = np.argsort(f_ranks)
        predicted = [1 if f_ranks[i] >= thr[0] - nm else 0 for i in i2]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i] >= thr[0] else 0 for i in i2]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i] >= thr[0] + nm else 0 for i in i2]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1, divider

    def printTitleBar(self, params):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotTitleBar(self.params['cval'], \
                self.params['atypes'], self.params)
        return ax

    def getMetrics(ana, actual = None, ahash = None):
        if actual is None:
            actual = [ana.aval[i] for i in ana.i1]
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
        res = None
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        res = "%.2f" % roc_auc
        return res

    def getMetrics2(ana, actual, ahash = None, fthr = None):
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
        res = [None, None, None]
        for i in range(3):
            fpr, tpr, thrs = roc_curve(actual, score, pos_label=i)
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                roc_auc = 1 - roc_auc
            res[i] = "%.2f" % roc_auc
        thr = hu.getThrData(score[:-2])
        nm = (np.max(score) - np.min(score))/15
        #print(thr, nm)
        if fthr is None or fthr == "thr1":
            fthr = thr[0]
        if fthr == "thr0":
            fthr = thr[0] - nm
        if fthr == "thr2":
            fthr = thr[0] + nm
        if fthr == "thr3":
            fthr = thr[0] + 3 * nm
        predicted = [1 if ana.f_ranks[i - ana.start] >= fthr else 0 for i in ana.i1]
        if "predicted" in ana.params:
            predicted = ana.params["predicted"]
        print(list(actual))
        print(list(predicted))
        res[1] = "%.2f" % accuracy_score(actual, predicted)
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        tab = pd.crosstab(df.y > 0, df.x > 0)
        if tab.shape == (2, 2):
            res[2] = "%.3g" % fisher_exact(tab)[1]
        else:
            res[2] = "1.0"
        return res

    def densityPlot(self, ax=None, color = None):
        if (color is None):
            color = acolor
        ax = plotDensity(self.cval[0], self.atypes, ax, color)
        self.addAxes(ax)
        return ax

    def getScores(ana, ahash = None):
        lval = [[] for i in ana.atypes]
        cval = ana.cval[0]
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
                cval = [ana.cval[0][i] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
            for i in range(len(cval)):
                lval[cval[i]] += [score[i]]
        return lval, score

    def printAllPvals(self, ahash = None, params = None):
        lval, score = self.getScores(ahash=ahash)
        cAllPvals(lval, self.atypes)

    def getPvals(self, label, ahash = None, params = None):
        lval, score = self.getScores(ahash=ahash)
        actual = [self.aval[i] for i in self.i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(score)
        nm = (np.max(score) - np.min(score))/15
        i2 = np.argsort(score)
        predicted = [1 if score[i] >= thr[0] - nm else 0 for i in i2]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if score[i] >= thr[0] else 0 for i in i2]  
            if self.params["thr"] == 2:
                predicted = [1 if score[i] >= thr[0] + nm else 0 for i in i2]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        pval_fe = getFisher(predicted, actual);
        t, p = ttest_ind(lval[0],lval[label], equal_var=False)
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=label)
        roc_auc = auc(fpr, tpr)
        desc = "%.3g, %.3g, %.3g" % (pval_fe, p, roc_auc)
        return desc

    def printScores(self, ahash = None, params = None):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        if params is not None:
            self.params.update(params)
        lval, score = self.getScores(ahash=ahash)
        atypes = self.params['atypes']
        atypes = [str(atypes[i]) + "("+str(len(lval[i]))+")"
                for i in range(len(atypes))]
        ax,bp = plotScores(lval, atypes, self.params)
        ax.text(ax.get_xlim()[1], ax.get_ylim()[1], self.source,
                horizontalalignment='left', verticalalignment='center')
        if ('vert' not in self.params or  self.params['vert'] == 0):
            for i in range(1, len(lval)):
                desc = self.getPvals(i, ahash, params)
                ax.text(ax.get_xlim()[1], i + 1, desc,
                        horizontalalignment='left', verticalalignment='center')
        self.addAxes(ax)
        self.addAxes(bp)
        return ax,bp

    def printViolin(self, ahash = None, params = None):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        if params is not None:
            self.params.update(params)
        lval, score = self.getScores(ahash=ahash)
        df = pd.DataFrame()
        df['score'] = score
        atypes = [str(self.atypes[i]) + "("+str(len(lval[i]))+")"
                for i in range(len(self.atypes))]
        df['category'] = [atypes[self.aval[i]] for i in self.i1]
        m1 = []
        pvals = []
        for i in range(1, len(lval)):
            if len(lval[i]) <= 0:
                pvals += [""]
                m1 += [0.1]
                continue
            m1 += [max(lval[i]) + (max(lval[i]) - min(lval[i])) * 0.1]
            t, p = ttest_ind(lval[0],lval[i], equal_var=False)
            if (p < 0.05):
                pvals += ["p=%.3g" % p]
            else:
                pvals += [""]
        params = self.params
        dpi = 100
        if 'dpi' in params:
            dpi = params['dpi']
        w,h = (1.5 * len(self.atypes), 4)
        if 'w' in params:
            w = params['w']
        if 'h' in params:
            h = params['h']
        color_sch1 = acolor
        if 'acolor' in params:
            color_sch1 = params['acolor']
        ax = None
        if 'ax' in params:
            ax = params['ax']
        if ax is None:
            fig,ax = plt.subplots(figsize=(w,h), dpi=dpi)
        sns.set()
        sns.set_style("white")
        sns.set_style({'xtick.color':'.5', 'ytick.color':'.5', 'axes.labelcolor': '.5'})
        sns.set_context("notebook")
        sns.set_palette([adj_light(c, 1.5, 1) for c in color_sch1])
        width = 1
        height = 1
        if 'width' in params:
            width = params['width']
        if 'vert' in params and params['vert'] == 1:
            ax = sns.violinplot(x="category", y="score", inner='quartile',
                    linewidth=0.5, width=width, ax = ax, data=df,
                    order = atypes)
            ax = sns.swarmplot(x="category", y="score", color = 'blue', alpha=0.2,
                    ax=ax, data=df, order = atypes)
            ax.set_xlabel("")
            pos = range(len(atypes))
            for tick,label in zip(pos[1:],ax.get_xticklabels()[1:]):
                ax.text(pos[tick], m1[tick - 1], pvals[tick - 1],
                        horizontalalignment='center', size=12,
                        color='0.3')
            ax.yaxis.grid(True, clip_on=False)
        else:
            ax = sns.violinplot(x="score", y="category", inner='quartile',
                    linewidth=0.5, width=width, ax = ax, data=df,
                    order = atypes)
            ax = sns.swarmplot(x="score", y="category", color = 'blue', alpha=0.2,
                    ax=ax, data=df, order = atypes)
            ax.set_ylabel("")
            pos = range(len(atypes))
            for tick,label in zip(pos[1:],ax.get_yticklabels()[1:]):
                ax.text(m1[tick - 1], pos[tick]-0.5, pvals[tick - 1],
                        horizontalalignment='center', size=12,
                        color='0.3')
            ax.xaxis.grid(True, clip_on=False)
        return ax

    def printGene(self, name, ahash = None, params = None):
        self.params = {'atypes': self.atypes, 'vert':1}
        if params is not None:
            self.params.update(params)
        id1 = self.h.getBestID(self.h.getIDs(name).keys())
        expr = self.h.getExprData(id1)
        if expr is None:
            print("Not Found")
            return None, None
        lval = [[] for i in self.params['atypes']]
        aval = self.aval
        if ahash is not None:
            aval = [ahash[i] if i in ahash else None for i in self.atype]
        for i in self.h.aRange():
            if aval[i] is None:
                continue
            lval[aval[i]] += [float(expr[i])]
        atypes = [str(self.params['atypes'][i]) + "("+str(len(lval[i]))+")"
                for i in range(len(self.params['atypes']))]
        if 'violin' in self.params and self.params['violin'] == 1:
            ax = plotViolin(lval, atypes, self.params)
            bp = None
        else:
            ax,bp = plotScores(lval, atypes, self.params)
        if self.params['vert'] == 0:
            ax.text(ax.get_xlim()[1], 1, self.h.getSource(),
                    horizontalalignment='left', verticalalignment='center')
        else:
            title = self.h.rdataset.getName() + " (" + self.h.getSource() + "; n = " + str(self.h.getNum()) + ")"
            ax.set_title(title)
            ax.set_ylabel(self.h.getSimpleName(id1))
        self.addAxes(ax)
        self.addAxes(bp)
        cAllPvals(lval, self.params['atypes'])
        return ax,bp

    def getROCAUCspecific(ana, m=0, n=1):
        actual = [ana.aval[i] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        score = [ana.f_ranks[i - ana.start] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=n)
        roc_auc = auc(fpr, tpr)
        return "%.2f" % roc_auc

    def getROCAUC(ana):
        res = []
        for k in range(1, len(ana.atypes)):
            v = ana.getROCAUCspecific(0, k)
            res += [v]
        return ",".join(res)

    def getTPvals(ana):
        res = []
        lval, score = ana.getScores()
        for k in range(1, len(ana.atypes)):
            t, p = ttest_ind(lval[0],lval[k], equal_var=False)
            res += ["%.3g" % p]
        return ",".join(res)

    def getStats(self, l1, wt1, annotation=[]):
        src = re.sub(" .*", "", self.source)
        species = annotation[1]
        if species == 'Hs' or species == 'Rm' :
            self.orderData(l1, wt1)
        else:
            genes = []
            self.convertMm(l1, genes)
            self.orderData(self.gene_groups, wt1)
        roc = self.getROCAUC()
        p = self.getTPvals()
        lval, score = self.getScores()
        n1 = np.sum([len(lval[i])for i in range(1, len(lval))])
        return [src, roc, p, len(lval[0]), n1] + annotation
        
    def getSurvival(self, dbid = "CRC35.3"):
        self.prepareData(dbid)
        atype = self.h.getSurvName("status")
        atypes = ['Censor', 'Relapse']
        ahash = {"0": 0, "1":1}
        self.initData(atype, atypes, ahash)

    def getSurvivalDf(self, dbid = "CRC35.3"):
        self.prepareDataDf(dbid)
        atype = self.getSurvName("status")
        atypes = ['Censor', 'Relapse']
        ahash = {"0": 0, "1":1}
        self.initData(atype, atypes, ahash)

    def printSurvival(self, fthr = None, pG = None, genex = "CDX2",
            ct = None, ax = None):
        f_ranks = self.f_ranks
        order = self.order
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/16
        if fthr is None:
            fthr = thr[0]
        if fthr == "thr0":
            fthr = thr[0] - nm
        if fthr == "thr2":
            fthr = thr[0] + nm
        if fthr == "thr3":
            fthr = thr[0] + 3 * nm
        print(thr)
        print(nm, fthr)
        g1 = [i for i in order if f_ranks[i - self.start] < fthr]
        g2 = [i for i in order if f_ranks[i - self.start] >= fthr]
        if pG is None:
            pG = [ ["Low", "green", g1], ["High", "red", g2]]
        obj = hu.getHegemonPatientData(self.dbid, 'time')
        time = obj[1]
        obj = hu.getHegemonPatientData(self.dbid, 'status')
        status = obj[1]
        if ct is not None:
            time, status = hu.censor(time, status, ct)
        sax = hu.survival(time, status, pG, ax)
        df = pd.DataFrame()
        df["f_ranks"] = pd.to_numeric(pd.Series(f_ranks))
        e = self.h.getExprData(genex)
        df[genex] = pd.to_numeric(pd.Series(e[2:]))
        ax = df.plot.scatter(x=genex, y='f_ranks')
        return sax, ax

    def printCoxSurvival(ana, fthr = None):
        score = [ana.f_ranks[i - ana.start] for i in ana.h.aRange()]
        thr = hu.getThrData(score)
        nm = (np.max(ana.f_ranks) - np.min(ana.f_ranks))/16
        if fthr is None:
            fthr = thr[0]
        if fthr == "thr0":
            fthr = thr[0] - nm
        if fthr == "thr2":
            fthr = thr[0] + nm
        if fthr == "thr3":
            fthr = thr[0] + 3 * nm
        print(thr)
        print(nm, fthr)
        c1 = ["", ""] + [1 if ana.f_ranks[i - ana.start] >= fthr else 0 for i in
                ana.h.aRange()]
        obj = hu.getHegemonPatientData(ana.dbid, 'time')
        time = obj[1]
        obj = hu.getHegemonPatientData(ana.dbid, 'status')
        status = obj[1]
        mdata = {"time": time, "status": status, "c1" : c1}
        df = pd.DataFrame(mdata)
        df.drop([0, 1], inplace=True)
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace=True)

        import rpy2.robjects as ro
        ro.r('library(survival)')
        ro.r("time <- c(" + ",".join(df['time']) + ")")
        ro.r("status <- c(" + ",".join(df['status']) + ")")
        ro.r("c1 <- c(" + ",".join([str(i) for i in df['c1']]) + ")")
        ro.r('x <- coxph(Surv(time, status) ~ c1)')
        ro.r('s <- summary(x)')
        print(ro.r('s'))
        t = [float(i) for i in df['time']]
        s = [int(i) for i in df['status']]
        g = [int(i) for i in df['c1']]
        from lifelines.statistics import multivariate_logrank_test
        from matplotlib.legend import Legend
        res = multivariate_logrank_test(t, g, s)
        print("p = %.2g" % res.p_value)

    def printMeanAbsoluteDeviation(ana, ofile):
        from scipy.stats import median_absolute_deviation
        fp = ana.h.fp;
        fp.seek(0, 0);
        head = fp.readline();
        of = open(ofile, "w")
        of.write("\t".join(["ArrayID", "MAD"]) + "\n")
        index = 0
        for line in fp:
          line = re.sub("[\r\n]", "", line)
          ll = line.split("\t")
          if len([i for i in ana.order if ll[i] == '']) > 0:
              continue
          v1 = [float(ll[i]) for i in ana.order]
          mad = median_absolute_deviation(v1)
          of.write("\t".join([ll[0], str(mad)]) +"\n")
          index += 1
        of.close()              
            

class MacAnalysis:

    def __init__(self):
        self.db = hu.Database("/booleanfs2/sahoo/Hegemon/explore.conf")
        self.normal = []
        self.uc = []
        self.cd = []
        self.ibd = []
        self.st1 = []
        self.st2 = []
        self.st3 = []
        self.otype = 0
        self.axes = []

    def addAxes(self, ax):
        self.axes += [ax]

    def printInfo(self):
        print(self.h.rdataset.getName() + " (n = " + str(self.h.getNum()) + ")")
        if len(self.ibd) > 0:
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=polyps&id="
            if self.dbid.startswith("CRC"):
                url = "http://hegemon.ucsd.edu/Tools/explore.php?key=colon&id="
            print(len(self.order), len(self.normal), \
                    len(self.uc), len(self.cd), self.h.getSource(), \
                    url + self.dbid, self.dbid)
            return
        url = "http://hegemon.ucsd.edu/Tools/explore.php?key=blood:leukemia&id="
        if self.dbid.startswith("PLP"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=polyps&id="
        if self.dbid.startswith("CRC"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=colon&id="
        if self.dbid.startswith("MAC"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=mac&id="
        if self.dbid.startswith("MACV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=macv&id="
        if self.dbid.startswith("LIV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=liver&id="
        if self.dbid.startswith("G16"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=gbm&id="
        if self.dbid.startswith("GL"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=global&id="
        if self.dbid.startswith("GS"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=gastric&id="
        if self.dbid.startswith("AD"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=ad&id="
        if self.dbid.startswith("COV"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=covid&id="
        if self.dbid.startswith("LU"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=lung&id="
        if self.dbid.startswith("HRT"):
            url = "http://hegemon.ucsd.edu/Tools/explore.php?key=heart&id="
        print(self.h.getSource() + " " + url + self.dbid)
        print(len(self.order), len(self.st1), len(self.st2), len(self.st3), self.dbid)

    def convertMm(self, gene_groups, genes):
        self.gene_groups = getGroupsMm(gene_groups)
        self.genes = getGroupsMm([genes])[0]

    def orderData(self, gene_groups, weight):
        self.col_labels = [self.h.headers[i] for i in self.order]
        #ranks, row_labels, expr = getRanks(gene_groups, h)
        ranks, row_labels, row_ids, row_numhi, expr = getRanks2(gene_groups,
                self.h)
        self.f_ranks = mergeRanks(self.h.aRange(), self.h.start, ranks, weight)
        #i1 = getOrder(self.order, self.h.start, ranks, weight)
        arr = [self.f_ranks[i - self.h.start] for i in self.order]
        i1 = [self.order[i] for i in np.argsort(arr)]
        index = np.array([i - self.h.start for i in i1])
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        if len(self.ind_r) > 0:
            self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        else:
            self.data = np.array([expr[i] for i in self.ind_r])
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        self.otype = 1

    def orderData2(self, gene_groups, weight):
        self.col_labels = [self.h.headers[i] for i in self.order]
        #ranks, row_labels, expr = getRanks(gene_groups, h)
        ranks, row_labels, row_ids, row_numhi, expr = getRanks3(gene_groups,
                self.h, self.order)
        self.f_ranks = mergeRanks2(self.order, ranks, weight)
        i1 = [self.order[i] for i in np.argsort(self.f_ranks)]
        index = np.argsort(self.f_ranks)
        self.cval = np.array([[self.aval[i] for i in i1]])
        #self.data = np.array(expr)[:,index]
        self.ind_r = np.array(sorted(range(len(row_labels)),
            key=lambda x: (row_numhi[x][0], row_numhi[x][1])))
        row_labels = [row_labels[i] for i in self.ind_r]
        row_ids = [row_ids[i] for i in self.ind_r]
        self.data = np.array([expr[i] for i in self.ind_r])[:,index]
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.index = index
        self.otype = 2

    def saveData(self, ofile1, ofile2):
        saveHeatmapData(ofile1, self.row_labels, self.row_numhi,
                self.row_ids, self.index, self.expr)
        saveCData(ofile2, self.h, self.i1, self.f_ranks)

    def readData(self, file1, file2):
        row_labels, row_numhi, row_ids, data = readHeatmapData(file1)
        i1, f_ranks = readCData(file2)
        self.ranks = ranks
        self.row_labels = row_labels
        self.row_ids = row_ids
        self.row_numhi = row_numhi
        self.expr = expr
        self.i1 = i1
        self.f_ranks = f_ranks

    def printHeatmap(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        self.addAxes(ax)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        predicted = [1 if f_ranks[i - self.h.start] >= thr[0] - nm else 0 for i in i1]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i - self.h.start] >= thr[0] else 0 for i in i1]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i - self.h.start] >= thr[0] + nm else 0 for i in i1]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        self.addAxes(ax1)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        if tab.shape == (2,2):
            print(fisher_exact(tab))
            print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1

    def printHeatmap2(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        self.addAxes(ax)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData([f_ranks[i - self.h.start] for i in i1])
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        predicted = [1 if f_ranks[i - self.h.start] >= thr[0] - nm else 0 for i in i1]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i - self.h.start] >= thr[0] else 0 for i in i1]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i - self.h.start] >= thr[0] + nm else 0 for i in i1]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        self.addAxes(ax1)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1

    def printHeatmap3(self, ofile, genes, params):
        i1 = self.i1
        f_ranks = self.f_ranks
        self.params = {'genes': genes,'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotHeatmap(ofile, self.data, self.col_labels, 
                self.row_labels, self.params)
        self.addAxes(ax)
        actual = [1 if self.aval[i] >= 1 else 0 for i in i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/15
        i2 = np.argsort(f_ranks)
        predicted = [1 if f_ranks[i] >= thr[0] - nm else 0 for i in i2]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if f_ranks[i] >= thr[0] else 0 for i in i2]  
            if self.params["thr"] == 2:
                predicted = [1 if f_ranks[i] >= thr[0] + nm else 0 for i in i2]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        target_names = ['Normal', 'IBD']
        score=convertScore(predicted)
        print(list(actual))
        print(list(predicted))
        ax1 = printReport(actual, predicted, score, target_names)
        self.addAxes(ax1)
        tab = pd.crosstab(df.y > 0, df.x > 0)
        print(tab)
        print(fisher_exact(tab))
        print('Fisher Exact pvalue =', fisher_exact(tab)[1])
        return ax, ax1

    def printTitleBar(self, params):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        self.params.update(params)
        ax = plotTitleBar(self.params['cval'], \
                self.params['atypes'], self.params)
        ax.text(len(self.params['cval'][0]), 0, self.h.getSource())
        self.addAxes(ax)
        return ax

    def normMacrophageGene(self, gene, params):
        exp1 = self.h.getExprData(gene)
        thr1 = self.h.getThrData(gene)
        if self.otype == 2:
            tdata = [[float(exp1[self.order[i]]), float(self.f_ranks[i])] \
                    for i in range(len(self.order))]
        else:
            tdata = [[float(exp1[i]), self.f_ranks[i - self.h.start]] \
                    for i in self.h.aRange()]

        data = pd.DataFrame(tdata, columns=["x", "y"])
        from sklearn.linear_model import LinearRegression
        linreg = LinearRegression(normalize=True)
        if "select" in params and params["select"] == 2:
            return
        if "select" in params and params["select"] == 1 and self.otype != 2:
            tdata = [[float(exp1[i]), self.f_ranks[i - self.h.start]] \
                    for i in self.order]
            df = pd.DataFrame(tdata, columns=["x", "y"])
            linreg.fit(df['x'].reshape(-1, 1),df['y'])
            y_pred = linreg.predict(data['x'].reshape(-1, 1))
        else:
            linreg.fit(data['x'].reshape(-1, 1),data['y'])
            y_pred = linreg.predict(data['x'].reshape(-1, 1))
        thr = 0
        if "thr" in params:
            if params["thr"] == 0:
                thr = thr1[2]
            if params["thr"] == 1:
                thr = thr1[0]
            if params["thr"] == 2:
                thr = thr1[3]
            if params["thr"] == 3:
                thr = 0
            if params["thr"] == 4 and "tval" in params:
                thr = params["tval"]
        def getCol(data, y_pred, i, thr):
            if data['x'][i] <= thr:
                return 'grey'
            if data['y'][i] >= y_pred[i]:
                return 'green'
            return 'red'
        if self.otype == 2:
            col = [getCol(data, y_pred, i, thr) for i in range(len(self.order))]
        else:
            col = [getCol(data, y_pred, i - self.h.start, thr) for i in self.h.aRange()]
            col = []
            for i in self.h.aRange():
                if i in self.order:
                    col.extend([getCol(data, y_pred, i - self.h.start, thr)])
                else:
                    col.extend(["#CCCCCC"])
        ax = data.plot.scatter(x='x', y='y', c=col)
        self.addAxes(ax)
        ax.plot(data['x'], y_pred, c='Orange')
        ax.set_xlabel(gene)
        ax.set_ylabel("Gene Signature")
        df = pd.DataFrame()
        df['x'] = data['x']
        df['y'] = data['y']
        df['y1'] = (data['y'] - y_pred)
        ax = df.plot.scatter(x='x', y='y1', c=col)
        self.addAxes(ax)
        ax.set_xlabel(gene)
        ax.set_ylabel("Gene Signature")
        if self.otype == 2:
            i3 = [i for i in range(len(self.order)) if df['x'][i] > thr]
            self.f_ranks = [df['y1'][i] for i in i3]
            self.order = [self.order[i] for i in i3]
            self.i1 = [self.order[i] for i in np.argsort(self.f_ranks)]
            index = [i3[i] for i in np.argsort(self.f_ranks)]
        else:
            self.f_ranks = df['y1']
            self.order = [i for i in self.order if df['x'][i - self.h.start] > thr]
            arr = [self.f_ranks[i - self.h.start] for i in self.order]
            self.i1 = [self.order[i] for i in np.argsort(arr)]
            index = np.array([i - self.h.start for i in self.i1])
        self.cval = np.array([[self.aval[i] for i in self.i1]])
        self.data = np.array([self.expr[i] for i in self.ind_r])[:,index]

    def normMacrophage(self, params):
        return self.normMacrophageGene("TYROBP", params)

    def getMetrics(ana, actual, ahash = None):
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.h.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.h.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
        res = [None, None, None]
        for i in range(3):
            fpr, tpr, thrs = roc_curve(actual, score, pos_label=i)
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                roc_auc = 1 - roc_auc
            res[i] = "%.2f" % roc_auc
        return res

    def getMetrics2(ana, actual, ahash = None, fthr = None):
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.h.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.h.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
        res = [None, None, None]
        for i in range(3):
            fpr, tpr, thrs = roc_curve(actual, score, pos_label=i)
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                roc_auc = 1 - roc_auc
            res[i] = "%.2f" % roc_auc
        thr = hu.getThrData(score[:-2])
        nm = (np.max(score) - np.min(score))/15
        #print(thr, nm)
        if fthr is None:
            fthr = thr[0]
        if fthr == "thr0":
            fthr = thr[0] - nm
        if fthr == "thr2":
            fthr = thr[0] + nm
        if fthr == "thr3":
            fthr = thr[0] + 3 * nm
        predicted = [1 if ana.f_ranks[i - ana.h.start] >= fthr else 0 for i in ana.i1]
        print(actual)
        print(predicted)
        res[1] = "%.2f" % accuracy_score(actual, predicted)
        data_list = {'x' : predicted}
        df = pd.DataFrame(data_list)
        df['y'] = pd.Series(np.array(actual))
        tab = pd.crosstab(df.y > 0, df.x > 0)
        res[2] = "%.3g" % fisher_exact(tab)[1]
        return res

    def getMetrics3(ana, actual = None, ahash = None):
        if actual is None:
            actual = [ana.aval[i] for i in ana.i1]
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.h.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.h.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
        res = None
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        res = "%.2f" % roc_auc
        return res

    def densityPlot(self, ax=None, color = None):
        if (color is None):
            color = acolor
        ax = plotDensity(self.cval[0], self.atypes, ax, color)
        self.addAxes(ax)
        return ax

    def getScores(ana, ahash = None):
        lval = [[] for i in ana.atypes]
        cval = ana.cval[0]
        if ana.otype == 2:
            score = [ana.f_ranks[i] for i in ana.i1]
        else:
            if ahash is None:
                score = [ana.f_ranks[i - ana.h.start] for i in ana.i1]
            else:
                score = [ana.f_ranks[ana.i1[i] - ana.h.start] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
                cval = [ana.cval[0][i] \
                        for i in range(len(ana.i1)) \
                        if ana.cval[0][i] in  ahash ]
            for i in range(len(cval)):
                lval[cval[i]] += [score[i]]
        return lval, score

    def printAllPvals(self, ahash = None, params = None):
        lval, score = self.getScores(ahash=ahash)
        cAllPvals(lval, self.atypes)

    def getPvals(self, label, ahash = None, params = None):
        lval, score = self.getScores(ahash=ahash)
        actual = [self.aval[i] for i in self.i1]
        if "actual" in self.params:
            actual = self.params["actual"]
        thr = hu.getThrData(score)
        nm = (np.max(score) - np.min(score))/15
        i2 = np.argsort(score)
        predicted = [1 if score[i] >= thr[0] - nm else 0 for i in i2]
        if "thr" in self.params:
            if self.params["thr"] == 1:
                predicted = [1 if score[i] >= thr[0] else 0 for i in i2]  
            if self.params["thr"] == 2:
                predicted = [1 if score[i] >= thr[0] + nm else 0 for i in i2]  
        if "predicted" in self.params:
            predicted = self.params["predicted"]
        pval_fe = getFisher(predicted, actual);
        t, p = ttest_ind(lval[0],lval[label], equal_var=False)
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=label)
        roc_auc = auc(fpr, tpr)
        desc = "%.3g, %.3g, %.3g" % (pval_fe, p, roc_auc)
        return desc

    def printScores(self, ahash = None, params = None):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        if params is not None:
            self.params.update(params)
        lval, score = self.getScores(ahash=ahash)
        ax,bp = plotScores(lval, self.params['atypes'], self.params)
        ax.text(ax.get_xlim()[1], ax.get_ylim()[1], self.h.getSource(),
                horizontalalignment='left', verticalalignment='center')
        if ('vert' not in self.params or  self.params['vert'] == 0):
            for i in range(1, len(lval)):
                desc = self.getPvals(i, ahash, params)
                ax.text(ax.get_xlim()[1], i + 1, desc,
                        horizontalalignment='left', verticalalignment='center')
        self.addAxes(ax)
        self.addAxes(bp)
        return ax,bp

    def printViolin(self, ahash = None, params = None):
        self.params = {'atypes': self.atypes,'cval': self.cval}
        if params is not None:
            self.params.update(params)
        lval, score = self.getScores(ahash=ahash)
        df = pd.DataFrame()
        df['score'] = score
        atypes = [str(self.atypes[i]) + "("+str(len(lval[i]))+")"
                for i in range(len(self.atypes))]
        df['category'] = [atypes[self.aval[i]] for i in self.i1]
        m1 = []
        pvals = []
        for i in range(1, len(lval)):
            if len(lval[i]) <= 0:
                m1 += [0]
                pvals += [""]
                continue
            m1 += [max(lval[i]) + (max(lval[i]) - min(lval[i])) * 0.1]
            t, p = ttest_ind(lval[0],lval[i], equal_var=False)
            if (p < 0.05):
                pvals += ["p=%.3g" % p]
            else:
                pvals += [""]
        params = self.params
        dpi = 100
        if 'dpi' in params:
            dpi = params['dpi']
        w,h = (1.5 * len(self.atypes), 4)
        if 'w' in params:
            w = params['w']
        if 'h' in params:
            h = params['h']
        color_sch1 = acolor
        if 'acolor' in params:
            color_sch1 = params['acolor']
        sns.set()
        sns.set_style("white")
        sns.set_style({'xtick.color':'.5', 'ytick.color':'.5', 'axes.labelcolor': '.5'})
        sns.set_context("notebook")
        sns.set_palette([adj_light(c, 1.5, 1) for c in color_sch1])
        ax = None
        if 'ax' in params:
            ax = params['ax']
        if ax is None:
            fig,ax = plt.subplots(figsize=(w,h), dpi=dpi)
        width = 1
        height = 1
        if 'width' in params:
            width = params['width']
        if 'vert' in params and params['vert'] == 1:
            ax = sns.violinplot(x="category", y="score", inner='quartile',
                    linewidth=0.5, width=width, ax = ax, data=df,
                    order = atypes)
            ax = sns.swarmplot(x="category", y="score", color = 'blue', alpha=0.2,
                    ax=ax, data=df, order = atypes)
            ax.set_xlabel("")
            pos = range(len(atypes))
            for tick,label in zip(pos[1:],ax.get_xticklabels()[1:]):
                ax.text(pos[tick], m1[tick - 1], pvals[tick - 1],
                        horizontalalignment='center', size=12,
                        color='0.3')
            ax.yaxis.grid(True, clip_on=False)
        else:
            ax = sns.violinplot(x="score", y="category", inner='quartile',
                    linewidth=0.5, width=width, ax = ax, data=df,
                    order = atypes)
            ax = sns.swarmplot(x="score", y="category", color = 'blue', alpha=0.2,
                    ax=ax, data=df, order = atypes)
            ax.set_ylabel("")
            pos = range(len(atypes))
            for tick,label in zip(pos[1:],ax.get_yticklabels()[1:]):
                ax.text(m1[tick - 1], pos[tick]-0.5, pvals[tick - 1],
                        horizontalalignment='center', size=12,
                        color='0.3')
            ax.xaxis.grid(True, clip_on=False)
        self.addAxes(ax)
        return ax

    def printGene(self, name, ahash = None, params = None):
        self.params = {'atypes': self.atypes, 'vert':1}
        if params is not None:
            self.params.update(params)
        id1 = self.h.getBestID(self.h.getIDs(name).keys())
        expr = self.h.getExprData(id1)
        if expr is None:
            print("Not Found")
            return None, None
        lval = [[] for i in self.params['atypes']]
        aval = self.aval
        if ahash is not None:
            aval = [ahash[i] if i in ahash else None for i in self.atype]
        for i in self.h.aRange():
            if aval[i] is None:
                continue
            lval[aval[i]] += [float(expr[i])]
        if 'violin' in self.params and self.params['violin'] == 1:
            ax = plotViolin(lval, self.params['atypes'], self.params)
            bp = None
        else:
            ax,bp = plotScores(lval, self.params['atypes'], self.params)
        if self.params['vert'] == 0:
            ax.text(ax.get_xlim()[1], 1, self.h.getSource(),
                    horizontalalignment='left', verticalalignment='center')
        else:
            title = self.h.rdataset.getName() + " (" + self.h.getSource() + "; n = " + str(self.h.getNum()) + ")"
            ax.set_title(title)
            ax.set_ylabel(self.h.getSimpleName(id1))
        self.addAxes(ax)
        self.addAxes(bp)
        cAllPvals(lval, self.params['atypes'])
        return ax,bp

    def getROCspecific(ana, m=0, n=1):
        actual = [ana.aval[i] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        score = [ana.f_ranks[i - ana.h.start] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=n)
        df = pd.DataFrame()
        df['fpr'] = fpr
        df['tpr'] = tpr
        df['thrs'] = thrs
        df['group'] = "-".join([str(m), str(n)])
        return df

    def getROCAUCspecific(ana, m=0, n=1):
        actual = [ana.aval[i] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        score = [ana.f_ranks[i - ana.h.start] for i in ana.i1
                if ana.aval[i] == m or ana.aval[i] == n]
        fpr, tpr, thrs = roc_curve(actual, score, pos_label=n)
        roc_auc = auc(fpr, tpr)
        return "%.2f" % roc_auc

    def getROCAUC(ana):
        res = []
        for k in range(1, len(ana.atypes)):
            v = ana.getROCAUCspecific(0, k)
            res += [v]
        return ",".join(res)

    def getROC(ana):
        res = None
        for k in range(1, len(ana.atypes)):
            v = ana.getROCspecific(0, k)
            if (res is None):
                res = v
            else:
                res = pd.concat([res, v], ignore_index=True, sort=False)
        return res

    def getTPvals(ana):
        res = []
        lval, score = ana.getScores()
        for k in range(1, len(ana.atypes)):
            t, p = ttest_ind(lval[0],lval[k], equal_var=False)
            res += ["%.3g" % p]
        return ",".join(res)

    def getStats(self, l1, wt1, annotation=[]):
        src = re.sub(" .*", "", self.h.getSource())
        species = annotation[1]
        if species == 'Hs' or species == 'Rm' :
            self.orderData(l1, wt1)
        else:
            genes = []
            self.convertMm(l1, genes)
            self.orderData(self.gene_groups, wt1)
        if len(self.ind_r) == 0:
            return [src, 0.5, 1, len(self.st1), len(self.st2)] + annotation
        roc = self.getROCAUC()
        p = self.getTPvals()
        lval, score = self.getScores()
        n1 = np.sum([len(lval[i])for i in range(1, len(lval))])
        return [src, roc, p, len(lval[0]), n1] + annotation
        
    def printMeanAbsoluteDeviation(ana, ofile):
        from scipy.stats import median_absolute_deviation
        fp = ana.h.fp;
        fp.seek(0, 0);
        head = fp.readline();
        of = open(ofile, "w")
        of.write("\t".join(["ArrayID", "MAD"]) + "\n")
        index = 0
        for line in fp:
          line = re.sub("[\r\n]", "", line)
          ll = line.split("\t")
          if len([i for i in ana.order if ll[i] == '']) > 0:
              continue
          v1 = [float(ll[i]) for i in ana.order]
          mad = median_absolute_deviation(v1)
          of.write("\t".join([ll[0], str(mad)]) +"\n")
          index += 1
        of.close()

    def prepareData(self, dbid, cfile =
            "/booleanfs2/sahoo/Hegemon/explore.conf"):
        self.db = hu.Database(cfile)
        self.dbid = dbid
        self.h = hu.Hegemon(self.db.getDataset(self.dbid))
        self.h.init()
        self.h.initPlatform()
        self.h.initSurv()

    def initData(self, atype, atypes, ahash):
        for i in range(len(atypes)):
            ahash[atypes[i]] = i
        aval = [ahash[i] if i in ahash else None for i in atype]
        expg = [i for i in self.h.aRange() if aval[i] is not None]
        self.st1 = [i for i in self.h.aRange() if aval[i] == 0]
        self.st2 = [i for i in self.h.aRange() if aval[i] == 1]
        self.st3 = [i for i in self.h.aRange() if aval[i] == 2]
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = expg
        self.printInfo()

    def getPeters(self, tn=1):
        self.prepareData("PLP7")
        atype = self.h.getSurvName("c clinical condition")
        atypes = ['Normal', 'UC', 'CD']
        ahash = {"control": 0, "Ulcerative Colitis":1, "Crohn's disease":2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c gender")
            atypes = ['F', 'M']
            ahash = {'female':0, 'male':1}
            atype = [atype[i] if aval[i] == 0 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getNoble(self):
        self.prepareData("PLP6")
        atype = self.h.getSurvName("c disease")
        atypes = ['Normal', 'UC']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getArijs2018(self, tn=1):
        self.prepareData("PLP10")
        atype = self.h.getSurvName("c Response")
        atypes = ['Control', 'UC R', 'UC NR', 'Active UC', 'UC other']
        ahash = {}
        if (tn == 2):
            atypes = ['UC R', 'UC NR']
        if (tn == 3):
            pid = self.h.getSurvName("c study individual number")
            res = self.h.getSurvName("c Response")
            phash = {}
            for i in range(len(atype)):
                if res[i] == 'UC R':
                    phash[pid[i]] = 'R'
                if res[i] == 'UC NR':
                    phash[pid[i]] = 'NR'
            time = self.h.getSurvName("c week (w)")
            atype = [phash[pid[i]] if pid[i] in phash and time[i] == 'W0'
                    else None for i in range(len(atype))]
            atypes = ['R', 'NR']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getWu2007(self):
        self.prepareData("PLP12")
        atype = self.h.getSurvName("c type")
        atypes = ['N', 'UC', 'CD']
        ahash = {'N': 0, 'UC-Aff': 1, 'CD-Aff': 2}
        atypes = ['N', 'UC-un', 'CD-un', 'IC-un', 'INF', 
                'UC-Aff', 'CD-Aff', 'IC-Aff']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getVancamelbeke(self):
        self.prepareData("PLP16")
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'UC', 'CD']
        ahash = {'Biopsy from inflamed colonic mucosa of active UC patient':1,
                'Biopsy from inflamed colonic mucosa of active CD patient':2,
                'Biopsy from normal colonic mucosa of control individual':0}
        self.initData(atype, atypes, ahash)

    def getDePreter(self, t1 = 1):
        self.prepareData("PLP24")
        atype = self.h.getSurvName("c src1")
        atypes = ['UC Rp', 'UC Rb', 'UC p', 'UC b']
        ahash = {'Colonic mucosal biopsy from UC patient in remission before probiotics intake':1,
                'Colonic mucosal biopsy from UC patient in remission before placebo intake':0,
                'Colonic mucosal biopsy from active UC patient before placebo intake':2,
                'Colonic mucosal biopsy from active UC patient before probiotics intake':3}
        if t1 == 2:
            atypes = ['UC R', 'UC']
            ahash = {'Colonic mucosal biopsy from UC patient in remission before placebo intake':0,
                    'Colonic mucosal biopsy from active UC patient before placebo intake':1}
            if t1 == 3:
                atypes = ['UC R', 'UC']
            ahash = {'Colonic mucosal biopsy from UC patient in remission before probiotics intake':0,
                    'Colonic mucosal biopsy from active UC patient before probiotics intake':1}
            if t1 == 4:
                atypes = ['UC R', 'UC']
            ahash = {'Colonic mucosal biopsy from UC patient in remission before probiotics intake':0,
                    'Colonic mucosal biopsy from UC patient in remission before placebo intake':0,
                    'Colonic mucosal biopsy from active UC patient before placebo intake':1,
                    'Colonic mucosal biopsy from active UC patient before probiotics intake':1}
        self.initData(atype, atypes, ahash)

    def getArijs2009(self, tn=1):
        self.prepareData("PLP27")
        atype = self.h.getSurvName("c tissue")
        ahash = {'Colon':0, 'Ileum':1}
        tissue = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c before or after first infliximab treatment")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'Before':1, 'After':2, 'Not':0}
        treatment = [ahash[i] if i in ahash else None for i in atype]
        response = self.h.getSurvName("c response to infliximab")
        atype = self.h.getSurvName("c disease")
        atypes = ['Control', 'UC', 'CD']
        ahash = {}
        if (tn == 2):
            atypes = ["R", "NR"]
            ahash = {"Yes": 0, "No": 1}
            atype = response
            atype = [atype[i] if tissue[i] == 0 and treatment[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ["R", "NR"]
            ahash = {"Yes": 0, "No": 1}
            atype = response
            atype = [atype[i] if tissue[i] == 0 and treatment[i] == 2
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHaberman2014(self, tn=1):
        self.prepareData("PLP11")
        atype = self.h.getSurvName("c deep ulcer")
        ahash = {'NA':0, 'No':1, 'Yes':2, 'no':1}
        ulcer = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['Control', 'UC', 'CD']
        ahash = {'UC':1, 'Not IBD':0, 'CD':2, 'not IBD':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if ulcer[i] == 0 or ulcer[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [ulcer[i] if aval[i] == 2
                    else None for i in range(len(atype))]
            atypes = ['No', 'Yes']
            ahash = {1:0, 2:1}
        self.initData(atype, atypes, ahash)

    def getHaberman2018(self):
        self.prepareData("PLP14")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['Control', 'CD']
        ahash = {'Non-IBD':0, 'CD':1}
        self.initData(atype, atypes, ahash)

    def getVanhove(self, dtype=0, tn=1):
        self.prepareData("PLP23")
        activity = self.h.getSurvName("c disease activity")
        atype = self.h.getSurvName("c disease")
        atypes = ['Control', 'UC', 'CD', 'I', 'A', 'N']
        ahash = {'ulcerative colitis':1, "Crohn's disease":2, 'control':0,
                'active':4, 'inactive':3, 'normal': 5}
        if (tn == 2):
            atypes = ['I', 'A']
            ahash = {'active':1, 'inactive':0}
            atype = activity
        self.initData(atype, atypes, ahash)

    def getVanderGoten(self, tn=1):
        self.prepareData("PLP25")
        activity = self.h.getSurvName("c disease activity")
        atype = self.h.getSurvName("c disease")
        atypes = ['Control', 'UC', 'CD', 'I', 'A', 'NA']
        ahash = {'control':0,
                'active':4, 'inactive':3, 'not applicable': 5}
        if (tn == 2):
            atypes = ['I', 'A']
            ahash = {'active':1, 'inactive':0}
            atype = activity
        self.initData(atype, atypes, ahash)

    def getPekow(self):
        self.prepareData("PLP59")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'qUC', 'nUC']
        ahash = {'normal control': 0, 
                'quiescent ulcerative colitis': 1,
                'ulcerative colitis with neoplasia': 2}
        self.initData(atype, atypes, ahash)

    def getGao(self, tn=1):
        self.prepareData("PLP34")
        atype = self.h.getSurvName("c group")
        atypes = ['C', 'D', 'A', 'A/D']
        ahash = {'control': 0, 'DSS': 1, 'AOM': 2, 'AOM/DSS': 3}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'control': 0, 'DSS': 1}
        self.initData(atype, atypes, ahash)

    def getWatanabe(self):
        self.prepareData("PLP57")
        atype = self.h.getSurvName("c desc")
        atypes = ['UC-NonCa', 'UC-Ca']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getEColi(self):
        self.prepareData("CRC141")
        series = self.h.getSurvName("c Series")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'K12', 'O157']
        ahash = {'control, 60min':0, 'K-12, 60min':1,
                'O157:H7, 120min':2, 'control, 90min':0,
                'O157:H7, 60min':2, 'O157:H7, 90min':2,
                'control, 120min':0, 'K12, 120min':1, 'K-12, 90min':1}
        self.initData(atype, atypes, ahash)

    def getMatsuki(self):
        self.prepareData("CRC142")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'Lc', 'Bb']
        ahash = {'Caco-2 cells cultured with B. breve':2,
                'Caco-2 cells alone':0,
                'Caco-2 cells cultured with L. casei': 1}
        self.initData(atype, atypes, ahash)

    def getArbibe(self):
        self.prepareData("CRC143")
        atype = self.h.getSurvName("c Title")
        atypes = ['C', 'M90T', 'OspF', 'OspF C']
        ahash = {'Caco2_OspF complementation infected_Rep1':3,
                'Caco2_Non infected_Rep2':0,
                'Caco2_OspF mutant infected_Rep1':2,
                'Caco2_OspF complementation infected_Rep2':3,
                'Caco2_M90T wild type infected_Rep1':1,
                'Caco2_Non infected_Rep1':0,
                'Caco2_M90T wild type infected_Rep2':1,
                'Caco2_OspF mutant infected_Rep2':2}
        self.initData(atype, atypes, ahash)

    def getPereiraCaro(self):
        self.prepareData("CRC141")
        series = self.h.getSurvName("c Series")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'HTy', 'EHTy']
        ahash = {'control':0}
        self.initData(atype, atypes, ahash)

    def getKonnikova(self):
        self.prepareData("PLP68")
        media = self.h.getSurvName("c collection media")
        atype = self.h.getSurvName("c status")
        atypes = ['uninflamed', 'inflamed', 'D', 'R']
        ahash = {'RNA Later':3, 'DMSO':2}
        self.initData(atype, atypes, ahash)

    def getKarns2019(self):
        self.prepareData("PLP69")
        atype = self.h.getSurvName("c disease subtype")
        atypes = ['Control', 'UC', 'CD', 'iCD']
        ahash = {'ileal CD':3, 'not IBD':0, 'colonic CD':2}
        self.initData(atype, atypes, ahash)

    def getPeck2015(self):
        self.prepareData("PLP70")
        inflamed = self.h.getSurvName("c inflamed")
        etype = self.h.getSurvName("c Type")
        atype = self.h.getSurvName("c disease_stage")
        atypes = ['NA', 'B1', 'B2', 'B3']
        ahash = {'B1/non-strictuing, non-penetrating':1,
                'B3/penetrating':3,
                'NA':0,
                'B2/stricturing':2}
        self.initData(atype, atypes, ahash)

    def getCorraliza2018(self):
        self.prepareData("PLP71")
        response = self.h.getSurvName("c hsct responder")
        atype = self.h.getSurvName("c disease")
        atypes = ['Control', 'CD', "YES", "NO", "C"]
        ahash = {'Healthy non-IBD':0, "Crohn's disease (CD)":1}
        self.initData(atype, atypes, ahash)

    def getArze2019(self):
        self.prepareData("PLP72")
        atype = self.h.getSurvName("c disease status")
        atypes = ['Control', 'UC', "CD"]
        ahash = {'Non IBD':0, 'Ulcerative Colitis':1, "Crohn's Disease":2}
        self.initData(atype, atypes, ahash)

    def getVerstockt2019(self):
        self.prepareData("PLP73")
        atype = self.h.getSurvName("c clinical history")
        atypes = ['R', 'NR']
        ahash = {'responder':0, 'non-responder':1}
        self.initData(atype, atypes, ahash)

    def getHasler(self):
        self.prepareData("PLP74")
        tissue = self.h.getSurvName("c tissue")
        inflammation = self.h.getSurvName("c inflammation")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['Control', 'UC', "CD", 'non inflamed', 'inflamed']
        ahash = {'disease control':0, 'healthy':0}
        self.initData(atype, atypes, ahash)

    def getZhao2019(self):
        self.prepareData("PLP75")
        atype = self.h.getSurvName("c disease state")
        atype = self.h.getSurvName("c src1")
        atypes = ['Normal', "CD u", "CD i"]
        ahash = {'control':0,
                'Crohn\xe2\x80\x99s disease uninvolved':1,
                'Crohn\xe2\x80\x99s disease involved':2}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getKugathasan2008(self):
        self.prepareData("PLP76")
        atype = self.h.getSurvName("c Type")
        atypes = ['Normal', 'UC', "CD", "CD i"]
        ahash = {'Healthy control':0,
                'Colon-only CD':2,
                'Ileo-colonic CD':3,
                'Ulcerative colitis':1,
                'Internal control':0}
        self.initData(atype, atypes, ahash)

    def getZhao2015(self):
        self.prepareData("PLP77")
        atype = self.h.getSurvName("c disease state")
        atypes = ['Normal', 'UC I', "UC A"]
        ahash = {'ulcerative colitis inactive':1,
                'healthy control':0,
                'ulcerative colitis active':2}
        self.initData(atype, atypes, ahash)

    def getTang2017(self):
        self.prepareData("PLP78")
        state = self.h.getSurvName("c inflammation")
        atype = self.h.getSurvName("c disease state")
        atypes = ['Normal', 'UC', "CD", 'Inactive', "Active"]
        ahash = {'non-IBD control':0}
        self.initData(atype, atypes, ahash)

    def getCarey2008(self):
        self.prepareData("PLP79")
        atype = self.h.getSurvName("c Type")
        atypes = ['Normal', 'UC', "CD", "CD t"]
        ahash = {'healthy control reference':0,
                'CD':2,
                'treated CD':3,
                'UC':1,
                'Internal Control':0}
        self.initData(atype, atypes, ahash)

    def getDotti2017(self):
        self.prepareData("PLP80")
        culture = self.h.getSurvName("c organoid_culture")
        atype = self.h.getSurvName("c case_phenotype")
        atypes = ['Normal', 'UC'];
        ahash = {'ulcerative colitis (UC) patient':1, 'non-IBD control':0}
        self.initData(atype, atypes, ahash)

    def getDenson2018(self):
        self.prepareData("PLP86")
        response = self.h.getSurvName("c week 4 remission")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['Control', 'UC', 'Yes', 'No', 'NA'];
        ahash = {'Ulcerative Colitis':1}
        self.initData(atype, atypes, ahash)

    def getBoyd2018(self):
        self.prepareData("PLP87")
        response = self.h.getSurvName("c condition")
        ahash = {'CD active':4, 'CD inactive':3, 'control':5,
                'UC active':4, 'UC inactive':3}
        rval = [ahash[i] if i in ahash else None for i in response]
        atype = self.h.getSurvName("c condition")
        atypes = ['Control', 'UC', 'CD', 'Inactive', 'Active', 'NA'];
        ahash = {'CD active':2, 'CD inactive':2, 'control':0,
                'UC active':1, 'UC inactive':1}
        self.initData(atype, atypes, ahash)

    def getBreynaert2013(self, tn=1):
        self.prepareData("PLP38")
        atype = self.h.getSurvName("c colitis group")
        atypes = ['C', 'DA', 'D1', 'D2', 'D3', 'A'];
        ahash = {'2 cycles DSS with additional recovery period':1,
                'control':0,
                '1 cycle DSS':2,
                'acute colitis':5,
                '3 cycles DSS':4,
                '2 cycles DSS':3}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'control':0, '3 cycles DSS':1}
        self.initData(atype, atypes, ahash)

    def getGerstgrasser2017(self):
        self.prepareData("PLP36")
        atype = self.h.getSurvName("c Title")
        atypes = ['C', 'DSS'];
        ahash = {'colon wt DSS rep3':1,
                'colon wt DSS rep2':1,
                'colon wt DSS rep4':1,
                'colon wt DSS rep1':1,
                'colon wt water rep3':0,
                'colon wt water rep2':0,
                'colon wt water rep1':0,
                'colon wt water rep4':0}
        self.initData(atype, atypes, ahash)

    def getTang2012(self, tn=1):
        self.prepareData("PLP40")
        atype = self.h.getSurvName("c disease state")
        atypes = ['N', 'I', 'LD', 'HD', 'C'];
        ahash = {'low grade dysplasia lesion':2,
                'inflamed colorectal mucosa':1,
                'high grade dysplasia':3,
                'normal':0,
                'colorectal adenocarcinoma':4}
        if (tn == 2):
            atypes = ['N', 'D']
            ahash = {'low grade dysplasia lesion':1,
                    'inflamed colorectal mucosa':1,
                    'high grade dysplasia':1,
                    'normal':0,
                    'colorectal adenocarcinoma':1}
        self.initData(atype, atypes, ahash)

    def getJensen2017(self):
        self.prepareData("PLP66")
        atype = self.h.getSurvName("c disease")
        atypes = ['normal', 'colitis']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getGkouskou2016(self, tn=1):
        self.prepareData("PLP84")
        tissue = self.h.getSurvName("c tissue")
        ahash = {'proximal colon':3, 'distal colon':4}
        rval = [ahash[i] if i in ahash else None for i in tissue]
        atype = self.h.getSurvName("c src1")
        atypes = ['normal', 'AD2', 'AD4', 'proximal', 'distal']
        ahash = {'UNTREATED':0, 'AOM, 4 DSS CYCLES':2, 'AOM, 2 DSS CYCLES':1}
        if (tn == 2):
            atype = [str(atype[i])+ " " + str(tissue[i]) for i in
                    range(len(atype))]
            atypes = ['proximal', 'distal']
            ahash = {'UNTREATED proximal colon':0,
                    'UNTREATED distal colon':1}
        if (tn == 3):
            atypes = ['normal', 'colitis']
            ahash = {'UNTREATED':0, 'AOM, 4 DSS CYCLES':1, 'AOM, 2 DSS CYCLES':1}
        self.initData(atype, atypes, ahash)

    def getGkouskou2016ProxDis(self):
        self.prepareData("PLP85")
        atype = self.h.getSurvName("c tissue")
        atypes = ['proximal', 'distal']
        ahash = {'PROXIMAL COLON':0, 'DISTAL COLON':1}
        self.initData(atype, atypes, ahash)

    def getLopezDee2012(self, tn=1):
        self.prepareData("PLP49")
        gtype = self.h.getSurvName("c genotype/variation")
        ahash = {'Wild-type':8, 'TSP-null':9}
        rval = [ahash[i] if i in ahash else None for i in gtype]
        atype = self.h.getSurvName("c src1")
        atypes = ['WC', 'TC', 'WD', 'TD', 'WDS', 'WDST2', 'WDST3', 'WRFK',
                'WT', 'TN']
        ahash = {'Wt, water control':0,
                'TSP-null-water':1,
                'Wt, DSS treated':2,
                'DSS-treated TSP-null':3,
                'Wt, DSS-saline treated':4,
                'Wt, DSS-TSR2 treated':5,
                'Wt, DSS-3TSR treated':6,
                'Wt, DSS-TSR+RFK treated':7}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'Wt, water control':0,
                    'Wt, DSS treated':1,
                    'Wt, DSS-saline treated':1,
                    'Wt, DSS-TSR2 treated':1,
                    'Wt, DSS-3TSR treated':1,
                    'Wt, DSS-TSR+RFK treated':1}
        self.initData(atype, atypes, ahash)

    def getSarvestani2018(self):
        self.prepareData("ORG16")
        atype = self.h.getSurvName("c Pathology")
        atypes = ['N', 'UC', 'D', 'C']
        ahash = {'Normal':0,
                'Chronic active colitis':1,
                'Colitis':1,
                'Colitis with benign strictures':1,
                'Colitis with fibrosis and treatment effect':1,
                'Low-grade dysplasia':2,
                'T3N1a':3}
        self.initData(atype, atypes, ahash)

    def getPlanell2013(self):
        self.prepareData("PLP88")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'NI', 'Re', 'I']
        ahash = {
                'Human colon biopsies from UC patient with active disease (involved mucosa)':3,
                'Human colon biopsies from non-inflammatory control':0,
                'Human colon biopsies from UC patient with active disease (non-involved mucosa)':1,
                'Human colon biopsies from UC patient in remission (involved mucosa)':2}
        self.initData(atype, atypes, ahash)

    def getLyons2018(self):
        self.prepareData("PLP89")
        atype = self.h.getSurvName("c inflammation level")
        atypes = ['NI', 'M', 'S']
        ahash = {'severe':2, 'moderate':1, 'non-inflamed':0}
        self.initData(atype, atypes, ahash)

    def getFang2012(self):
        self.prepareData("PLP90")
        atype = self.h.getSurvName("c time")
        atypes = ['W0', 'W2', 'W4', 'W6']
        ahash = {'0 week':0, '4 weeks':2, '6 weeks':3, '2 weeks':1}
        self.initData(atype, atypes, ahash)

    def getSchiering2014(self):
        self.prepareData("PLP91")
        gtype = self.h.getSurvName("c genotype")
        ahash = {'wild type':3, 'Il23r-/-':4, 'Foxp3gfp':5}
        rval = [ahash[i] if i in ahash else None for i in gtype]
        atype = self.h.getSurvName("c cell type")
        atypes = ['TC', 'TP', 'TN', 'WT', 'I23', 'F3']
        ahash = {
                'TCR\xce\xb2+CD4+ T cells from colon':0,
                'TCR\xce\xb2+CD4+Foxp3+ from colon lamina propria (cLP)':1,
                'TCR\xce\xb2+CD4+Foxp3+ from mesenteric lymph node (MLN)':2}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getKremer2012(self):
        self.prepareData("PLP92")
        atype = self.h.getSurvName("c disease status")
        atypes = ['N', 'UC']
        ahash = {'TNBS colitis':1, 'Healthy Control':0}
        self.initData(atype, atypes, ahash)

    def getHo2014(self):
        self.prepareData("PLP93")
        gtype = self.h.getSurvName("c tissue")
        ahash = {'Spleen':3, 'Colon':4}
        rval = [ahash[i] if i in ahash else None for i in gtype]
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'UC', 'UCt', 'Sp', 'CO']
        ahash = {
                'Mock':0,
                'EA treatment/TNBS-induced colitis':2,
                'TNBS-induced colitis':1}
        self.initData(atype, atypes, ahash)

    def getDohi2014(self):
        self.prepareData("PLP94")
        gtype = self.h.getSurvName("c treated with")
        ahash = {'none (untreated control)':2,
                '10 mg/kg control IgG2a mAb (anti-human CD20)':3,
                '0.3 mg/kg TNFR-Ig':4,
                '10 mg/kg anti-TWEAK mP2D10':5,
                'combination of TNFR-Fc (0.3 mg/kg) and anti-TWEAK mP2D10 (10 mg/kg)':6}
        rval = [ahash[i] if i in ahash else None for i in gtype]
        atype = self.h.getSurvName("c injected with")
        atypes = ['N', 'UC']
        ahash = {'trinitrobenzene sulfonic acid (TNBS)':1,
                'none (na\xc3\xafve control)':0}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getDeBuhr2006(self):
        self.prepareData("PLP95")
        atype = self.h.getSurvName("c Title")
        atypes = ['B6-WT', 'B6-IL10', 'C3-WT', 'C3-IL10']
        ahash = {'C57BL/6J, sample 1':0,
                'C57BL/6J, 4 week old, sample 2':0,
                'C57BL/6J-Il10tm1Cgn, sample 2':1,
                'C57BL/6J-Il10tm1Cgn, sample 1':1,
                'C3H/HeJBir, sample 2':2,
                'C3H/HeJBir, sample-1':2,
                'C3H/HeJBir-Il10tm1Cgn, sample 2':3,
                'C3H/HeJBir-Il10tm1Cgn, sample 1':3}
        self.initData(atype, atypes, ahash)

    def getRuss2013(self, tval):
        self.prepareData("PLP96")
        tissue = self.h.getSurvName("c tissue")
        ahash = {'colon epithelium':2, 'colon':3}
        rval = [ahash[i] if i in ahash else None for i in tissue]
        atype = self.h.getSurvName("c genotype/variation")
        atypes = ['WT', 'IL10']
        ahash = {'IL10-/-':1, 'wildtype':0}
        atype = [atype[i] if rval[i] == tval else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPunit2015(self):
        self.prepareData("PLP97")
        atype = self.h.getSurvName("c genotype")
        atypes = ['WT', 'TNFR2']
        ahash = {'Wildtype':0, 'TNFR2-knockout':1}
        self.initData(atype, atypes, ahash)

    def getTam2019(self):
        self.prepareData("PLP98")
        atype = self.h.getSurvName("c genotype")
        atypes = ['WT', 'Tnfr1']
        ahash = {'Tnfrsf1a-/-':1, 'WT':0}
        self.initData(atype, atypes, ahash)

    def getLamas2018(self, tn=1):
        self.prepareData("PLP99")
        atype = self.h.getSurvName("c src1")
        atypes = ['D0', 'D4', 'D12', 'D22']
        ahash = {'Cecum_Flore WT_Day0':0,
                'Cecum_Flore KO_Day0':0,
                'Cecum_Flore WT_Day4':1,
                'Cecum_Flore KO_Day4':1,
                'Cecum_Flore WT_Day12':2,
                'Cecum_Flore KO_Day12':2,
                'Cecum_Flore WT_Day22':3,
                'Cecum_Flore KO_Day22':3}
        if (tn == 2):
            atypes = ['WT', 'KO']
            ahash = {'Cecum_Flore WT_Day0':0,
                    'Cecum_Flore KO_Day0':1,
                    'Cecum_Flore WT_Day4':0,
                    'Cecum_Flore KO_Day4':1,
                    'Cecum_Flore WT_Day12':0,
                    'Cecum_Flore KO_Day12':1,
                    'Cecum_Flore WT_Day22':0,
                    'Cecum_Flore KO_Day22':1}
        self.initData(atype, atypes, ahash)

    def getFuso1(self):
        self.prepareData("CRC112.2")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'FN']
        ahash = {"non-infected": 0, "F. nucleatum":1}
        self.initData(atype, atypes, ahash)

    def getGEOMac(self):
        self.prepareData("GL4")
        atype = self.h.getSurvName("c Source")
        atypes = ['Control', 'Treated']
        aval = [None, None]
        for i in range(2, len(atype)):
            if re.search(r'control', atype[i]) or \
                    re.search(r'untreated', atype[i]):
                        aval += [0]
            else:
                aval += [1]
        self.st1 = [ i for i in self.h.aRange() if aval[i] == 0]
        self.st2 = [ i for i in self.h.aRange() if aval[i] == 1]
        self.st3 = []
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = self.st1 + self.st2 + self.st3
        self.printInfo()

    def getGEOMacAnn(self):
        self.prepareData("G16")
        atype = self.h.getSurvName("c Type")
        atypes = ['M0', 'M1', "M2"]
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBeyer2012(self):
        self.prepareData("MAC1")
        atype = self.h.getSurvName("c cell type")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M1 macrophages':1, 'M0 macrophages':0, 'M2 macrophages':2}
        self.initData(atype, atypes, ahash)

    def getSaliba2016(self):
        self.prepareData("MAC5")
        atype = self.h.getSurvName("c group")
        atypes = ['NM', 'M1', 'M2', 'BM']
        ahash = {'Macrophage with non growing bacteria':1,
                'Macrophage with growing bacteria':2,
                'Naive Macrophage':0,
                'Bystanders':3}
        self.initData(atype, atypes, ahash)

    def getAhrens2013(self):
        self.prepareData("LIV9")
        atype = self.h.getSurvName("c group")
        atypes = ['C', 'HO', 'S', 'N']
        ahash = {'Control':0, 'Healthy obese':1, 'Nash':3, 'Steatosis':2}
        self.initData(atype, atypes, ahash)

    def getSmith2015(self):
        self.prepareData("PLP82")
        atype = self.h.getSurvName("c disease")
        atypes = ['HC', 'CD']
        ahash = {'Healthy Control':0, "Crohn's Disease":1 }
        self.initData(atype, atypes, ahash)

    def getDill2012HPC(self):
        self.prepareData("LIV10")
        atype = self.h.getSurvName("c disease state")
        atypes = ['HC', 'AH']
        ahash = {'acute hepatitis':1, 'healthy':0}
        self.initData(atype, atypes, ahash)

    def getDill2012IFN(self):
        self.prepareData("LIV11")
        atype = self.h.getSurvName("c treatment")
        atypes = ['Un', 'IFNa', 'TFNg']
        ahash = {'IFNalpha':1, 'IFNgamma':2, 'untreated':0}
        self.initData(atype, atypes, ahash)

    def getTrepo2018(self):
        self.prepareData("LIV12")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'AH', 'AFL', 'AC']
        ahash = {'liver tissue':0,
                'Alcoholic_cirrhosis, liver tissue':3,
                'Mild_acute_alcoholic_hepatitis, liver tissue':1,
                'alcoholic_steatosis, liver tissue':2}
        self.initData(atype, atypes, ahash)


    def getHoshida2013(self, tn=1):
        self.prepareData("LIV15.2")
        time1 = self.h.getSurvName("c days to death")
        time1 = ["", ""] + [float(time1[k]) if time1[k] != 'NA' else None for k in self.h.aRange()]
        time2 = self.h.getSurvName("c days to decomp")
        time2 = ["", ""] + [float(time2[k]) if time2[k] != 'NA' else None for k in self.h.aRange()]
        time3 = self.h.getSurvName("c days to child")
        time3 = ["", ""] + [float(time3[k]) if time3[k] != 'NA' else None for k in self.h.aRange()]
        time4 = self.h.getSurvName("c days to hcc")
        time4 = ["", ""] + [float(time4[k]) if time4[k] != 'NA' else None for k in self.h.aRange()]
        status1 = self.h.getSurvName("c death")
        status1 = ["", ""] + [int(status1[k]) if status1[k] != 'NA' else None for k in self.h.aRange()]
        status2 = self.h.getSurvName("c decomp")
        status2 = ["", ""] + [int(status2[k]) if status2[k] != 'NA' else None for k in self.h.aRange()]
        status3 = self.h.getSurvName("c child")
        status3 = ["", ""] + [int(status3[k]) if status3[k] != 'NA' else None for k in self.h.aRange()]
        status4 = self.h.getSurvName("c hcc")
        status4 = ["", ""] + [int(status4[k]) if status4[k] != 'NA' else None for k in self.h.aRange()]
        days = 365 * 15
        atype = [None] * len(time1)
        for k in self.h.aRange():
            for g in [[time1, status1], [time2, status2],
                      [time3, status3], [time4, status4]]:
                if (g[0][k] is not None and g[1][k] is not None): 
                    if (g[0][k] < days and g[1][k] == 1):
                        atype[k] = 1
                        break
                    else:
                        atype[k] = 0
        atypes = [0, 1]
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getRamilo2007v1(self):
        self.prepareData("BL9.1")
        atype = self.h.getSurvName("c Pathogen")
        atypes = ['None', 'Ecoli', 'MRSA', 'MSSA', 'InfA', 'pneu']
        ahash = {'S. aureus, MRSA':2,
                'E. coli':1,
                'S. aureus, MSSA':3,
                'None':0,
                'Influenza A':4,
                'S. pneumoniae':5}
        self.initData(atype, atypes, ahash)

    def getPG2019lps(self, tn=1):
        self.prepareData("MAC28.3")
        atype = self.h.getSurvName("c Type")
        atypes = ['LPS0', 'LPS5', 'GIV0', 'GIV5', 'U']
        ahash = {'sh2_GIV_LPS_0hr':2, 'Undetermined':4, 'sh1_GIV_LPS_5hr':3,
                'shC_LPS_0hr':0, 'sh2_GIV_LPS_5hr':3, 'sh1_GIV_LPS_0hr':2,
                'shC_LPS_5hr':1}
        if (tn == 2):
            atypes = ['LPS0', 'LPS5', 'GIV0', 'GIV5']
            ahash = {'sh2_GIV_LPS_0hr':2, 'sh1_GIV_LPS_5hr':3, 'shC_LPS_0hr':0,
                    'sh2_GIV_LPS_5hr':3, 'sh1_GIV_LPS_0hr':2, 'shC_LPS_5hr':1}
        self.initData(atype, atypes, ahash)

    def getHugo2016(self):
        self.prepareData("ML12")
        atype = self.h.getSurvName("c anti-pd-1 response")
        atypes = ['CR', 'PR', 'PD'];
        ahash = {'Progressive Disease':2,
                'Partial Response':1,
                'Complete Response':0}
        self.initData(atype, atypes, ahash)

    def getPrat2017(self):
        self.prepareData("ML13")
        atype = self.h.getSurvName("c response")
        atypes = ['RC_RP_SD', 'PD'];
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getCassetta2019(self):
        self.prepareData("MAC29")
        atype = self.h.getSurvName("c condition")
        atypes = ['N', 'NB', 'NE', 'BC', 'EC'];
        ahash = {'Endometrial cancer':4,
                'Normal':0,
                'Breast cancer':3,
                'Normal Breast':1,
                'Normal Endometrial':2}
        self.initData(atype, atypes, ahash)

    def getPoczobutt2016(self):
        self.prepareData("MAC30")
        atype = self.h.getSurvName("c src1")
        atypes = ['L', 'TL2', 'TL3']
        ahash = {'tumor-bearing lung, 2 week':1,
                'tumor-bearing lung, 3 week':2,
                'lung':0}
        self.initData(atype, atypes, ahash)

    def getWoetzel2014(self):
        self.prepareData("MAC31")
        atype = self.h.getSurvName("c disease state")
        atype2 = self.h.getSurvName("c clinical status")
        atype = [atype[i] + atype2[i] for i in range(len(atype))]
        atypes = ['HC', 'RA', 'OA']
        ahash = {'healthy control':0,
                'rheumatoid arthritis':1,
                'synovial tissue isolated from osteoarthritic joint':2,
                'osteoarthritis':2,
                'normal control':0}
        self.initData(atype, atypes, ahash)

    def getXue2014(self, mt=1):
        self.prepareData("MAC2")
        atype = self.h.getSurvName("c Type")
        atypes = ['M0', 'M1', 'M2']
        ahash = {\
                'M_GMCSF_IL4_72h':2,
                'M_GMCSF_IFNg_72h':1,
                'M0_GMCSF_72h':0}
        if mt == 2:
            ahash = {\
                    'M_GMCSF_IL4_24h':2,
                    'M_GMCSF_IFNg_24h':1,
                    'M0_GMCSF_24h':0}
        if mt == 3:
            ahash = {\
                    'M_GMCSF_IL4_12h':2,
                    'M_GMCSF_IFNg_12h':1,
                    'M0_GMCSF_12h':0}
        if mt == 4:
            ahash = {\
                    'M0_GMCSF_0h':0,
                    'M0_GMCSF_12h':0,
                    'M0_GMCSF_24h':0,
                    'M0_GMCSF_48h':0,
                    'M0_GMCSF_6h':0,
                    'M0_GMCSF_72h':0,
                    'M0_MCSF_0h':0,
                    'M1/2_GMCSF_24h':0,
                    'M_GMCSF_IFNg_30min':0,
                    'M_GMCSF_IFNg_1h':0,
                    'M_GMCSF_IFNg_2h':0,
                    'M_GMCSF_IFNg_4h':1,
                    'M_GMCSF_IFNg_6h':1,
                    'M_GMCSF_IFNg_12h':1,
                    'M_GMCSF_IFNg_24h':1,
                    'M_GMCSF_IFNg_72h':1,
                    'M_GMCSF_IL4_30min':2,
                    'M_GMCSF_IL4_1h':2,
                    'M_GMCSF_IL4_2h':2,
                    'M_GMCSF_IL4_4h':2,
                    'M_GMCSF_IL4_6h':2,
                    'M_GMCSF_IL4_12h':2,
                    'M_GMCSF_IL4_24h':2,
                    'M_GMCSF_IL4_72h':2,
                    'M_MCSF_IL4_72h':2}
        self.initData(atype, atypes, ahash)

    def getShaykhiev2009(self, tn=1):
        self.prepareData("MAC11")
        atype = self.h.getSurvName("c desc")
        atype = [str(i).split("-")[0] for i in atype]
        atypes = ['NS', 'S', 'COPD']
        ahash = {}
        if (tn == 2):
            atypes = ['H', 'COPD']
            ahash = {'NS':0, 'S':0}
        self.initData(atype, atypes, ahash)

    def getWoodruff2005(self, tn=1):
        self.prepareData("MAC12")
        atype = self.h.getSurvName("c status")
        atypes = ['NS', 'S', 'A']
        ahash = {'Asthmatic':2, 'Smoker':1, 'Nonsmoker':0}
        if (tn == 2):
            atypes = ['H', 'A']
            ahash = {'Asthmatic':1, 'Smoker':0, 'Nonsmoker':0}
        self.initData(atype, atypes, ahash)

    def getWS2009(self):
        self.prepareData("MAC12.2")
        atype = self.h.getSurvName("c Type")
        atypes = ['NS', 'S', 'A', 'C']
        ahash = {'Nonsmoker':0, 'Asthmatic':2, 'Smoker':1, 'smoker':1,
                'non-smoker':0, 'COPD':3}
        self.initData(atype, atypes, ahash)

    def getZhang2015(self, tn=1):
        self.prepareData("MAC3")
        atype = self.h.getSurvName("c Type")
        atypes = ['M', 'iM', 'M1', 'iM1', 'M2', 'iM2', 'i']
        ahash = {'IPSDM M2':5,
                'IPSDM MAC':1,
                'IPSDM M1':3,
                'iPS':6,
                'HMDM M1':2,
                'HMDM MAC':0,
                'HMDM M2':4}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'IPSDM M2':2,
                    'IPSDM MAC':0,
                    'IPSDM M1':1}
        if (tn == 3):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'HMDM M1':1,
                    'HMDM MAC':0,
                    'HMDM M2':2}
        self.initData(atype, atypes, ahash)

    def getHaribhai2016(self):
        self.prepareData("MAC13")
        atype = self.h.getSurvName("c cell type")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M2a macrophages':2, 'M0 macrophages':0, 'M1 macrophages':1}
        self.initData(atype, atypes, ahash)

    def getOhradanovaRepic2018(self, tn=1):
        self.prepareData("MAC14")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2', 'IL10']
        ahash = {'mock-activated (medium only; control) for 2d':0,
                'activated with 100 ng/ml LPS + 25ng/ml IFN\xce\xb3 for 2d':1,
                'activated with 20 ng/ml IL-4 for 2d':2,
                'activated with 20 ng/ml IL-10 for 2d':3}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'mock-activated (medium only; control) for 2d':0,
                    'activated with 100 ng/ml LPS + 25ng/ml IFN\xce\xb3 for 2d':1,
                    'activated with 20 ng/ml IL-4 for 2d':2}
        if (tn == 3):
            atypes = ['M0', 'IL10']
            ahash = {'mock-activated (medium only; control) for 2d':0,
                    'activated with 20 ng/ml IL-10 for 2d':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getGharib2019CF(self, val = 0):
        self.prepareData("MAC15")
        atype = self.h.getSurvName("c patient identification number")
        atype = [str(i).split(" ")[0] for i in atype]
        ahash = {'Non':0, 'CF':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c title")
        atype = [ str(i).split(" ")[-1] for i in atype ]
        atypes = ['None', 'IL4', 'IL10', 'MP', 'Az', 'PMNs']
        ahash = {'alone':0,
                'methylprednisolone':3,
                'azithromycin':4}
        atype = [atype[i] if rval[i] == val else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getGharib2019Alv(self, tn=1, ri=0):
        self.prepareData("MAC16")
        atype = self.h.getSurvName("c time point")
        atypes = ['D1', 'D4', 'D8']
        ahash = {'Day 1':0, 'Day 4':1, 'Day 8':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("n ventilator-free days (vfd)")
        atypes = ['VFD-Extubated/Alive', 'VFD-Intubated/Dead']
        ahash = {'0':1, '7':0, '18':0, '19':0, '21':0,
                '22':0, '23':0, '24':0, '25':0}
        atype = [atype[i] if rval[i] == ri else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getOrozco2012(self):
        self.prepareData("MAC17")
        atype = self.h.getSurvName("c treatment condition")
        atypes = ['Control', 'LPS', 'OxPAPC']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getRegan2018(self):
        self.prepareData("MAC18")
        treatment = self.h.getSurvName("c drug treatment")
        ahash = {'1 UNT':0,
                '2 CTRL':1,
                '3 Escitalopram':2,
                '4 Nortriptyline':3,
                '5 Anti-TNFa':4,
                '6 Indomethacin':5,
                '7 Prednisolone':6}
        rval = [ahash[i] if i in ahash else None for i in treatment]
        atype = self.h.getSurvName("c inflammatory stimulus")
        atypes = ['None', 'IFN7', 'IFN24', 'LPS7', 'LPS24']
        ahash = {'24 h No inflammation':0,
                'IFN 07 h':1,
                'IFN 24 h':2,
                'LPS 07 h':3,
                'LPS 24 h':4}
        self.initData(atype, atypes, ahash)

    def getMartinez2013II(self, tn=1):
        self.prepareData("MAC19.5")
        atype = self.h.getSurvName("c src1")
        atypes = ['None', 'Mono', 'IFN', 'IL4', 'IL10', 'MCSF', 'Med']
        ahash = {'monocyte-derived macrophages, IFN-y':2,
                'monocyte-derived macrophages, M-CSF':5,
                'monocyte-derived macrophages':0,
                'monocyte-derived macrophages, IL-4':3,
                'monocyte-derived macrophages, IL-10':4,
                'monocytes':1,
                'monocyte-derived macrophages, Medium':6}
        if (tn == 2):
            atypes = ['Mono', 'Mac']
            ahash = {'monocyte-derived macrophages':1, 'monocytes':0}
        if (tn == 3):
            atypes = ['Mac', 'MCSF']
            ahash = {'monocyte-derived macrophages':0,
                    'monocyte-derived macrophages, M-CSF':1}
        self.initData(atype, atypes, ahash)

    def getMartinez2013(self):
        self.prepareData("MAC19.1")
        atype = self.h.getSurvName("c Type")
        atypes = ['M0', 'M1', 'M2', 'T0', 'T3']
        ahash = {'Monocyte at 3 days':4,
                'classical or M1 activated macrophages':1,
                'Macrophage at 7 days':0,
                'Alternative or M2 activated macrophages':2,
                'Monocyte at T0':3,
                'Monocyteat 3 days':4}
        self.initData(atype, atypes, ahash)

    def getMartinez2013Mm(self):
        self.prepareData("MAC19.3")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'none':0, '18 hours with 20 ng/ml of mIL-4':2}
        self.initData(atype, atypes, ahash)

    def getKoziel2009(self):
        self.prepareData("MAC22")
        atype = self.h.getSurvName("c characteristics")
        atypes = ['control hMDMs', 'SA-treated hMDMs']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getJiang2017(self):
        self.prepareData("MAC24")
        atype = self.h.getSurvName("c source name")
        atypes = ['M1', 'M2']
        ahash = {'Bone marrow-derived macrophages, M1, LPS+IFN-r stimulated BMDM':0,
                'Bone marrow-derived macrophages, M2, IL-4 stimulated BMDM':1}
        self.initData(atype, atypes, ahash)

    def getHan2017(self, tn=1):
        self.prepareData("MAC25")
        atype = self.h.getSurvName("c Title")
        atype = [str(i).split(" ")[2] if len(str(i).split(" ")) > 3 else i \
                for i in atype]
        atypes = ['SR1078', 'M1', 'M0', 'Veh', 'M2', 'SR3335']
        ahash = {}
        if tn == 2:
            atypes = ['M0', 'M1', 'M2']
        self.initData(atype, atypes, ahash)

    def getFuentesDuculan2010(self):
        self.prepareData("MAC26")
        atype = self.h.getSurvName("c source name")
        ahash = {'Macrophages culture':0,
                'Psoriasis Non-lesional skin':1,
                'Psoriasis Lesional skin':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment group")
        atypes = ['control', 'IL17', 'IFNg', 'IL4', 'TNFa', 'LPS', "M1", "PS" ]
        ahash = {'LPS and IFNg':6, "":7}
        self.initData(atype, atypes, ahash)

    def getZhou2009(self):
        self.prepareData("MAC27")
        atype = self.h.getSurvName("c time")
        ahash = {'1h':0, '4h':1, '24h':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease state")
        atypes = ['control (lean)', 'obesity']
        ahash = {}
        atype = [atype[i] if rval[i] == 2 else None for i in range(len(atype))]
        self.rval = rval
        self.initData(atype, atypes, ahash)

#    def getSurvival(self, dbid = "TNB8"):
#        self.prepareData(dbid)
#        atype = self.h.getSurvName("status")
#        atypes = ['0', '1']
#        ahash = {"0": 0, "1":1}
#        self.initData(atype, atypes, ahash)
        
        
        
    def getSurvival(self, dbid = "CRC35.3"):
        self.prepareData(dbid, "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("status")
        atypes = ['Censor', 'Relapse']
        ahash = {"0": 0, "1":1}
        self.initData(atype, atypes, ahash)

    def getBestThr(self, ct=None, tn=3):
        f_ranks = self.f_ranks
        order = self.order
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/16
        print(thr)
        print(nm)
        time = self.h.getSurvName('time')
        status = self.h.getSurvName('status')
        vals = list(range(self.h.start)) + list(f_ranks)
        return hu.getBestThr(time, status, vals, order, None, ct, tn)

    def printSurvival(self, fthr = None, pG = None, genex = "CDX2",
            ct = None, ax = None):
        f_ranks = self.f_ranks
        order = self.order
        thr = hu.getThrData(f_ranks)
        nm = (np.max(f_ranks) - np.min(f_ranks))/16
        print(thr)
        print(nm)
        if fthr is None:
            fthr = thr[0]
        if fthr == "thr0":
            fthr = thr[0] - nm
        if fthr == "thr1":
            fthr = thr[0]
        if fthr == "thr2":
            fthr = thr[0] + nm
        if fthr == "thr2.5":
            fthr = thr[0] + 2.5 * nm
        if fthr == "thr3":
            fthr = thr[0] + 3 * nm
        g1 = [i for i in order if f_ranks[i - self.h.start] < fthr]
        g2 = [i for i in order if f_ranks[i - self.h.start] >= fthr]
        if pG is None:
            pG = [ ["Low", "red", g1], ["High", "green", g2]]
        time = self.h.getSurvName('time')
        status = self.h.getSurvName('status')
        if ct is not None:
            time, status = hu.censor(time, status, ct)
        sax = hu.survival(time, status, pG, ax)
        df = pd.DataFrame()
        df["f_ranks"] = pd.to_numeric(pd.Series(f_ranks))
        if genex is None:
            ax = None
        else:
            e = self.h.getExprData(genex)
            df[genex] = pd.to_numeric(pd.Series(e[2:]))
            ax = df.plot.scatter(x=genex, y='f_ranks')
        return sax, ax

    def getJSTOM(self):
        self.getSurvival("CRC35.3")

    def getSveen(self):
        self.getSurvival("CRC28.1")  
        
    def getMedemaRMA(self):
        self.getSurvival("CRC35.2")  
        
    def getTCGA_RMA(self):
        self.getSurvival("CRC81")    
        
    def getHU(self):
        self.getSurvival("CRC93")
        
    def getDELRIO(self):
        self.getSurvival("CRC92")
        
    def getGaedcke(self):
        self.getSurvival("CRC59")  
        
    def getStratford(self):
        self.getSurvival("PANC2")        
        
    def getMSIMSS(self):
        self.getSurvival("CRC25")
        
    def getBos(self):
        self.getSurvival("BC20")        

    def getJablonski2015(self):
        self.prepareData("MAC32")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'received media alone (M0 condition)':0,
                'classically activated (M1 condition) with LPS + IFN-gamma':1,
                'alternatively activated (M2 condition) with IL-4':2}
        self.initData(atype, atypes, ahash)

    def getZhao2017(self):
        self.prepareData("MAC33")
        atype = self.h.getSurvName("c genotype/variation")
        atype2 = self.h.getSurvName("c Stage")
        atype = [ str(atype[i]) + " " + str(atype2[i]) for i in range(self.h.end+1)]
        atypes = ['W5', 'K5', 'W24', 'K24']
        ahash = {'wildtype week 5':0,
                'Mecp2 knockout week 5':1,
                'Mecp2 knockout week 24':3,
                'wildtype week 24':2}
        self.initData(atype, atypes, ahash)

    def getChiu2013(self):
        self.prepareData("MAC34")
        atype = self.h.getSurvName("c sample type")
        atypes = ['U', 'L', 'PSC', 'PS', 'ESC', 'ES']
        ahash = {
                'B6 untreated':0,
                'Pre-symptomatic':3,
                'Symptomatic':3,
                'End-stage':5,
                'B6 End-stage':5,
                'Pre-symptomatic control':2,
                'Symptomatic control':2,
                'WTSOD1 symptomatic control':2,
                'End-stage control':4,
                'WTSOD1 end-stage control':4,
                'LPS injected 48 hr timepoint':1}
        self.initData(atype, atypes, ahash)

    def getGrabert2016(self):
        self.prepareData("MAC35")
        atype = self.h.getSurvName("c age")
        ahash = {'22month':2, '12month':1, '4month':0}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Title")
        atype = [ " ".join(str(i).split(" ")[0:2]) if \
                len(str(i).split(" ")) > 3 else None for i in atype]
        atypes = ['CeM', 'CoM', 'HiM', 'StM',
                'CeH', 'CoH', 'HiH', 'StH']
        ahash = {'Cerebellar microglia':0,
                'Cortical microglia':1,
                'Hippocampal microglia':2,
                'Striatal microglia':3,
                'Cerebellar homogenates':4,
                'Cortical homogenates':5,
                'Hippocampal homogenates':6,
                'Striatal homogenates':7}
        self.initData(atype, atypes, ahash)

    def getMatcovitchNatan2016(self):
        self.prepareData("MAC36")
        atype = self.h.getSurvName("c developmental stage")
        atypes = ['E', 'N', 'A']
        ahash = {'E11.5':0, 'E12.5':0, 'E13.5':0, 'E14.5':0, 'E16.5':0, 'E10.5':0,
                'newborn':1, 'day3':1, 'day6':1, 'day9':1, '4wk':2, '5wk':2, '6wk':2,
                '8wk':2 }
        self.initData(atype, atypes, ahash)

    def getCho2019(self):
        self.prepareData("MAC37")
        atype = self.h.getSurvName("c Treatment")
        atypes = ['M0', 'M1', 'M2', 'IL10', 'M1+M2']
        ahash = {'LPS_lo + IFNg':1,
                'IL4':2,
                'None': 0,
                'LPS_lo + IL4 + IL10 + IL13':4,
                'IL10':3}
        self.initData(atype, atypes, ahash)

    def getKrasemann2017(self):
        self.prepareData("MAC38")
        atype = self.h.getSurvName("c Title")
        atype = [str(i)[0:-3] for i in atype]
        atypes = ['WP', 'WN', 'AP', 'AN', 'SK', 'SH', 'ACp', 'ACn', 'WCn']
        ahash = {
                'WT_Phagocytic':0,
                'WT_NonPhagocytic':1,
                'Apoe knock-out_Phagocytic':2,
                'Apoe knock-out_NonPhagocytic':3,
                'SOD1:TREM2-KO (Female)':4,
                'SOD1:TREM2-KO_(Male)':4,
                'SOD1:TREM2-Het (Female)':5,
                'SOD1:TREM2-Het (Male)':5,
                'APP-PS1_Clec7apositive':6,
                'APP-PS1_Clec7anegative':7,
                'WT_Clec7anegative':8}
        self.initData(atype, atypes, ahash)

    def getZhang2013(self, dbid = "MAC39.1"):
        self.dbid = dbid
        atype = self.h.getSurvName("c disease")
        atypes = ['A', 'N']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getCoates2008(self):
        self.prepareData("MAC40")
        atype = self.h.getSurvName("c Title")
        atype = [ str(i)[0:-2] for i in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'C57BL/6 4Gy':2, 'CBA/Ca 0Gy':0, 'CBA/Ca 4Gy':1,
                'C57BL/6 0Gy':0}
        self.initData(atype, atypes, ahash)

    def getHutchins2015(self):
        self.prepareData("MAC41")
        atype = self.h.getSurvName("c Replicate")
        ahash = {'1':1, '2':2}
        kval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Title")
        atype = [str(i).split(" ")[0] for i in atype]
        atypes = ['Mast', 'N', 'S', 'E', 'Mac']
        ahash = {'Mast':0,
                'Neutrophil':1,
                'Splenic':2,
                'Eosinophils':3,
                'PEC':4,
                'na\xc3\xafve':5}
        ahash = asciiNorm(ahash)
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['None', 'IL10', 'IL10, LPS', 'LPS']
        ahash = {}
        atype = [atype[i] if kval[i] == 2 else None for i in range(len(atype))]
        self.rval = rval
        self.initData(atype, atypes, ahash)

    def getHutchins2015TPM(self, tn = 1):
        self.prepareData("MAC41.2")
        atype = self.h.getSurvName("c source_name")
        ahash = {'Bone marrow neutrophil':2,
                'spleen-purified dendritic cells':3,
                'Peritoneal exudate cells (adherent cells)':4,
                'Bone marrow-derived Eosinophils':5,
                'Bone marrow-derived mast cell':6,
                'CD4+ na\xc3\xafve T cells':7}
        ahash = asciiNorm(ahash)
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'LPS':1, 'IL10':2,'IL10\\, LPS':1, 'None':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn >= 2):
            atype = [atype[i] if rval[i] == tn or aval[i] == 0
                    else None for i in range(len(atype))]
        self.rval = rval
        self.initData(atype, atypes, ahash)

    def getMo2018(self, tn=1):
        self.prepareData("PLP15")
        atype = self.h.getSurvName("c disease state (diagnosis)")
        atypes = ['Normal', 'UC', 'CD', 'sJ', 'oJ', 'pJ']
        ahash = {'Control':0,
                'Ulcerative Colitis':1,
                "Crohn's Disease":2,
                'Systemic JIA':3,
                'Oligoarticular JIA':4,
                'Polyarticular JIA':5}
        if (tn == 2):
            atypes = ['Normal', 'UC', 'CD']
            ahash = {'Control':0,
                    'Ulcerative Colitis':1,
                    "Crohn's Disease":2}
        self.initData(atype, atypes, ahash)

    def getBurczynski2006(self):
        self.prepareData("PLP13")
        atype = self.h.getSurvName("c Disease")
        atypes = ['Normal', 'UC', 'CD']
        ahash = {'Ulcerative Colitis':1, 'Normal':0, "Crohn's Disease":2}
        self.initData(atype, atypes, ahash)

    def getPlanell2017(self):
        self.prepareData("PLP17")
        atype = self.h.getSurvName("c case_phenotype")
        atypes = ['Normal', 'UC', 'CD']
        ahash = {'Crohn':2, 'Colitis':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getMehraj2013(self, tn=1):
        self.prepareData("MAC42")
        atype = self.h.getSurvName("c cell type")
        ahash = {'Monocyte':0, 'Monocytes-Derived Macrophages':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'NS':0, 'IFNg':1, 'IL4':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        atype = [atype[i] if rval[i] == 1
                else None for i in range(len(atype))]
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub(" [0-9]*$", "", str(k)) for k in atype]
            ahash = {'Monocyte NS 6h', 'Monocyte IFNG 6h', 'Macrophage IL4 18h',
                    'Monocyte IFNG 18h', 'Monocyte IL4 18h', 'Macrophage IFNG 18h',
                    'Macrophage NS 18h', 'Monocyte IL4 6h', 'Monocyte NS 18h'}
            atypes = ['Mono', 'Mac']
            ahash = {'Macrophage NS 18h':1, 'Monocyte NS 6h':0, 'Monocyte NS 18h':0}
        if (tn == 3):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub(" [0-9]*$", "", str(k)) for k in atype]
            ahash = {'Monocyte NS 6h':0, 'Monocyte IFNG 6h':1,
                    'Monocyte IFNG 18h':2, 'Monocyte IL4 18h':2,
                    'Monocyte IL4 6h':2, 'Monocyte NS 18h':0}
            atypes = ['M0', 'M1', 'M2']
        self.initData(atype, atypes, ahash)

    def getReynier2012(self):
        self.prepareData("MAC43")
        atype = self.h.getSurvName("c treatment")
        atypes = ['saline', 'LPS']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getChamberland2009(self):
        self.prepareData("MAC44")
        atype = self.h.getSurvName("c src1")
        atypes = ['H', 'Asthma']
        ahash = {'control_bronchial biopsies':0,
                'allergic_asthmatic_bronchial biopsies':1}
        self.initData(atype, atypes, ahash)

    def getNascimento2009(self):
        self.prepareData("MAC45")
        atype = self.h.getSurvName("c src1")
        atypes = ['ND', 'DF', 'DHF']
        ahash = {'PBMCs from DF patient':1,
                'PBMCs from DHF patient':2,
                'PBMCs from ND patient':0}
        self.initData(atype, atypes, ahash)

    def getWu2013(self):
        self.prepareData("MAC46")
        atype = self.h.getSurvName("c disease status")
        atypes = ['HC', 'R', 'NR']
        ahash = {'virological failure':2,
                'HIV sero-negative healthy controls':0,
                'sustained virus suppression':1}
        self.initData(atype, atypes, ahash)

    def getWong2009(self, tn=1):
        self.prepareData("MAC47")
        ctype = self.h.getSurvName("c disease state")
        ttype = self.h.getSurvName("c outcome")
        atype = [ str(ctype[i]) + " " + str(ttype[i]) for i in
                                range(len(ctype))]
        atypes = ['HC', 'Alive', 'Dead']
        ahash = {'septic shock patient Survivor':1,
                'septic shock patient Nonsurvivor':2,
                'normal control Survivor':0,
                'normal control n/a':0}
        if (tn == 2):
            atypes = ['HC', 'Alive', 'Dead']
            ahash = {'septic shock patient Survivor':1,
                    'septic shock patient Nonsurvivor':2,
                    'normal control Survivor':0}
        self.initData(atype, atypes, ahash)

    def getYoon2011(self, tn=1):
        self.prepareData("MAC48")
        atype = self.h.getSurvName("c disease state")
        atypes = ['Normal', 'mild', 'severe', 'THP1']
        ahash = {'asthma severe':2, 'asthma mild':1, '':3, 'normal':0}
        if (tn == 2):
            atypes = ['H', 'Asthma']
            ahash = {'asthma severe':1, 'asthma mild':1, 'normal':0}
        self.initData(atype, atypes, ahash)

    def getVoraphani2014(self, tn=1):
        self.prepareData("MAC49")
        atype = self.h.getSurvName("c disease state")
        atypes = ['Control', 'Moderate', 'Severe']
        ahash = {'Severe Asthma':2, 'Moderate Asthma':1, 'Control':0}
        if (tn == 2):
            atypes = ['H', 'Asthma']
            ahash = {'Severe Asthma':1, 'Moderate Asthma':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getLund2003(self):
        self.prepareData("MAC50.1")
        atype = self.h.getSurvName("c Cell Type")
        atypes = ['NT1', 'NT2', 'IL4', 'IL12', 'IL12+TGFb', 'IL4+TGFb']
        ahash = {'antiCD3+antiCD28+IL4':2,
                'antiCD3+antiCD28':1,
                'antiCD3+antiCD28+IL12':3,
                'no treatment':0,
                'antiCD3+antiCD28+IL12+TGFbeta':4,
                'antiCD3+antiCD28+IL4+TGFbeta':5}
        self.initData(atype, atypes, ahash)

    def getLund2003II(self):
        self.prepareData("MAC50.2")
        atype = self.h.getSurvName("c Cell Type")
        atypes = ['NT1', 'NT2', 'IL4', 'IL12', 'IL12+TGFb', 'IL4+TGFb']
        ahash = {'antiCD3+antiCD28+IL4':2,
                'antiCD3+antiCD28':1,
                'antiCD3+antiCD28+IL12':3,
                'no treatment':0,
                'antiCD3+antiCD28+IL12+TGFbeta':4,
                'antiCD3+antiCD28+IL4+TGFbeta':5}
        self.initData(atype, atypes, ahash)

    def getYanez2019(self):
        self.prepareData("MAC51")
        atype = self.h.getSurvName("c culture condition")
        atypes = ['NT', 'Th0', 'Th1', 'Th2', 'Act']
        ahash = {
                'Stimulated under Th0 for 20 hours':1,
                'Stimulated under Th0 for 12 hours':1,
                'Stimulated under Th2 for 8 hours':3,
                'Stimulated for 16 hours and than Actinomysin added':4,
                'Stimulated under Th2 for 20 hours':3,
                'Purified Unstimulated CD4 T cells':0,
                'Stimulated under Th0 for 4 hours':1,
                'Stimulated under Th1 for 4 hours':2,
                'Stimulated under Th1 for 12 hours':2,
                'Stimulated under Th0 for 16 hours':1,
                'Stimulated under Th2 for 12 hours':3,
                'Stimulated under Th0 for 8 hours':1,
                'Stimulated under Th0 for 24 hours':1,
                'Stimulated under Th2 for 24 hours':3,
                'Stimulated under Th1 for 16 hours':2,
                'Stimulated under Th1 for 20 hours':2,
                'Stimulated under Th2 for 4 hours':3,
                'Stimulated under Th1 for 8 hours':2,
                'Stimulated under Th2 for 16 hours':3,
                'Stimulated under Th1 for 24 hours':2}
        self.initData(atype, atypes, ahash)

    def getSpurlock2015(self):
        self.prepareData("MAC52")
        atype = self.h.getSurvName("c polarizing conditions")
        atypes = ['TH1', 'TH2', 'TH17']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMicosse2019(self):
        self.prepareData("MAC53")
        atype = self.h.getSurvName("c Characteristics[cell type]")
        atypes = ['TH1', 'TH2', 'TH17', 'TH9']
        ahash = {'T-helper 1 cell':0,
                'T-helper 17 cell':2,
                'T-helper 2 cell':1,
                'T-helper 9 cell':3}
        self.initData(atype, atypes, ahash)

    def getGonzalezLeal2014(self, tn=1):
        self.prepareData("MAC54")
        atype = self.h.getSurvName("c treatment")
        atypes = ['CLIK', 'DMSO', 'CA074', 'pTH1', 'aTH1', 'pTH2', 'aTH2']
        ahash = {'pro TH2':5,
                'anti TH2':6,
                'CLIK control':0,
                'DMSO control':1,
                'CA074 control':2,
                'pro TH1':3,
                'anti TH1':4}
        if (tn == 2):
            atypes = ['C', 'TNFa']
            ahash = {'pro TH2':1,
                    'CLIK control':0,
                    'DMSO control':0,
                    'CA074 control':0}
        self.initData(atype, atypes, ahash)

    def getBelmont(self, tn=1):
        self.prepareData("PLP41")
        atype = self.h.getSurvName("c genotype")
        atypes = ['Normal', 'Adenoma', 'Cancer']
        atypes = ['Normal', 'Adenoma']
        ahash = {'wild type':0, 'APC mutant': 1, 'APC KRAS mutant': 2}
        ahash = {'wild type':0, 'APC mutant': 1}
        if (tn == 2):
            atype = self.h.getSurvName("c src1")
            atypes = ['Normal', 'Cancer']
            ahash = {'primary colon cancer':1, 'normal colon control':0}
        if (tn == 3):
            atype = self.h.getSurvName("c genotype")
            atypes = ['Normal', 'Cancer']
            ahash = {'wild type':0, 'APC BRAF mutant':1,
                    'APC BRAF P53 mutant':1}
        if (tn == 4):
            atype = self.h.getSurvName("c genotype")
            batch = self.h.getSurvName("c batch")
            atype = [str(atype[i])+ " " + str(batch[i]) for i in
                    range(len(atype))]
            atypes = ['Normal', 'Cancer']
            ahash = {'wild type 3':0, 'APC KRAS mutant 3': 1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        expg = [ i for i in self.h.aRange() if aval[i] is not None]
        self.normal = [ i for i in self.h.aRange() if aval[i] == 0]
        self.ad = [ i for i in self.h.aRange() if aval[i] == 1]
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = expg
        self.ibd = self.normal + self.ad
        self.printInfo()

    def getGerling2016(self, tn=1):
        self.prepareData("PLP42")
        atype = self.h.getSurvName("c genotype")
        atypes = ['td', 'fl/+', 'fl/fl', 'fl/fl td']
        ahash = {'R26-tdTomato':0,
                'Ptch1fl/+':1,
                'Col1a2CreER; Ptch1fl/fl; R26-LSL-tdTomato':3,
                'Col1a2CreER; Ptch1fl/fl':2}
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atype = [re.split("[ _]", str(i))[0] for i in atype]
            atypes = ['wt', 'Hh']
            ahash = {'wt': 0, 'Hh':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        expg = [ i for i in self.h.aRange() if aval[i] is not None]
        self.normal = [ i for i in self.h.aRange() if aval[i] == 0]
        self.ad = [ i for i in self.h.aRange() if aval[i] == 1]
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = expg
        self.ibd = self.normal + self.ad
        self.printInfo()

    def getMcNeil2016(self, tn=1):
        self.prepareData("PLP43")
        atype = self.h.getSurvName("c Stage")
        atypes = ['U', 'P', 'I', 'LT']
        ahash = {'p52':3, 'p5':1, 'p8':2, 'p51':3, 'p0':0, 'p3':1,
                'p66':3, 'p6':2, 'p54':3, 'p7':2, 'p55':3, 'p58':3}
        if (tn == 2):
            atypes = ['U', 'LT']
            ahash = {'p52':1, 'p51':1, 'p0':0,
                    'p66':1, 'p54':1, 'p55':1, 'p58':1}
        self.initData(atype, atypes, ahash)

    def getNeufert2013(self, tn=1):
        self.prepareData("PLP44")
        atype = self.h.getSurvName("c src1")
        atypes = ['CS', 'CC', 'TS', 'TC']
        ahash = {'colorectal control epithelium_colitis-associated':1,
                'colorectal tumor_colitis-associated':2,
                'colorectal control epithelium_sporadic':0,
                'colorectal tumor_sporadic':3}
        if (tn == 2):
            atypes = ['N', 'T']
            ahash = {'colorectal control epithelium_colitis-associated':0,
                    'colorectal tumor_colitis-associated':1,
                    'colorectal control epithelium_sporadic':0,
                    'colorectal tumor_sporadic':1}
        self.initData(atype, atypes, ahash)

    def getLeclerc2004(self):
        self.prepareData("PLP45")
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'T']
        ahash = {'Normal intestine':0, 'Tumor':1}
        self.initData(atype, atypes, ahash)

    def getZhu2016(self):
        self.prepareData("PLP46")
        atype = self.h.getSurvName("c sample type")
        atypes = ['N', 'T']
        ahash = {'Mouse tumor':1, 'Mouse primary cells':0}
        self.initData(atype, atypes, ahash)

    def getPaoni(self, tn=1):
        self.prepareData("PLP51")
        atype = self.h.getSurvName("c Title")
        atypes = ['Normal', 'Adenoma', 'Cancer']
        atypes = ['Normal', 'Adenoma']
        if (tn == 2):
            atypes = ['Normal', 'Adenoma', 'Cancer']
        ahash = {}
        aval = [ahash[i] if i in ahash else None for i in atype]
        normal = [ i for i in self.h.aRange() if atype[i].find("-WT") > 0]
        for i in normal:
            aval[i] = 0
        ad = [ i for i in self.h.aRange() if atype[i].find("-adenoma") > 0]
        for i in ad:
            aval[i] = 1
        ca = [ i for i in self.h.aRange() if atype[i].find("-carc") > 0]
        for i in ca:
            aval[i] = 2
        self.aval = aval
        self.atype = atype
        self.atypes = atypes
        self.order = normal + ad
        if (tn == 2):
            self.order = normal + ad + ca
        self.normal = normal
        self.ad = ad
        self.ca = ca
        self.ibd = self.normal + self.ad
        self.printInfo()

    def getKaiser2007(self):
        self.prepareData("PLP54")
        atype = self.h.getSurvName("c src1")
        atypes = ['b', 'C']
        ahash = {' b':0, '3 b':0, '2 b':0, 'Colon tissue':1, '1 b':0}
        self.initData(atype, atypes, ahash)

    def getNoble2010(self, tn=1):
        self.prepareData("PLP58")
        atype = self.h.getSurvName("c src1")
        atypes = ['A', 'D', 'S', 'T', 'A_CD', 'D_CD', 'S_CD', 'T_CD']
        ahash = {'ascending colon biopsy from healthy subject':0,
                'descending colon biopsy from healthy subject':1,
                'sigmoid colon biopsy from healthy subject':2,
                'terminal ileum biopsy from healthy subject':3,
                'ascending colon biopsy from crohns disease subject':4,
                'descending colon biopsy from crohns disease subject':5,
                'sigmoid colon biopsy from crohns disease subject':6,
                'terminal ileum biopsy from crohns disease subject':7}
        if (tn == 2):
            atypes = ['A', 'D']
            ahash = {'ascending colon biopsy from healthy subject':0,
                    'descending colon biopsy from healthy subject':1}
        self.initData(atype, atypes, ahash)

    def getColonGEO(self):
        self.prepareData("CRC115")
        self.normal = normal
        self.ad = ad
        self.ca = ca
        self.ibd = self.normal + self.ad
        self.printInfo()

    def getColonGEO(self):
        self.prepareData("CRC115")
        atype = self.h.getSurvName("c Histology")
        atypes = ['Normal', 'Adenoma', 'Carcinoma']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getSchridde2017(self):
        self.prepareData("MAC55")
        atype = self.h.getSurvName("c desc")
        atype = [str(i).split(" ")[4] if len(str(i).split(" ")) > 5 \
                else None for i in atype]
        atypes = ['P1', 'P2', 'P3', 'P4']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getQu2016(self, tn=1):
        self.prepareData("PLP50")
        atype = self.h.getSurvName("c tissue type")
        atypes = ['NS', 'NC', 'A', 'C', 'M']
        ahash = {'Metastasis':4,
                'Carcinoma':3,
                'Normal crypt epithelium':1,
                'Adenoma':2,
                'Normal surface epithelium':0}
        if (tn == 2):
            atypes = ['NC', 'NS']
            ahash = {'Normal crypt epithelium':0,
                    'Normal surface epithelium':1}
        self.initData(atype, atypes, ahash)

    def getGalamb2008(self):
        self.prepareData("PLP83")
        atype = self.h.getSurvName("c desc")
        atype = [ " ".join(str(i).split(" ")[-2:]) for i in atype]
        atypes = ['N', 'IBD', 'A', 'C']
        ahash = {'bowel disease':1,
                'colorectal cancer':3,
                'colon adenoma':2,
                'healthy control':0}
        self.initData(atype, atypes, ahash)

    def getArendt(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV4")
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['HC', 'NASH']
        ahash = {'HC':0, 'NASH':1}
        self.initData(atype, atypes, ahash)  
        return

    def getTung2011(self):
        self.prepareData("LIV1")
        atype = self.h.getSurvName("c src1")
        atypes = ['H', 'NT', 'C', 'HCC']
        ahash = {'non_tumor':1, 'tumor':3, 'cirrhotic':2, 'healthy':0}
        self.initData(atype, atypes, ahash)

    def getBaker2010(self):
        self.prepareData("LIV7")
        atype = self.h.getSurvName("c disease state")
        atypes = ['H', 'SH']
        ahash = {'non-alcoholic steatohepatitis (NASH)':1,
                'normal (control)':0}
        self.initData(atype, atypes, ahash)

    def getAlao2016(self):
        self.prepareData("LIV16")
        atype = self.h.getSurvName("c treatment")
        atypes = ['Base', 'Wk2', 'Wk4']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMeissner2016HCV(self, tn=1):
        self.prepareData("LIV17")
        atype = self.h.getSurvName("c time point")
        ahash = {'pre-treatment':0, 'post-treatment':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c tissue")
        ahash = {'blood':0, 'liver':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['none', 'simtuzumab']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMeissner2016Cir(self, tn=1):
        self.prepareData("LIV18")
        atype = self.h.getSurvName("c treatment time")
        atypes = ['Pre', 'Post-DAA']
        ahash = {'EOT':1, 'PRE':0}
        if (tn == 2):
            paired = self.h.getSurvName('c paired mate')
            paired = [re.sub("sample ", "", str(k)) for k in paired]
            phash = {'1', '5', '6', '3', '11', '15', '16', '13'}
            atype = [atype[i] if paired[i] in phash else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPeters2017(self, dtype=0):
        self.prepareData("PLP18")
        atype = self.h.getSurvName("c src1")
        ahash = {'Transverse colon':0, 'Ascending colon':0, 'Rectum':0,
                'Descending colon':0, 'Sigmoid colon':0, 'Normal':0,
                'Terminal Ileum':1, 'Blood':2,
                'DELETED':3, 'Not Available':3, 'Not Collected':3}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c inflammation")
        atypes = ['Normal', 'NI', 'I', 'NA']
        ahash = {'Non-involved area':1, 'NA':3, 'na':3, 'Inflamed area':2,
                'Normal':0, 'DELETED':3, 'N/A (Sample not received)':3}
        self.initData(atype, atypes, ahash)

    def getKF(self):
        self.prepareData("CRC11")
        kras = self.h.getSurvName('c KRAS Mutation')
        khash = {'WT':1, 'c.35G>T':2, 'c.35G>A':2, 'c.35G>C':2,
                'c.38G>A':2, 'c.34G>A':2}
        kras_m = [khash[i] if i in khash else 0 for i in kras]
        atype = self.h.getSurvName('c Best Clinical Response Assessment')
        atypes = ['DCG', 'PD']
        ahash = {"CR": 0, "PR":0, "SD":0, "CR + PR":0, "PD":1}
        self.rval = kras_m
        self.initData(atype, atypes, ahash)

    def getDelRoy(self):
        self.prepareData("CRC101")
        atype = self.h.getSurvName('c Status')
        atypes = ['R', 'NR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLissner(self, tn=1):
        self.prepareData("MAC56")
        atype = self.h.getSurvName('c Agent')
        ahash = {'Lm':0, 'LPS':1}
        bval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Timepoint')
        ahash = {'6hr':6, '2hr':2, '1hr':1, '0hr':0}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Type')
        atypes = ['N', 'A', 'O']
        ahash = {'Neonate':0, 'Adult':1, 'OlderAdult':2}
        if (tn == 2):
            atype = [atype[i] if bval[i] == 0 and rval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if bval[i] == 1 and rval[i] <= 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHennenberg(self):
        self.prepareData("PLP100")
        atype = self.h.getSurvName('c src1')
        atypes = ['WT', 'KO']
        ahash = {'MDR1A KO, Tumor, Colon, AOM/DSS':1,
                'WT, Tumor, Colon, AOM/DSS':0}
        self.initData(atype, atypes, ahash)

    def getTakahara2015(self, ri=0):
        self.prepareData("PLP47")
        atype = self.h.getSurvName('c genotype')
        ahash = {'WT':0, 'ATF6bKO':1, 'ATF6aKO':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['Un', 'DSS']
        ahash = {'DSS for 3 days':1, 'untreated':0}
        atype = [atype[i] if rval[i] == ri else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDevHs(self):
        self.prepareData("PLP103")
        atype = self.h.getSurvName('c Type')
        atypes = ['YS', 'Pl', 'ES', 'ST', 'SI', 'CO']
        ahash = {'yolk sac':0, 'placenta':1,
                'esophagus':2, 'stomach':3,
                'small intestine':4, 'colon':5}
        self.initData(atype, atypes, ahash)

    def getDevMm(self):
        self.prepareData("PLP104")
        atype = self.h.getSurvName('c Type')
        atypes = ['YS', 'E13.5 Ca', 'E13.5 I', 'E13.5 Co']
        ahash = {'yolk sac':0, 'Caecum':1, 'Ileum':2, 'Colon':3}
        self.initData(atype, atypes, ahash)

    def getSmith2009(self, ri=0):
        self.prepareData("MAC57")
        atype = self.h.getSurvName('c Sample Characteristic[stimulus]')
        ahash = {'none':0,
                'heat-killed Escherichia coli':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Sample Characteristic[disease]')
        atypes = ['N', 'iCD', 'cCD']
        ahash = {'normal':0,
                "ileal Crohn's disease":1,
                "colonic Crohn's disease":2}
        self.initData(atype, atypes, ahash)

    def getDeSchepper2018(self):
        self.prepareData("MAC58")
        atype = self.h.getSurvName('c Factor Value[organism part]')
        ahash = {'lamina propria of small intestine':0,
                'muscularis externa layer of small intestine':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Characteristics[phenotype]')
        atypes = ['LPMR', 'LPSM', 'MEMR', 'MESM']
        ahash = {'YFP negative':0, 'YFP positive':1}
        self.initData(atype, atypes, ahash)

    def getGrover2018(self):
        self.prepareData("PLP105")
        atype = self.h.getSurvName('c sample type')
        atypes = ['NDNG', 'DNG', 'DG', 'IG']
        ahash = {'diabetic gastroparetics':2,
                'non-diabetic non-gastroparetic':0,
                'diabetic non-gastroparetics controls':1,
                'idiopathic gastroparetics':3}
        self.initData(atype, atypes, ahash)

    def getXu2014(self):
        self.prepareData("GS2.7")
        atype = self.h.getSurvName('c src1')
        atypes = ['LGD', 'HGD', 'inf', 'EGC']
        ahash = {'inflammation':2, 'HGD':1, 'EGC':3, 'LGD':0}
        self.initData(atype, atypes, ahash)

    def getBadawi2019(self, tn=1):
        self.prepareData("MHP5")
        atype = self.h.getSurvName('c time post surgery')
        atypes = ['0', '45m', '24h']
        ahash = {'45 minutes':1, '24 hours':2, '0 minutes':0}
        if (tn == 2):
            atypes = ['0', '24h']
            ahash = {'45 minutes':0, '24 hours':1, '0 minutes':0}
        self.initData(atype, atypes, ahash)

    def getRock2005(self):
        self.prepareData("MAC73")
        atype = self.h.getSurvName('c Group')
        atypes = ['Control', 'IFNG']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getPolak2014(self, tn=1):
        self.prepareData("MAC69")
        atype = self.h.getSurvName('c treatment')
        ahash = {'TNF-alpha':0, 'TSLP':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c time')
        atypes = ['0', '2', '8', '24']
        ahash = {'8h':2, '2h':1, '24h':3, '0h':0}
        if (tn == 2):
            atypes = ['C', 'TNF-a']
            ahash = {'8h':1, '24h':1, '0h':0}
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSchirmer2009(self, tn=1):
        self.prepareData("KR5")
        atype = self.h.getSurvName('c Disease Status')
        ahash = {'control':0, 'patient':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c src1')
        atypes = ['RM', 'T', 'SM', 'SC', 'M']
        ahash = {'resting monocyes':0,
                'T cells':1,
                'stimulated monocyes':2,
                'stem cells':3,
                'macrophages':4}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPritchard2011(self):
        self.prepareData("KR7")
        atype = self.h.getSurvName('c src1')
        atypes = ['C', 'Ath']
        ahash = {'Male Subject with Carotid Disease':1, 'Male Control':0}
        self.initData(atype, atypes, ahash)

    def getHagg2008(self):
        self.prepareData("KR1")
        atype = self.h.getSurvName('c Sample Type')
        atypes = ['MC', 'MD', 'FC', 'FD']
        ahash = {'Baseline macrophages without atherosclerosis':0,
                'Baseline macrophages with atherosclerosis':1,
                'Foam cells with atherosclerosis':3,
                'Foam cells without atherosclerosis':2}
        self.initData(atype, atypes, ahash)

    def getBondar2017(self, tn=1):
        self.prepareData("MAC76")
        atype = self.h.getSurvName('c gender')
        ahash = {'Female':0, 'Male':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c underlying disease')
        atypes = ['NICM', 'ICM', 'PPCM', 'NCIM', 'ChemoCM']
        if (tn == 2):
            atypes = ['NICM', 'ICM']
        if (tn == 3):
            atypes = ['NICM', 'ICM']
            atype = [atype[i] if tval[i] == 1 else None
                for i in range(len(atype))]
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMaciejak2015(self, tn=1):
        self.prepareData("MAC75")
        atype = self.h.getSurvName('c samples collection')
        ahash = {'on the 1st day of MI (admission)':1,
                '1 month after MI':3, 'after 4-6 days of MI (discharge)':2,
                '6 months after MI':4, 'N/A':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c hf progression')
        atypes = ['non-HF', 'HF', 'stable CAD']
        if (tn == 2):
            atypes = ['non-HF', 'HF']
        if (tn == 3):
            atypes = ['non-HF', 'HF']
            atype = [atype[i] if tval[i] == 3 else None
                for i in range(len(atype))]
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getNorona2019(self, tn=1):
        self.prepareData("MAC78")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'TGF', 'MTX']
        ahash = {'TGF':1, 'MTX':2, 'VEH':0}
        if (tn == 2):
            atypes = ['C', 'TGF']
            ahash = {'TGF':1, 'VEH':0}
        self.initData(atype, atypes, ahash)

    def getEijgelaar2010(self, tn=1):
        self.prepareData("MAC79")
        atype = self.h.getSurvName('c Sample')
        atypes = ['LI', 'LU', 'SP', 'CA']
        if (tn == 2):
            atypes = ['CA', 'LI']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getKeller2009(self, tn=1):
        self.prepareData("MAC80")
        atype = self.h.getSurvName('c time')
        atype = [str(k).split(" ")[-1] for k in atype]
        atypes = ['CT0', 'CT4', 'CT8', 'CT12', 'CT16', 'CT20']
        if (tn == 2):
            atype = self.h.getSurvName('c time')
            atype = [str(k).split(" ")[0] for k in atype]
            atypes = ['first', 'second']
        ahash = {}
        if (tn == 3):
            atype = self.h.getSurvName('c time')
            atypes = ['f0', 'f4', 'f8', 'f12', 'f16', 'f20',
                    's0', 's4', 's8', 's12', 's16', 's20']
            ahash = {'first day CT0':0,
                    'first day CT4':1,
                    'first day CT8':2,
                    'first day CT12':3,
                    'first day CT16':4,
                    'first day CT20':5,
                    'second day CT0':6,
                    'second day CT4':7,
                    'second day CT8':8,
                    'second day CT12':9,
                    'second day CT16':10,
                    'second day CT20':11}
        self.initData(atype, atypes, ahash)

    def getImmGenULI(self, tn=1):
        self.prepareData("MAC81")
        atype = self.h.getSurvName('c Title')
        atypes = ['M0', 'M1']
        ahash = {}
        if (tn == 1):
            ahash = {'MF.11cpSigFp.BAL.1':0,
                    'MF.11cpSigFp.BAL.2':0,
                    'MF.SSChipSigFn.LPS.d3.BAL.1':1,
                    'MF.SSChipSigFn.LPS.d3.BAL.2':1,
                    'MF.11cpSigFp.LPS.d6.BAL.1':1,
                    'MF.11cpSigFp.LPS.d6.BAL.2':1,
                    'MF.SSChipSigFn.LPS.d6.BAL.1':1,
                    'MF.SSChipSigFn.LPS.d6.BAL.2':1}
            ahash = {'MF.11cpSigFp.BAL.1':0,
                    'MF.11cpSigFp.BAL.2':0,
                    'MF.SSChipSigFn.LPS.d3.BAL.1':1,
                    'MF.SSChipSigFn.LPS.d3.BAL.2':1}
        if (tn == 2):
            atypes = ['M0', 'M1']
            ahash = {'MF.64p6Cn206nIIp.LPS.d3.Lu.1':1,
                    'MF.64p6Cn206nIIp.LPS.d3.Lu.2':1,
                    'MF.64p6Cn206nIIp.LPS.d6.Lu.1':1,
                    'MF.64p6Cn206nIIp.LPS.d6.Lu.2':1,
                    'MF.64p6Cn206nIIp.Lu.1':0,
                    'MF.64p6Cn206nIIp.Lu.2':0}
        if (tn == 3):
            atypes = ['M0', 'M1']
            ahash = {'MF.64p6Cn206pIIn.LPS.d3.Lu.1':1,
                    'MF.64p6Cn206pIIn.LPS.d3.Lu.2':1,
                    'MF.64p6Cn206pIIn.LPS.d6.Lu.1':1,
                    'MF.64p6Cn206pIIn.LPS.d6.Lu.2':1,
                    'MF.64p6Cn206pIIn.Lu.1':0,
                    'MF.64p6Cn206pIIn.Lu.2':0}
        if (tn == 4):
            atypes = ['M0', 'M1']
            ahash = {'MF.F.10kIFN.PC#1':1,
                    'MF.F.10kIFN.PC#2':1,
                    'MF.F.10kIFN.PC#3':1,
                    'MF.F.PC#1':0,
                    'MF.F.PC#1_':0,
                    'MF.F.PC#2':0,
                    'MF.F.PC#2_':0,
                    'MF.F.PC#3':0,
                    'MF.F.PC.1':0,
                    'MF.F.PC.2':0,
                    'MF.F.PC.3':0,
                    'MF.Fem.PC#1_RNA-seq':0,
                    'MF.Fem.PC#2_RNA-seq':0}
        if (tn == 5):
            atypes = ['M0', 'M1']
            ahash = {'MF.M.10kIFN.PC#1':1,
                    'MF.M.10kIFN.PC#2':1,
                    'MF.M.10kIFN.PC#3':1,
                    'MF.M.PC#1':0,
                    'MF.M.PC#2':0,
                    'MF.M.PC#3':0}
            for k in ['MF.PC#1.1', 'MF.PC#1.10', 'MF.PC#1.11', 'MF.PC#1.12',
                    'MF.PC#1.13', 'MF.PC#1.14', 'MF.PC#1.15', 'MF.PC#1.16',
                    'MF.PC#1.2', 'MF.PC#1.3', 'MF.PC#1.4', 'MF.PC#1.5', 'MF.PC#1.6',
                    'MF.PC#1.7', 'MF.PC#1.8', 'MF.PC#1.9', 'MF.PC#1_RNA-seq',
                    'MF.PC#2.1', 'MF.PC#2.2', 'MF.PC#2.3', 'MF.PC#2.4', 'MF.PC#2.5',
                    'MF.PC#2.6', 'MF.PC#2.7', 'MF.PC#2.8', 'MF.PC#2_RNA-seq',
                    'MF.PC#3', 'MF.PC#3_RNA-seq', 'MF.PC#4', 'MF.PC#4_RNA-seq',
                    'MF.PC.01', 'MF.PC.02', 'MF.PC.03', 'MF.PC.04', 'MF.PC.05',
                    'MF.PC.06', 'MF.PC.07', 'MF.PC.08', 'MF.PC.09', 'MF.PC.10',
                    'MF.PC.11', 'MF.PC.12', 'MF.PC.13', 'MF.PC.14', 'MF.PC.15',
                    'MF.PC.17', 'MF.PC.18', 'MF.PC.19', 'MF.PC.20', 'MF.PC.21',
                    'MF.PC.23', 'MF.PC.24', 'MF.PC.25', 'MF.PC.26', 'MF.PC.37'
                    'MF.PC.38', 'MF.PC.39', 'MF.PC.40']:
                ahash[k] = 0
        if (tn == 6):
            atypes = ['M0', 'M1']
            ahash = {'Mo.6Chi11bp.APAP.36h.Lv.1':1,
                    'Mo.6Chi11bp.APAP.36h.Lv.2':1,
                    'Mo.6Chi11bp.APAP.36h.Lv.3':1,
                    'Mo.6Chi11bp.APAP.36h.Lv.4':1,
                    'Mo.6Chi11bp.PBS.Lv.1':0,
                    'Mo.6Chi11bp.PBS.Lv.2':0,
                    'Mo.6Chi11bp.PBS.Lv.3':0,
                    'Mo.6Chi11bp.PBS.Lv.4':0}
        if (tn == 7):
            atypes = ['M0', 'M1']
            ahash = {
                    'NKT.F.Sp#1':0,
                    'NKT.F.Sp#2':0,
                    'NKT.F.Sp#3':0,
                    'NKT.M.Sp#1':0,
                    'NKT.M.Sp#2':0,
                    'NKT.M.Sp#3':0,
                    'NKT.Sp#3_RNA-seq':0,
                    'NKT.Sp.LPS.18hr#1_RNA-seq':1,
                    'NKT.Sp.LPS.18hr#2_RNA-seq':1,
                    'NKT.Sp.LPS.3hr#1_RNA-seq':1,
                    'NKT.Sp.LPS.3hr#2_RNA-seq':1}
        if (tn == 8):
            atypes = ['M0', 'M1']
            ahash = {
                    'T4.F.10kIFN.Sp#1':1,
                    'T4.F.10kIFN.Sp#2':1,
                    'T4.F.10kIFN.Sp#3':1,
                    'T4.F.Sp#1':0,
                    'T4.F.Sp#2':0,
                    'T4.F.Sp#3':0,
                    'T4.M.10kIFN.Sp#1':1,
                    'T4.M.10kIFN.Sp#2':1,
                    'T4.M.10kIFN.Sp#3':1,
                    'T4.M.Sp#1':0,
                    'T4.M.Sp#2':0,
                    'T4.M.Sp#3':0}
        if (tn == 9):
            atypes = ['M0', 'M1']
            ahash = {
                    'B.17m.F.Sp#1':0,
                    'B.20m.Sp#1':0,
                    'B.2m.F.Sp#1':0,
                    'B.2m.Sp#1':0,
                    'B.6m.F.Sp#1':0,
                    'B.6m.F.Sp#2':0,
                    'B.6m.Sp#1':0,
                    'B.6m.Sp#2':0,
                    'B.F.10kIFN.Sp#1':1,
                    'B.F.10kIFN.Sp#2':1,
                    'B.F.10kIFN.Sp#3':1,
                    'B.F.1kIFN.Sp#1':1,
                    'B.F.1kIFN.Sp#2':1,
                    'B.F.1kIFN.Sp#3':1,
                    'B.F.Sp#1':0,
                    'B.F.Sp#1_':0,
                    'B.F.Sp#2':0,
                    'B.F.Sp#2_':0,
                    'B.F.Sp#3':0,
                    'B.Fem.Sp#1_RNA-seq':0,
                    'B.Fo.Sp#1_RNA-seq':0,
                    'B.Fo.Sp#2_RNA-seq':0,
                    'B.Fo.Sp#3_RNA-seq':0,
                    'B.Fo.Sp#4_RNA-seq':0}
        if (tn == 10):
            atypes = ['M0', 'M1']
            ahash = {
                    'GN.17m.F.Sp#1':0,
                    'GN.20m.Sp#1':0,
                    'GN.F.10kIFN.Sp#1':1,
                    'GN.F.10kIFN.Sp#2':1,
                    'GN.F.10kIFN.Sp#3':1,
                    'GN.F.Sp#1':0,
                    'GN.F.Sp#2':0,
                    'GN.F.Sp#3':0,
                    'GN.M.10kIFN.Sp#1':1,
                    'GN.M.10kIFN.Sp#2':1,
                    'GN.M.10kIFN.Sp#3':1,
                    'GN.M.Sp#1':0,
                    'GN.M.Sp#2':0,
                    'GN.M.Sp#3':0,
                    'GN.Sp#3_RNA-seq':0,
                    'GN.Sp#4_RNA-seq':0}
        if (tn == 11):
            atypes = ['M0', 'M1', 'M2']
            ahash = {
                    'MF.KC.Clec4FpTim4p64p.APAP.12h.Lv.1':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.12h.Lv.2':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.12h.Lv.4':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.36h.Lv.1':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.36h.Lv.2':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.36h.Lv.3':2,
                    'MF.KC.Clec4FpTim4p64p.APAP.36h.Lv.4':2,
                    'MF.KC.Clec4FpTim4p64p.Lv.2':0,
                    'MF.KC.Clec4FpTim4p64p.Lv.3':0,
                    'MF.KC.Clec4FpTim4p64p.Lv.4':0,
                    'MF.KC.Clec4FpTim4p64p.PBS.Lv.1':0,
                    'MF.KC.Clec4FpTim4p64p.PBS.Lv.2':0,
                    'MF.KC.Clec4FpTim4p64p.PBS.Lv.3':0,
                    'MF.KC.Clec4FpTim4p64p.PBS.Lv.4':0}
        self.initData(atype, atypes, ahash)

    def getZigmond2014(self, tn=1):
        self.prepareData("MAC82")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['C', 'Il10', 'Il10ra']
        ahash = {'wild type; Cx3cr1gfp/+':0,
                'Interleukin-10 deficient; IL10-/- CX3CR1gfp/+':1,
                'macrophage-restricted interleukin-10 receptor deficient; CX3CR1cre:IL10Raflox/flox':2}
        self.initData(atype, atypes, ahash)

    def getRamanan2016(self, tn=1):
        self.prepareData("PLP108")
        atype = self.h.getSurvName('c Title')
        gtype = [str(k).split(" ")[0] if len(str(k).split(" ")) > 0 else None
                for k in atype]
        ttype = [str(k).split(" ")[1] if len(str(k).split(" ")) > 1 else None
                for k in atype]
        atype = [ str(gtype[i]) + " " + str(ttype[i]) for i in
                range(len(atype))]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Nod2-/-, IL-13':2,
                'Nod2-/-, untreated,':0}
        self.initData(atype, atypes, ahash)

    def getMishra2019(self, tn=1):
        self.prepareData("MAC83")
        atype = self.h.getSurvName('c treatment')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'exposed to IL4/IL13':2,
                'exposed to LPS/IL4/IL13':0,
                'exposed to IFN gamma/LPS':1}
        self.initData(atype, atypes, ahash)

    def getGuler2015(self, tn=1):
        self.prepareData("MAC84.3")
        atype = self.h.getSurvName('c Title')
        atype = ["_".join(str(k).split("_")[0:-2]) 
                if len(str(k).split("_")) > 2 else None for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'IFNg':1,
                'IL4IL13':2,
                'M.tb_IL4IL13':0,
                'M.tb':1,
                'Ust':0,
                'M.tb_IFNg':1,
                'M.tb_IL41L13':0}
        if (tn == 2):
            atype = self.h.getSurvName('c Title')
            atype = ["_".join(str(k).split("_")[0:-1]) 
                    if len(str(k).split("_")) > 2 else None for k in atype]
            ahash = {
                    'Ust_28h':0,
                    'IL4IL13_28h':2,
                    'M.tb_IFNg_28h':1,
                    'IFNg_28h':1,
                    'M.tb_28h':1}
        self.initData(atype, atypes, ahash)

    def getZhang2010(self, tn=1):
        self.prepareData("MAC85")
        atype = self.h.getSurvName('c Title')
        atype = [str(k).split("_")[0] if len(str(k).split("_")) > 0 else None
                for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'IFNG':1,
                'TNF':1,
                'L.':1,
                'Control':0,
                'IFNB':1,
                'IL4':2,
                'LPS':1,
                'T.':1,
                'IL10':2,
                'IL17':2}
        if (tn == 2):
            atype = self.h.getSurvName('c Title')
            atype = [re.sub("_rep.*", "", str(k)) for k in atype]
            ahash = {'IFNG_12h':1,
                    'TNF_12h':1,
                    'Control_6h':0,
                    'IFNB_24h':1,
                    'Control_2h':0,
                    'Control_12h':0,
                    'LPS_6h':1,
                    'Control_24h':0,
                    'IL4_12h':2,
                    'LPS_2h':1,
                    'IFNB_2h':1,
                    'Control_0h_for_IL10':0,
                    'Control_0h_for_Control':0,
                    'TNF_6h':1,
                    'IFNG_24h':1,
                    'LPS_12h':1,
                    'IL10_24h':2,
                    'Control_0h_for_IL4':0,
                    'Control_0h_for_IL17':0,
                    'IL17_6h':2,
                    'IFNB_12h':1,
                    'Control_0h_for_TNF':0,
                    'IL10_12h':2,
                    'IL4_6h':2,
                    'IFNB_6h':1,
                    'IL4_2h':2,
                    'LPS_24h':1,
                    'IL17_12h':2,
                    'IL17_2h':2,
                    'IL10_2h_2nd_scan':2,
                    'Control_0h_for_LPS':0,
                    'IFNG_6h':1,
                    'IL10_6h':2,
                    'Control_0h_for_IFNG':0,
                    'IFNG_2h':1,
                    'LPS_12h_2nd_scan':1,
                    'TNF_2h':1,
                    'Control_0h_for_IFNB':0,
                    'IL17_24h':2,
                    'IL10_2h':2,
                    'TNF_24h':1}
        self.initData(atype, atypes, ahash)

    def getHealy2016(self, tn=1):
        self.prepareData("MAC86")
        atype = self.h.getSurvName('c src1')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Human adult brain-derived microglia, M2a':2,
                'Human adult brain-derived microglia, M1':1,
                'Human adult brain-derived microglia, M2c':2,
                'Human adult brain-derived microglia, M0':0,
                'Human adult brain-derived microglia, Mtgf':2}
        if (tn == 2):
            ahash = {'Human adult peripheral blood-derived macrophages, M1':1,
                    'Human adult peripheral blood-derived macrophages, M2a':2,
                    'Human adult peripheral blood-derived macrophages, M0':0,
                    'Human adult peripheral blood-derived macrophages, M2c':2,
                    'Human adult peripheral blood-derived macrophages, Mtgf':2}
        self.initData(atype, atypes, ahash)

    def getSvensson2011(self, tn=1):
        self.prepareData("MAC87")
        atype = self.h.getSurvName('c factors')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M-CSF + E/P/IL-10/4/13':2, 'M-CSF + IL-10':2,
                'GM-CSF':0, 'M-CSF':0, 'M-CSF + IL4/13':2,
                'GM-CSF + M-CSF':0, 'GM-CSF/LPS/IFN':1, 'GM-CSF + IL4/13':2,
                'GM-CSF + IL-10':2, 'Decidual macrophages':0, 'Blood monocytes':0,
                'GM-CSF + M-CSF +IL-10':2, 'M-CSF + M-CSF':0, 'GM-CSF/LPS/IFN D6':1,
                'GM-CSF + E/P/IL-10/4/13':2}
        if (tn == 2):
            atypes = ['GMCSF', 'MCSF']
            ahash = {'GM-CSF':0, 'GM-CSF + M-CSF':0, 'M-CSF':1, 'M-CSF + M-CSF':1}
        self.initData(atype, atypes, ahash)

    def getChandriani2014(self, tn=1):
        self.prepareData("MAC88")
        source = self.h.getSurvName('c src1')
        atype = self.h.getSurvName('c treatment')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'unstimulated':0, 'IL13':2, 'TGFb':1, 'IL10':2, 'IL4':2,
                'Dex':1}
        if (tn == 2):
            atype = source
            ahash = {'Monocytes, unstimulated, 24h':0, 'Monocytes, IL13, 24h':2,
                    'Monocytes, unstimulated, 6h':0, 'Monocytes, IL4, 24h':2,
                    'Monocytes, IL4, 6h':2, 'Monocytes, IL13, 6h':2}
        if (tn == 3):
            atype = source
            ahash = {'Macrophages, unstimulated, 24h':0, 'Macrophages, IL13, 24h':2,
                    'Macrophages, IL10, 24h':2, 'Macrophages, TGFb, 24h':1,
                    'Macrophages, IL4, 24h':2, 'Macrophages, Dex, 24h':1}
        if (tn == 4):
            atype = source
            ahash = {'Normal lung fibroblasts, TGFb, 24h':1,
                    'Normal lung fibroblasts, IL13, 24h':2,
                    'Normal lung fibroblasts, IL4, 24h':2,
                    'Normal lung fibroblasts, unstimulated, 24h':0}
        if (tn == 5):
            atype = source
            atypes = ['Mono', 'Mac']
            ahash = {'Monocytes, unstimulated, 24h':0,
                    'Monocytes, unstimulated, 6h':0,
                    'Macrophages, unstimulated, 24h':1}
        self.initData(atype, atypes, ahash)

    def getMartinez2015(self, tn=1):
        self.prepareData("MAC89")
        atype = self.h.getSurvName('c src1')
        atypes = ['M0', 'M1', 'M2']
        ahash = { 'Monocyte-derived macrophages polarized with IL-4 for 5 days':2,
                'Monocyte-derived macrophages polarized with IL-10 for 5 days':2,
                'Monocyte-derived macrophages polarized with IFNgTNFa for 5 days':1,
                'Monocyte-derived macrophages':0}
        self.initData(atype, atypes, ahash)

    def getDas2018(self, tn=1):
        self.prepareData("MAC90")
        atype = self.h.getSurvName('c src1')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'cultured for 4 hrs':0,
                'treated with IFN-\xce\xb3 (100 U/ml) and LPS (100 ng/ml) for 12 hrs':1,
                'treated with IFN-\xce\xb3 (100 U/ml) and LPS (100 ng/ml) for 4 hrs':1,
                'treated with IFN-\xce\xb3 (100 U/ml) and LPS (100 ng/ml) for 24 hrs':1,
                'treated with IFN-\xce\xb3 (100 U/ml) and LPS (100 ng/ml) for 1 hr':1,
                'treated with LPS (100 ng/ml) for 4 hrs':1,
                'treated with IL-13 (10\xe2\x80\x89ng/ml) for 12 hrs':2,
                'treated with IL-4 (10\xe2\x80\x89ng/ml) for 12 hrs':2}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getDaniel2018(self, tn=1):
        self.prepareData("MAC91")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(".rep.*", "", str(k)) for k in atype]
        atype = [re.sub("mm_BMDM_", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Wt_2nd_stim_ctrl_RNA':0,
                'Wt_1st_stim_3hIL4_RNA':2,
                'PpargKO_2nd_stim_3hIL4_RNA':2,
                'ctrl_24hVeh_RNA':0,
                'PpargKO_1st_stim_3hIL4_RNA':2,
                'PpargKO_1st_stim_ctrl_RNA':0,
                'Wt_1st_stim_ctrl_RNA':0,
                'Wt_2nd_stim_3hIL4_RNA':2,
                'PpargKO_2nd_stim_ctrl_RNA':0}
        self.initData(atype, atypes, ahash)

    def getPiccolo2017(self, tn=1):
        self.prepareData("MAC92")
        atype = self.h.getSurvName('c Name')
        atype = [re.sub("_R.*", "", str(k)) for k in atype]
        atype = [re.sub("^_", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'IL4_4h':2,
                'IFNy_2h':1,
                'UT':0,
                'IFNy_IL4_2h':1,
                'IL4_2h':2,
                'IFNy_4h':1,
                'IFNy_IL4_4h':1}
        if (tn == 2):
            atype = self.h.getSurvName('c Name')
            atype = [re.sub("_R.*", "", str(k)) for k in atype]
            ahash = {'_shMYC_UT':0,
                    '_scramble_IL-4_4h':2,
                    '_scramble_IL-4_2h':2,
                    '_scramble_UT':0,
                    '_shMYC_IL-4_4h':2,
                    '_shMYC_IL-4_2h':2}
            if (tn == 3):
                atype = self.h.getSurvName('c Name')
            atype = [re.sub("_R.*", "", str(k)) for k in atype]
            ahash = {'shCEBP-beta_UT':0,
                    'shJunB_UT':0,
                    'shJunB_IFNy_4h':1,
                    'scramble_UT':0,
                    'scramble_IFNy_4h':1,
                    'shCEBP-beta_IFNy_4h':1}
        self.initData(atype, atypes, ahash)

    def getOstuni2013(self, tn=1):
        self.prepareData("MAC93")
        atype = self.h.getSurvName('c treatment')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'TGFb1 (1 ng/ml) for 4hrs':1,
                'IL4 (10 ng/ml) for 4hrs':2,
                'No treatment':0,
                'IFNg (100 ng/ml) for 4hrs':1,
                'TNFa (10 ng/ml) for 4hrs':1}
        self.initData(atype, atypes, ahash)

    def getRochaResende(self, tn=1):
        self.prepareData("MAC94.2")
        atype = self.h.getSurvName('c treatment')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'none':0, 'LPS':1, 'IL-4':2}
        self.initData(atype, atypes, ahash)

    def getHill2018(self, tn=1):
        self.prepareData("MAC95")
        atype = self.h.getSurvName('c condition')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Cd9+ macrophage transfer':1,
                'Ly6c+ macrophage transfer':1,
                'PBS transfer':0,
                'High fat diet':2,
                'IL4':2,
                'LPS':1,
                'Veh':0}
        if (tn == 2):
            ahash = {'PBS transfer':0,
                    'IL4':2,
                    'LPS':1,
                    'Veh':0}
        self.initData(atype, atypes, ahash)

    def getFreemerman2019(self, tn=1):
        self.prepareData("MAC96")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("\..*", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'GLUT1 WT M1':1,
                'GLUT1 KO M2':2,
                'GLUT1 WT M0':0,
                'GLUT1 WT M2':2,
                'GLUT1 KO M1':1,
                'GLUT1 KO M0':0}
        if (tn == 2):
            ahash = {'GLUT1 WT M1':1,
                    'GLUT1 WT M0':0,
                    'GLUT1 WT M2':2}
            if (tn == 3):
                ahash = {'GLUT1 KO M2':2,
                        'GLUT1 KO M1':1,
                        'GLUT1 KO M0':0}
        self.initData(atype, atypes, ahash)

    def getEl2010(self, tn=1):
        self.prepareData("MAC97")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(" [A-D]$", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Irf4+/- IL4 4h':2,
                'Irf4-/- IL4 18h':2,
                'Irf4-/- Mock':0,
                'Irf4+/- IL4 18h':2,
                'Irf4+/- Mock':0,
                'Irf4-/- IL4 4h':2}
        if (tn == 2):
            ahash = {'Irf4+/- IL4 4h':2,
                    'Irf4+/- IL4 18h':2,
                    'Irf4+/- Mock':0}
            if (tn == 3):
                ahash = {'Irf4-/- IL4 18h':2,
                        'Irf4-/- Mock':0,
                        'Irf4-/- IL4 4h':2}
        self.initData(atype, atypes, ahash)

    def getLi2015(self, tn=1):
        self.prepareData("MAC98")
        atype = self.h.getSurvName('c src1')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'Mouse macrophage at M2 treated with nutlin-3a':2,
                'Mouse macrophage at M2':2,
                'Mouse macrophage at M2 treated with 10058F4':2,
                'Mouse macrophage at M1':1,
                'Mouse macrophage at M0':0}
        self.initData(atype, atypes, ahash)

    def getRamsey2014(self, tn=1):
        self.prepareData("MAC99")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("\..$", "", str(k)) for k in atype]
        atypes = ['Rp', 'Rs', 'Lp']
        ahash = {'LdlrKO.male.polyic':2,
                'Reversa.female.polyic':0,
                'Reversa.female.saline':1,
                'Reversa.male.saline':1,
                'Reversa.male.polyic':0}
        if (tn == 2):
            ahash = {'LdlrKO.male.polyic':2,
                    'Reversa.male.saline':1,
                    'Reversa.male.polyic':0}
        self.initData(atype, atypes, ahash)

    def getKuo2011(self, tn=1):
        self.prepareData("MAC100")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("_m[0-9]*_", "_", str(k)) for k in atype]
        atypes = ['ML', 'MML', 'aML', 'aL']
        ahash = {'macrophage_C57BL/6-Ldlr-/-':0,
                'macrophage_C57BL/6.MOLFc4(51Mb)-Ldlr-/-':1,
                'aorta_C57BL/6.MOLFc4(51Mb)-Ldlr-/-':2,
                'aorta_C57BL/6-Ldlr-/-':3}
        self.initData(atype, atypes, ahash)

    def getPrice2017(self, tn=1):
        self.prepareData("MAC101")
        atype = self.h.getSurvName('c genotype')
        atypes = ['DKO', 'LDLR', 'miR-33', 'WT']
        ahash = {'LDLR-/-/miR-33-/-':0, 'LDLR-/-':1, 'Wildtype':3,
                'miR-33-/-':2}
        self.initData(atype, atypes, ahash)

    def getNicolaou2017(self, tn=1):
        self.prepareData("MAC102")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("_rep.*", "", str(k)) for k in atype]
        atype = [re.sub("mouse_", "", str(k)) for k in atype]
        atype = [re.sub("primary macrophages", "mac", str(k)) for k in atype]
        atype = [re.sub("_B6.Ldlr....Adam17", "_", str(k)) for k in atype]
        atypes = ['mwm', 'awm', 'mem', 'aem', 'mwf', 'awf', 'mef', 'aef']
        ahash = {'mac_wt/wt_male':0,
                'aorta_wt/wt_male':1,
                'mac_wt/wt_female':4,
                'aorta_wt/wt_female':5,
                'mac_ex/ex_female':6,
                'aorta_ex/ex_male':3,
                'mac_ex/ex_male':2,
                'aorta_ex/ex_female':7}
        self.initData(atype, atypes, ahash)

    def getPG2019lpsRep(self, tn=1):
        #self.prepareData("MACPH1", cfile="/Users/mahdi/public_html/Hegemon/explore.conf")
        self.prepareData("MAC125")
        ttype = self.h.getSurvName("c type")
        mtype = self.h.getSurvName("c times")
        atype = [ str(ttype[i]) + " " + str(mtype[i]) for i in
                range(len(ttype))]
        atypes = ['KO 0', 'WT 0', 'WT 6hr', 'KO 6hr']
        ahash = {}
        if (tn == 2):
            atypes = ['WT 6hr', 'KO 6hr']
        self.initData(atype, atypes, ahash)

    def getPG2019lpsII(self, tn=1):
        self.prepareData("MAC125.2")
        atype = self.h.getSurvName("c Group")
        atypes = ['KO-0', 'WT-0', 'WT-6h', 'KO-6h']
        ahash = {}
        if (tn == 2):
            atypes = ['WT-6h', 'KO-6h']
        self.initData(atype, atypes, ahash)

    def getZhang2014(self, tn=1):
        self.prepareData("GL19")
        atype = self.h.getSurvName('c Collection time')
        atypes = ['CT22', 'CT28', 'CT34', 'CT40', 'CT46', 'CT52', 'CT58',
                'CT64']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMure2018(self, tn=1):
        self.prepareData("GL21.2")
        tissue = self.h.getSurvName('c tissue')
        time = self.h.getSurvName('c time')
        atype = time
        atypes = ['ZT00', 'ZT02', 'ZT04', 'ZT06', 'ZT08', 'ZT10',
                'ZT12', 'ZT14', 'ZT16', 'ZT18', 'ZT20', 'ZT22']
        ahash = {}
        if tn == 2:
            t1 = "Descending colon"
            atype = [ ",".join([str(k[i]) for k in [tissue, time]]) 
                    for i in range(len(atype))]
            for k in range(len(atypes)):
                ahash[",".join([t1, atypes[k]])] = k
        if tn == 3:
            t1 = "Ascending colon"
            t1 = "Retina"
            atype = [ ",".join([str(k[i]) for k in [tissue, time]]) 
                    for i in range(len(atype))]
            for k in range(len(atypes)):
                ahash[",".join([t1, atypes[k]])] = k
        self.initData(atype, atypes, ahash)

    def getWu2018(self, tn=1):
        self.prepareData("GL22")
        atype = self.h.getSurvName('c collection time')
        atypes = ['0', '6', '12', '18']
        ahash = {'0':0, '1200':2, '600':1, '1800':3}
        self.initData(atype, atypes, ahash)

    def getBraun2018(self, tn=1):
        self.prepareData("GL25")
        atype = self.h.getSurvName('c hour')
        atypes = ['1', '3', '5', '7', '9', '11', '13', '15', '17',
                '19', '21', '23', '25', '27', '29']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getKervezee2019(self, tn=1):
        self.prepareData("GL26")
        atype = self.h.getSurvName('c relclocktime')
        atypes = [ '0', '4-8', '12-16', '17-18', '19-22', '24-26',
                '30-34', '36-38', '39-42']
        ahash = {'0.2':0, '0.3':0, '0.4':0, '0.5':0, '0.8':0, 
                '4.2':1, 
                '4.3':1, '6.2':1, '8.2':1, '8.3':1, '8.4':1, '8.5':1, '8.8':1,
                '12.2':2, '12.3':2, '13.2':2, '14.4':2, '16.2':2, '16.3':2,
                '17.2':3,
                '18.2':3, '18.3':3, '18.4':3, '18.9':3, '19':4, '20.2':4, '20.3':4,
                '21.8':4, '22.2':4, '22.3':4, '24.2':5, '24.3':5, '24.4':5,
                '26.2':5, '26.3':5, '26.5':5,
                '30.2':6, '30.3':6, '32.1':6, '34.2':6, '34.3':6, '34.4':6,
                '36.3':7, '38.2':7, '38.3':7, '38.8':7, '39.2':8, 
                '42.1':8, '42.2':8, '42.3':8}
        self.initData(atype, atypes, ahash)

    def getMargerie2019(self, tn=1):
        self.prepareData("MAC103")
        atype = self.h.getSurvName('c statin')
        atypes = [ 'NS', 'R', 'A', 'S', 'P', 'F']
        ahash = {'No Statin':0, 'Rosuvastatin':1, 'Atorvastatin':2, 'Simvastatin':3,
                'Pravastatin':4, 'Fluvastatin':5}
        if (tn == 2):
            atypes = ['NS', 'R']
            ahash = {'No Statin':0, 'Rosuvastatin':1}
        self.initData(atype, atypes, ahash)

    def getScicluna2015(self, tn=1):
        self.prepareData("MAC104")
        atype = self.h.getSurvName('c diabetes_mellitus')
        atypes = ['No_DM', 'NA', 'DM']
        ahash = {}
        if (tn == 2):
            atypes = ['No_DM', 'DM']
        self.initData(atype, atypes, ahash)

    def getKallionpaa2014I(self, tn=1):
        self.prepareData("MAC105.1")
        atype = self.h.getSurvName('c time from t1d diagnosis')
        atypes = ['N', 'E', 'L']
        ahash = {'no T1D diagnosis':0}
        for i in self.h.aRange():
            if atype[i] != 'no T1D diagnosis':
                v = float(atype[i])
                if (v >= -24 and v <= 0):
                    ahash[atype[i]] = 1
                else:
                    ahash[atype[i]] = 2
        if (tn == 2):
            atypes = ['N', 'T1D']
            ahash = {'no T1D diagnosis':0}
            for i in self.h.aRange():
                if atype[i] != 'no T1D diagnosis':
                    ahash[atype[i]] = 1
        self.initData(atype, atypes, ahash)

    def getKallionpaa2014II(self, tn=1):
        self.prepareData("MAC105.2")
        atype = self.h.getSurvName('c time from t1d diagnosis')
        atypes = ['N', 'E', 'L']
        ahash = {'no diagnosis':0}
        for i in self.h.aRange():
            if atype[i] != 'no diagnosis':
                v = float(atype[i])
                if (v >= -24 and v <= 0):
                    ahash[atype[i]] = 1
                else:
                    ahash[atype[i]] = 2
        if (tn == 2):
            atypes = ['N', 'T1D']
            ahash = {'no diagnosis':0}
            for i in self.h.aRange():
                if atype[i] != 'no diagnosis':
                    ahash[atype[i]] = 1
        self.initData(atype, atypes, ahash)

    def getKallionpaa2014III(self, tn=1):
        self.prepareData("MAC105.3")
        atype = self.h.getSurvName('c time from t1d diagnosis')
        atypes = ['N', 'E', 'L']
        ahash = {'no T1D diagnosis':0}
        for i in self.h.aRange():
            if atype[i] != 'no T1D diagnosis':
                v = float(atype[i])
                if (v >= -24 and v <= 0):
                    ahash[atype[i]] = 1
                else:
                    ahash[atype[i]] = 2
        if (tn == 2):
            atypes = ['N', 'T1D']
            ahash = {'no T1D diagnosis':0}
            for i in self.h.aRange():
                if atype[i] != 'no T1D diagnosis':
                    ahash[atype[i]] = 1
        self.initData(atype, atypes, ahash)

    def getRam2016(self, tn=1):
        self.prepareData("MAC106")
        btype = self.h.getSurvName('c type1_diabetes')
        atype = self.h.getSurvName('c condition')
        atypes = ['PMA 6h', 'Basal', 'CD8+', 'CD4+']
        ahash = {}
        if (tn == 2):
            atypes = ['1', '2']
            atype = btype
        self.initData(atype, atypes, ahash)

    def getAlmon2009(self, tn=1):
        self.prepareData("MAC107")
        atype = self.h.getSurvName("c desc")
        ttype = [str(k).split(" ")[5] if len(str(k).split(" ")) > 5 else None
                for k in atype]
        atype = [re.sub(".*feeded with ", "", str(k)) for k in atype]
        mtype = [re.sub("and .*", "", str(k)) for k in atype]
        atype = [str(ttype[i]) + " " + mtype[i] for i in range(len(atype))]
        atypes = ['aN', 'aH', 'lN', 'lH', 'mN', 'mH']
        ahash = {'adipose ND ':0,
                'adipose HFD ':1,
                'liver ND ':2,
                'liver HFD ':3,
                'muscle ND ':4,
                'muscle HFD ':5,
                'livadipose ND ':0}
        self.initData(atype, atypes, ahash)

    def getChen2014(self, tn=1):
        self.prepareData("MAC109")
        atype = self.h.getSurvName("c stimulated with")
        atype = [re.sub("auto.*\(", "(", str(k)) for k in atype]
        atype = [re.sub(" HLA risk sibling", "", str(k)) for k in atype]
        atype = [re.sub(" plasma", "", str(k)) for k in atype]
        atype = [re.sub(" series", "", str(k)) for k in atype]
        atype = [re.sub("Longitudinal ", "", str(k)) for k in atype]
        atypes = ['nl', 'nh', 'phNp', 'o', 'nhNp', 'nlNp', 'P']
        ahash = {'(AA-) low':0,
                '(AA-) high':1,
                '(AA+) high non-progressor':2,
                'recent onset cultured with IL1RA':3,
                '(AA-) high non-progressor':4,
                '(AA-) low non-progressor':5,
                'progressor':6}
        self.initData(atype, atypes, ahash)

    def getLefebvre2017(self, tn=1):
        self.prepareData("LIV3")
        atype = self.h.getSurvName("c Title")
        atype = [str(k).replace("liver biopsy ", "") for k in atype]
        self.patient = [str(k).split(" ")[0] for k in atype]
        atype = [str(k).split(" ")[1] if len(str(k).split(" ")) > 1 else "" for
                k in atype]
        self.paired = [re.sub("\((.*)\)", "\\1", str(k)) for k in atype]
        self.time = self.h.getSurvName("c time")
        self.treatment = self.h.getSurvName("c type of intervention")
        atype = self.h.getSurvName("c src1")
        atypes = ['b', 'f', 'Nb', 'Nf', 'ub', 'uf']
        ahash = {'no NASH liver baseline':0,
                'no NASH liver follow-up':1,
                'NASH liver follow-up':3,
                'NASH liver baseline':2,
                'undefined liver baseline':4,
                'undefined liver follow-up':5}
        self.aval = [ahash[i] if i in ahash else None for i in atype]
        phash = {}
        for i in self.h.aRange():
            phash[self.patient[i]] = i
        self.rtype = [None, None] + [str(self.aval[phash[self.paired[i]]])+" "+\
                str(self.aval[i])+" " + str(self.treatment[i]) \
                if self.paired[i] in phash and self.paired[i] != "" \
                else str(self.aval[i])+" " + str(self.treatment[i]) \
                for i in self.h.aRange()]
        fhash = {}
        for i in self.h.aRange():
            if self.paired[i] in phash and self.paired[i] != "":
                fhash[self.paired[i]] = i
        self.ftype = [None, None] + [str(self.aval[i])+" "+str(self.treatment[i])\
                + " " + str(self.aval[fhash[self.patient[i]]])\
                if self.patient[i] in fhash\
                else str(self.aval[i])+" " + str(self.treatment[i]) \
                for i in self.h.aRange()]
        if (tn == 2):
            atypes = ['b', 'Nb', 'ub']
            ahash = {'no NASH liver baseline':0,
                    'NASH liver baseline':1,
                    'undefined liver baseline':2}
        if (tn == 3):
            atypes = ['b', 'Nb']
            ahash = {'no NASH liver baseline':0,
                    'NASH liver baseline':1}
        if (tn == 4):
            atypes = ['f', 'Nf']
            ahash = {'no NASH liver follow-up':0,
                    'NASH liver follow-up':1}
        if (tn == 5):
            atypes = ['R', 'NR']
            atype = self.ftype
            ahash = {'2 Diet 1': 0, '2 Diet 3': 1}
        self.initData(atype, atypes, ahash)

    def getGallagher2010(self, tn=1):
        self.prepareData("MAC110")
        atype = self.h.getSurvName("c Type")
        atypes = ['N', 'GI', 'D']
        ahash = {'diabetic':2, 'glucoseIntolerant':1, 'normal':0}
        self.initData(atype, atypes, ahash)

    def getDuPlessis2015(self, tn=1):
        self.prepareData("MAC111")
        atype = self.h.getSurvName("c src1")
        atypes = ['S1', 'V1', 'S2', 'V2', 'S3', 'V3', 'S4', 'V4']
        ahash = {'Subc Fat, Histology class 1':0,
                'Visceral Fat, Histology class 1':1,
                'Subc Fat, Histology class 2':2,
                'Visceral Fat, Histology class 2':3,
                'Subc Fat, Histology class 3':4,
                'Visceral Fat, Histology class 3':5,
                'Subc Fat, Histology class 4':6,
                'Visceral Fat, Histology class 4':7}
        if (tn == 2):
            atypes = ['N', 'F']
            ahash = {'Subc Fat, Histology class 1':0,
                    'Visceral Fat, Histology class 1':0,
                    'Subc Fat, Histology class 2':0,
                    'Visceral Fat, Histology class 2':0,
                    'Subc Fat, Histology class 3':0,
                    'Visceral Fat, Histology class 3':0,
                    'Subc Fat, Histology class 4':1,
                    'Visceral Fat, Histology class 4':1}
        self.initData(atype, atypes, ahash)

    def getWu2007IS(self, tn=1):
        self.prepareData("MAC112")
        self.prepareData("MAC112")
        atype = self.h.getSurvName("c status")
        atypes = ['IS', 'IR', 'D']
        ahash = {'insulin sensitive':0, 'diabetic':2, 'insulin resistant':1}
        if (tn == 2):
            atypes = ['IS', 'IR']
            ahash = {'insulin sensitive':0, 'insulin resistant':1}
        if (tn == 3):
            atypes = ['IS', 'D']
            ahash = {'insulin sensitive':0, 'diabetic':1}
        self.initData(atype, atypes, ahash)

    def getStenvers2019(self, tn=1):
        self.prepareData("MAC113")
        atype = self.h.getSurvName("c subject status")
        atypes = ['H', 'T2D']
        ahash = {'Type 2 diabetes':1, 'Healthy':0}
        if (tn >= 2):
            stype = self.h.getSurvName("c subject status")
            ttype = self.h.getSurvName("c timepoint")
            atype = [ str(stype[i]) + " " + str(ttype[i]) for i in
                    range(len(stype))]
            atypes = ['H1', 'H2', 'H3', 'H4', 'D1', 'D2', 'D3', 'D4']
            ahash = {'Type 2 diabetes D2_ZT_15:30':4,
                    'Type 2 diabetes D3_ZT_0:15':5,
                    'Type 2 diabetes D3_ZT_5:45':6,
                    'Type 2 diabetes D3_ZT_11:15':7,
                    'Healthy D2_ZT_15:30':0,
                    'Healthy D3_ZT_0:15':1,
                    'Healthy D3_ZT_5:45':2,
                    'Healthy D3_ZT_11:15':3}
        if (tn == 3):
            atypes = ['H1', 'H2', 'H3', 'H4']
            ahash = {'Healthy D2_ZT_15:30':0,
                    'Healthy D3_ZT_0:15':1,
                    'Healthy D3_ZT_5:45':2,
                    'Healthy D3_ZT_11:15':3}
        if (tn == 4):
            atypes = ['D1', 'D2', 'D3', 'D4']
            ahash = {'Type 2 diabetes D2_ZT_15:30':0,
                    'Type 2 diabetes D3_ZT_0:15':1,
                    'Type 2 diabetes D3_ZT_5:45':2,
                    'Type 2 diabetes D3_ZT_11:15':3}
        if (tn == 5):
            atypes = ['H1', 'H2', 'D1', 'D2']
            ahash = {'Type 2 diabetes D2_ZT_15:30':2,
                    'Type 2 diabetes D3_ZT_0:15':3,
                    'Healthy D2_ZT_15:30':0,
                    'Healthy D3_ZT_0:15':1}
        self.initData(atype, atypes, ahash)

    def getDAmore2018(self, tn=1):
        self.prepareData("MAC114")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'MetS']
        ahash = {'control':0, 'Metabolic Syndrome':1}
        self.initData(atype, atypes, ahash)

    def getArnardottir2014(self, tn=1):
        self.prepareData("GL27")
        rtype = self.h.getSurvName("c responder")
        htype = self.h.getSurvName("c hour")
        btype = self.h.getSurvName("c biological group")
        atype = [ str(btype[i]) + " " + str(rtype[i]) + " " + str(htype[i])
                         for i in range(len(rtype))]
        atypes = ['bl0', 'bl4', 'bl8', 'bl12', 'bl16', 'bl20',
                'bh0', 'bh4', 'bh8', 'bh12', 'bh16', 'bh20',
                'sl0', 'sl4', 'sl8', 'sl12', 'sl16', 'sl20',
                'rl0', 'rl4', 'rl8',
                'sh0', 'sh4', 'sh8', 'sh12', 'sh16', 'sh20',
                'rh0', 'rh4', 'rh8']
        ahash = {
                'baseline low 0':0,
                'baseline low 4':1,
                'baseline low 8':2,
                'baseline low 12':3,
                'baseline low 16':4,
                'baseline low 20':5,
                'baseline high 0':6,
                'baseline high 4':7,
                'baseline high 8':8,
                'baseline high 12':9,
                'baseline high 16':10,
                'baseline high 20':11,
                'sleep deprivation low 0':12,
                'sleep deprivation low 4':13,
                'sleep deprivation low 8':14,
                'sleep deprivation low 12':15,
                'sleep deprivation low 16':16,
                'sleep deprivation low 20':17,
                'recovery low 0':18,
                'recovery low 4':19,
                'recovery low 8':20,
                'sleep deprivation high 0':21,
                'sleep deprivation high 4':22,
                'sleep deprivation high 8':23,
                'sleep deprivation high 12':24,
                'sleep deprivation high 16':25,
                'sleep deprivation high 20':26,
                'recovery high 0':27,
                'recovery high 4':28,
                'recovery high 8':29}
        if (tn == 2):
            atype = [ str(btype[i]) + " " + str(rtype[i])
                    for i in range(len(rtype))]
            atypes = ['bl', 'bh', 'sl', 'rl', 'sh', 'rh']
            ahash = {'baseline low':0,
                    'baseline high':1,
                    'sleep deprivation low':2,
                    'recovery low':3,
                    'sleep deprivation high':4,
                    'recovery high':5}
        if (tn == 3):
            atype = rtype
            atypes = ['high', 'low']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getADPooledDyn(self, tn=1):
        self.prepareData("AD7")
        atype = self.h.getSurvName('c AD specific');
        ahash = {'0':0, '1':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Disease State');
        atypes = ['N', 'AD']
        ahash = {'Normal':0, "Alzheimer's Disease":1, 'normal':0,
                'definite AD':0, 'Control':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLiang2007(self):
        self.prepareData("AD2")
        atype = self.h.getSurvName('c Disease State');
        atypes = ['N', 'AD']
        ahash = {'normal\xa0':0, "Alzheimer's Disease\xa0":1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getFriedman2017(self):
        self.prepareData("AD3")
        atype = self.h.getSurvName('c diagnosis');
        atypes = ['N', 'AD']
        ahash = {'control':0, "Alzheimer's disease":1}
        self.initData(atype, atypes, ahash)

    def getBerchtold2018(self, tn = 1):
        self.prepareData("AZ3", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c physical activity tier');
        atypes = ['H', 'M', 'L']
        ahash = {'high activity':0,
                'low activity':2,
                'moderate activity':1}
        if tn == 2:
            atypes = ['H', 'L']
            ahash = {'high activity':0,
                    'low activity':1,
                    'moderate activity':0}
        self.initData(atype, atypes, ahash)

    def getSimpson2015(self, tn = 1):
        self.prepareData("AZ5", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c neuronal ddr');
        atypes = ['H', 'L']
        ahash = {'High DNA damage':0, 'Low DNA damage':1}
        self.initData(atype, atypes, ahash)

    def getPiras2019(self, tn = 1):
        self.prepareData("AZ1", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        age = self.h.getSurvName('c expired_age');
        atype = self.h.getSurvName('c diagnosis');
        atypes = ['N', 'AD']
        ahash = {'ND':0, 'AD':1}
        if tn == 2:
            atype = [atype[i] if int(age[i]) > 90
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPatel2019(self, tn = 1):
        self.prepareData("AD8")
        atype = self.h.getSurvName('c tissue');
        ahash = {'Temporal_Cortex':3,
                'Cerebellum':4,
                'Frontal_Cortex':5,
                'Entorhinal_Cortex':6}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease state');
        atypes = ['N', 'A', 'AD']
        ahash = {'AsymAD':1, 'AD':2, 'control':0}
        if (tn >= 2):
            atypes = ['N', 'AD']
            ahash = {'AD':1, 'control':0}
        if (tn > 2):
            atype = [atype[i] if rval[i] == tn else None for i in range(len(atype))]
        self.rval = rval
        self.initData(atype, atypes, ahash)

    def getNarayanan2014(self, tn = 1):
        self.prepareData("AZ8", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease status');
        atypes = ['N', 'AD', 'HD']
        ahash = {'non-demented':0,
                "Alzheimer's disease":1,
                "Huntington's disease":2}
        if tn == 2:
            atypes = ['N', 'AD']
            ahash = {'non-demented':0,
                    "Alzheimer's disease":1}
        self.initData(atype, atypes, ahash)

    def getZhang2013m(self, dbid = 'AZ12', tn = 1):
        self.db = hu.Database("/Users/mgosztyl/public_html/Hegemon/explore.conf")
        self.dbid = dbid
        atype = self.h.getSurvName('c disease');
        atypes = ['N', 'A']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBerchtold2014(self, tn = 1):
        self.prepareData("AZ11", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c src1")
        atype = [str(i) for i in atype]
        res = []
        for k in atype:
            l1 = k.split(",")
            if (len(l1) != 4):
                res.extend([k])
            else:
                res.extend([l1[1].strip() + "_" + l1[2].strip()])
        atype = res
        atypes = ['N', 'AD']
        ahash = {'entorhinal cortex_male':0,
                'entorhinal cortex_male_AD':1,
                'entorhinal cortex_female':0,
                'entorhinal cortex_female_AD':1,
                'superior frontal gyrus_male':0,
                'superior frontal gyrus_male_AD':1,
                'superior frontal gyrus_female':0,
                'superior frontal gyrus_female_AD':1,
                'postcentral gyrus_male':0,
                'post-central gyrus_male_AD':1,
                'postcentral gyrus_female':0,
                'post-central gyrus_female_AD':1,
                'hippocampus_male':0,
                'hippocampus_male_AD':1,
                'hippocampus_female':0,
                'hippocampus_female_AD':1}
        if (tn == 2):
            ahash = {'entorhinal cortex_male':0,
                    'entorhinal cortex_male_AD':1,
                    'superior frontal gyrus_male':0,
                    'superior frontal gyrus_male_AD':1,
                    'postcentral gyrus_male':0,
                    'post-central gyrus_male_AD':1,
                    'hippocampus_male':0,
                    'hippocampus_male_AD':1}
        if (tn == 3):
            ahash = {'entorhinal cortex_female':0,
                    'entorhinal cortex_female_AD':1,
                    'superior frontal gyrus_female':0,
                    'superior frontal gyrus_female_AD':1,
                    'postcentral gyrus_female':0,
                    'post-central gyrus_female_AD':1,
                    'hippocampus_female':0,
                    'hippocampus_female_AD':1}
        if (tn == 4):
            ahash = {'entorhinal cortex_male':0,
                    'entorhinal cortex_male_AD':1,
                    'entorhinal cortex_female':0,
                    'entorhinal cortex_female_AD':1}
        if (tn == 5):
            ahash = {'superior frontal gyrus_male':0,
                    'superior frontal gyrus_male_AD':1,
                    'superior frontal gyrus_female':0,
                    'superior frontal gyrus_female_AD':1}
        if (tn == 6):
            ahash = {'postcentral gyrus_male':0,
                    'post-central gyrus_male_AD':1,
                    'postcentral gyrus_female':0,
                    'post-central gyrus_female_AD':1}
        if (tn == 7):
            ahash = {'hippocampus_male':0,
                    'hippocampus_male_AD':1,
                    'hippocampus_female':0,
                    'hippocampus_female_AD':1}
        self.initData(atype, atypes, ahash)

    def getWang2016(self, dbid="AD9", tn = 1):
        self.dbid = dbid
        atype = self.h.getSurvName('c brain region')
        ahash = {'Inferior Temporal Gyrus':3,
                'Parahippocampal Gyrus':4,
                'Middle Temporal Gyrus':5,
                'Occipital Visual Cortex':6,
                'Prefrontal Cortex':7,
                'Hippocampus':8,
                'Caudate Nucleus':9,
                'Frontal Pole':10,
                'Precentral Gyrus':11,
                'Posterior Cingulate Cortex':12,
                'Superior Temporal Gyrus':13,
                'Superior Parietal Lobule':14,
                'Temporal Pole':15,
                'Anterior Cingulate':16,
                'Inferior Frontal Gyrus':17,
                'Dorsolateral Prefrontal Cortex':18,
                'Putamen':19}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c neuropathological category')
        atypes = ['N', 'AD']
        ahash = {'definite AD':1, 'Possible AD':1,
                'Normal':0, 'Probable AD':1}
        if (tn >= 2):
            atypes = ['N', 'AD']
            ahash = {'definite AD':1, 'Normal':0}
        if (tn >= 3):
            atype = [atype[i] if rval[i] == tn else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBerchtold2014RMA(self, tn=1):
        self.prepareData("AD11")
        atype = self.h.getSurvName('c AD specific');
        ahash = {'0':0, '1':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c src1")
        atype = [str(i) for i in atype]
        res = []
        for k in atype:
            l1 = k.split(",")
            if (len(l1) != 4):
                res.extend([k])
            else:
                res.extend([l1[1].strip() + "_" + l1[2].strip()])
        atype = res
        atypes = ['N', 'AD']
        ahash = {'entorhinal cortex_male':0,
                'entorhinal cortex_male_AD':1,
                'entorhinal cortex_female':0,
                'entorhinal cortex_female_AD':1,
                'superior frontal gyrus_male':0,
                'superior frontal gyrus_male_AD':1,
                'superior frontal gyrus_female':0,
                'superior frontal gyrus_female_AD':1,
                'postcentral gyrus_male':0,
                'post-central gyrus_male_AD':1,
                'postcentral gyrus_female':0,
                'post-central gyrus_female_AD':1,
                'hippocampus_male':0,
                'hippocampus_male_AD':1,
                'hippocampus_female':0,
                'hippocampus_female_AD':1}
        if (tn == 4):
            ahash = {'entorhinal cortex_male':0,
                    'entorhinal cortex_male_AD':1,
                    'superior frontal gyrus_male':0,
                    'superior frontal gyrus_male_AD':1,
                    'postcentral gyrus_male':0,
                    'post-central gyrus_male_AD':1,
                    'hippocampus_male':0,
                    'hippocampus_male_AD':1}
        if (tn == 5):
            ahash = {'entorhinal cortex_female':0,
                    'entorhinal cortex_female_AD':1,
                    'superior frontal gyrus_female':0,
                    'superior frontal gyrus_female_AD':1,
                    'postcentral gyrus_female':0,
                    'post-central gyrus_female_AD':1,
                    'hippocampus_female':0,
                    'hippocampus_female_AD':1}
        if (tn == 6):
            ahash = {'entorhinal cortex_male':0,
                    'entorhinal cortex_male_AD':1,
                    'entorhinal cortex_female':0,
                    'entorhinal cortex_female_AD':1}
        if (tn == 7):
            ahash = {'superior frontal gyrus_male':0,
                    'superior frontal gyrus_male_AD':1,
                    'superior frontal gyrus_female':0,
                    'superior frontal gyrus_female_AD':1}
        if (tn == 8):
            ahash = {'postcentral gyrus_male':0,
                    'post-central gyrus_male_AD':1,
                    'postcentral gyrus_female':0,
                    'post-central gyrus_female_AD':1}
        if (tn == 9):
            ahash = {'hippocampus_male':0,
                    'hippocampus_male_AD':1,
                    'hippocampus_female':0,
                    'hippocampus_female_AD':1}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getGelman2012(self, tn = 1):
        self.prepareData("DE39", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c phenotype');
        atypes = ['N', 'HAND']
        ahash = {'HIV Infected':0,
                'HIV Infected with neurocognitive impairment (HAD: HIV-associated dementia)':1,
                'HIV Infected with HAD and HIV encephalitis (HIVE)':1,
                'normal (control)':0,
                'HIV Infected with HAD':1,
                'HIV Infected with HAD and encephalitis':1}
        self.initData(atype, atypes, ahash)
        
    def getChenPlotkin2008(self, tn = 1):
        self.prepareData("DE1", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c src1")
        atype = [str(k).split(" ")[1] if len(str(k).split(" ")) > 1 else None
                for k in atype]
        atype = [str(k).split("-")[0] for k in atype]
        atypes = ['N', 'P', 'S']
        ahash = {'Normal':0,
                'Progranulin':1,
                'Sporadic':2}
        if tn == 2:
            atypes = ['N', 'FTD']
            ahash = {'Normal':0,
                    'Progranulin':1}
        if tn == 3:
            atype = self.h.getSurvName('c disease and tissue')
            atypes = ['N', 'FTD']
            ahash = {'Normal hippocampus':0,
                'Progranulin hippocampus':1,
                'Sporadic hippocampus':1}
        self.initData(atype, atypes, ahash)
        
    def getOlmosSerrano2016(self, tn = 1):
        self.prepareData("DE49", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease status');
        atypes = ['N', 'DS']
        ahash = {'CTL':0,
                'DS':1}
        if tn == 2:
            atype = self.h.getSurvName('c disease and tissue');
            atypes = ['N', 'DS']
            ahash = {'CTL ITC':0,
                'CTL STC':0,
                'CTL HIP':0,
                'DS ITC':1, 'DS STC':1, 'DS HIP':1}
        self.initData(atype, atypes, ahash)
        
    def getWilliams2009(self, tn = 1):
        self.prepareData("DE51", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Disease State');
        atypes = ['N', 'MCI']
        ahash = {'Control':0, 'MCI':1}
        self.initData(atype, atypes, ahash)
        
    def getBartolettiStella2019 (self, tn = 1):
        self.prepareData("DE52", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c condition');
        atypes = ['N', 'CJD']
        ahash = {'Control':0, 'sCJD affected':1}
        self.initData(atype, atypes, ahash)
        
    def getWes2014a (self, tn = 1):
        self.prepareData("AZ78", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype');
        atypes = ['N', 'rTg4510']
        ahash = {'Dbl Neg':0, 'Tg4510':1}
        self.initData(atype, atypes, ahash)

    def getWes2014b (self, tn = 1):
        self.prepareData("AZ77", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype');
        atypes = ['N', 'rTg4510']
        ahash = {'Dbl Neg':0, 'Tg4510':1}
        self.initData(atype, atypes, ahash)
        
    def getWes2014c (self, tn = 1):
        self.prepareData("AZ78", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype');
        atypes = ['N', 'rTg4510']
        ahash = {'Dbl Neg':0, 'Tg4510':1}
        self.initData(atype, atypes, ahash)
        
    def getHokama2013 (self, tn = 1):
        self.prepareData("AZ49", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype');
        atypes = ['N', '3xTg']
        ahash = {'non-Tg':0, '3xTg-Homo':1}
        self.initData(atype, atypes, ahash)
        
    def getWakutani2014 (self, tn = 1):
        self.prepareData("AZ60", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genetic background');
        atypes = ['N', 'TgCRND8']
        ahash = {'non-transgenic littermate mouse':0,
                'TgCRND8 transgenic mouse':1}
        self.initData(atype, atypes, ahash)
        
    def getMeyer2019 (self, dbid = "AZ29", tn = 1):
        self.db = hu.Database("/Users/mgosztyl/public_html/Hegemon/explore.conf")
        self.dbid = dbid
        atype = self.h.getSurvName('c diagnosis');
        atypes = ['N', 'AD']
        ahash = {'normal':0,
                'sporadic Alzheimer\'s disease':1}
        self.initData(atype, atypes, ahash)
        
    def getScheckel2019 (self, tn = 1):
        self.prepareData("AZ95", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)');
        atypes = ['N', 'AD']
        ahash = {'wiltype iPSC-derived neurons':0,
                'APP/PSEN1 double mutant iPSC-derived neurons':1}
        if tn == 2:
            ahash = {'wiltype iPSC-derived neurons':0,
                'APP mutant iPSC-derived neurons':1}
        if tn == 3:
            ahash = {'wiltype iPSC-derived neurons':0,
                'PSEN1 mutant iPSC-derived neurons':1}
        self.initData(atype, atypes, ahash)

    def getTsalik2015(self, tn=1):
        self.prepareData("ms32.0", "/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c sirs outcomes (ch1)")
        atypes = ['SHK', 'SS', 'SIRS', 'US', 'SD']
        ahash = {'Septic shock':0,
                'severe sepsis':1,
                'SIRS':2,
                'Uncomplicated sepsis':3,
                'sepsis death':4}
        if (tn == 2):
            atype = self.h.getSurvName("c sirs vs sepsis (ch1)")
            atypes = ['Sepsis', 'SIRS']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getBarcella2018I(self, tn=1):
        self.prepareData("ms33.1", "/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c clinical classification (ch1)")
        atypes = ['R', 'NR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBarcella2018II(self, tn=1):
        self.prepareData("ms33.2", "/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c clinical classification (ch1)")
        atypes = ['R', 'NR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBarcella2018(self, tn=1):
        self.prepareData("MAC115.1")
        ctype = self.h.getSurvName("c clinical classification")
        ttype = self.h.getSurvName("c timepoint")
        atype = [ str(ctype[i]) + " " + str(ttype[i]) for i in
                                range(len(ctype))]
        atypes = ['R T1', 'R T2', 'NR T1', 'NR T2']
        ahash = {}
        if (tn == 2):
            atypes = ['R', 'NR']
            ahash = {'R T2':0, 'NR T1':1}
        self.initData(atype, atypes, ahash)

    def getDolinay2012(self, tn=1):
        self.prepareData("MAC116")
        atype = self.h.getSurvName("c src1")
        atypes = ['U', 'RS0', 'S0', 'S7', 'SA0', 'SA7']
        ahash = {'SIRS Day 0':1, 'Sepsis Day 7':3, 'se/ARDS Day 7':5,
                'se/ARDS Day 0':4, 'untreated':0, 'Sepsis Day 0':2}
        if (tn == 2):
            atypes = ['U', 'S']
            ahash = {'SIRS Day 0':1, 'Sepsis Day 7':1, 'se/ARDS Day 7':1,
                    'se/ARDS Day 0':1, 'untreated':0, 'Sepsis Day 0':1}
        self.initData(atype, atypes, ahash)

    def getJuss2016(self, tn=1):
        self.prepareData("MAC117")
        atype = self.h.getSurvName("c src1")
        atypes = ['HVT', 'ARDS', 'C0', 'C6', 'pan6', 'E6', 'd6', 'g6', 'p6']
        ahash = {'PMNs from HVT':0,
                'PMNs from ARDS patient':1,
                'Cultured PMNs from HVT, DMSO, t=0':2,
                'Cultured PMNs from HVT, DMSO, t=6':3,
                'Cultured PMNs from HVT + panPI3K Inhibitor, t=6h':4,
                'Cultured PMNs from HVT + GM-CSF, t=6h':5,
                'Cultured PMNs from HVT + GM-CSF + PI3Kd Inhibitor, t=6h':6,
                'Cultured PMNs from HVT + GM-CSF + PI3Kg Inhibitor, t=6h':7,
                'Cultured PMNs from HVT + GM-CSF + PanPI3K Inhibitor, t=6h':8}
        if (tn == 2):
            atypes = ['HVT', 'ARDS']
            ahash = {'PMNs from HVT':0,
                    'PMNs from ARDS patient':1}
        if (tn == 3):
            atypes = ['C', 'T']
            ahash = {'Cultured PMNs from HVT, DMSO, t=0':0,
                    'Cultured PMNs from HVT + GM-CSF, t=6h':1}
                    
        self.initData(atype, atypes, ahash)

    def getKangelaris2015(self, tn=1):
        self.prepareData("MAC118")
        atype = self.h.getSurvName("c disease state")
        atypes = ['S', 'SA']
        ahash = {'sepsis alone':0, 'sepsis with ARDS':1}
        self.initData(atype, atypes, ahash)

    def getLu2014(self, tn = 1):
        self.prepareData("AZ21", "/Users/mgosztyl/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c age")
        atype = [str(k).split(" ")[0] for k in atype]
        atypes = ['Y', 'O']
        ahash = {}
        for i in self.h.aRange():
            if float(atype[i]) > 60:
                ahash[atype[i]] = 1
            else:
                ahash[atype[i]] = 0
        self.initData(atype, atypes, ahash)

    def getMarttinen2019(self, tn=1):
        self.prepareData("AD12")
        atype = self.h.getSurvName("c braak stage")
        atypes = ['0', '1', '2', '3', '4', '5', '6']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c age")
            atypes = ['Y', 'O']
            for i in self.h.aRange():
                if float(atype[i]) > 70:
                    ahash[atype[i]] = 1
                else:
                    ahash[atype[i]] = 0
        if (tn == 3):
            atypes = ['L', 'H']
            ahash = {'0':0, '2':0, '3':1, '4':1, '5':1, '6':1}
        self.initData(atype, atypes, ahash)

    def getDonega2019(self, tn=1):
        self.prepareData("AD13.2")
        atype = self.h.getSurvName("c Type")
        ahash = {'SVZ':2, 'CD271':3, 'CD11b':4}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Disease")
        atypes = ['C', 'PD']
        ahash = {'Cntr':0, 'PD':1}
        if (tn > 1):
            atype = [atype[i] if rval[i] == tn
                    else None for i in range(len(atype))]
        self.rval = rval
        self.initData(atype, atypes, ahash)

    def getWatson2017(self, tn=1):
        self.prepareData("MAC120")
        atype = self.h.getSurvName("c sleep duration")
        atypes = ['long', 'short']
        if (tn == 2):
            atypes = ['short', 'long']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getUyhelji2018(self, tn=1):
        self.prepareData("MAC119")
        atype = self.h.getSurvName("c subject group")
        atypes = ['Sleep Deprived', 'Control']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMaret2007(self, tn=1):
        self.prepareData("MAC121")
        atype = self.h.getSurvName("c src1")
        strain = [re.split(", *", str(i))[0] for i in atype]
        expt = [re.split(", *", str(i))[1] if len(re.split(", *", str(i))) > 1
                else None for i in atype]
        time = [re.split(", *", str(i))[2] if len(re.split(", *", str(i))) > 2
                else None for i in atype]
        rep = [re.split(", *", str(i))[3] if len(re.split(", *", str(i))) > 3
                else None for i in atype]
        tissue = [re.split(", *", str(i))[4] if len(re.split(", *", str(i))) >
                4 else None for i in atype]
        atype = expt
        atypes = ['C', 'SD']
        ahash = {'sleep deprived':1,
                'control':0,
                '6 hrs sleep deprivation':1,
                '6hrs sleep deprivation':1}
        if (tn == 2):
            atype = [ ",".join([str(k[i]) for k in [strain, expt, time,
                tissue]]) for i in range(len(atype))]
            atypes = ['C', 'SD']
            ahash = {'C57BL/6J,sleep deprived,time of sacrifice ZT 0,None':1,
                    'C57BL/6J,control,time of sacrifice ZT 0,None':0}
        if (tn == 3):
            atype = [ ",".join([str(k[i]) for k in [strain, expt, time,
                tissue]]) for i in range(len(atype))]
            atypes = ['C', 'SD']
            ahash = {'AKR/J,control,time of sacrifice ZT 0,None':0,
                    'AKR/J,sleep deprived,time of sacrifice ZT 0,None':1}
        self.initData(atype, atypes, ahash)

    def getLaing2017(self, tn=1):
        self.prepareData("MAC122")
        time = self.h.getSurvName("c time sample taken")
        atype = self.h.getSurvName("c sleep protocol")
        atype = [ ",".join([str(k[i]) for k in [atype, time]])
                         for i in range(len(atype))]
        atypes = ['R1', 'R2', 'R3', 'R3', 'E1', 'E2', 'E3']
        ahash = {'Sleep Restriction,1':0,
                'Sleep Restriction,6':1,
                'Sleep Restriction,7':2,
                'Sleep Restriction,10':3,
                'Sleep Extension,1':4,
                'Sleep Extension,6':5,
                'Sleep Extension,10':6}
        if (tn == 2):
            atypes = ['R1', 'R2', 'R3', 'E1', 'E2', 'E3']
            ahash = {'Sleep Restriction,1':0,
                    'Sleep Restriction,6':1,
                    'Sleep Restriction,10':2,
                    'Sleep Extension,1':3,
                    'Sleep Extension,6':4,
                    'Sleep Extension,10':5}
        self.initData(atype, atypes, ahash)

    def getResuehr2019(self, tn=1):
        self.prepareData("MAC123")
        atype = self.h.getSurvName("c work shift")
        atypes = ['D', 'N']
        ahash = {'Day-Shift':0, 'Night-Shift':1}
        if (tn == 2):
            shift = self.h.getSurvName("c work shift")
            time = self.h.getSurvName("c time of sample")
            atype = [time[i] if shift[i] == 'Day-Shift' else None
                    for i in range(len(time))]
            atypes = ['9', '12', '15', '18', '21', '24', '27', '30']
            ahash = {}
        if (tn == 3):
            shift = self.h.getSurvName("c work shift")
            time = self.h.getSurvName("c time of sample")
            atype = [time[i] if shift[i] == 'Night-Shift' else None
                    for i in range(len(time))]
            atypes = ['9', '12', '15', '18', '21', '24', '27', '30']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getKervezee2018(self, tn=1):
        self.prepareData("GL28")
        atype = self.h.getSurvName('c condition')
        atypes = ['D', 'N']
        ahash = {'NightOrientedSchedule':1, 'DayOrientedSchedule':0}
        if (tn == 2):
            atype = self.h.getSurvName("c relclocktime")
            atypes = ['D1', 'D2', 'N1', 'N2', 'N3']
            v1 = [6, 12, 18, 24, 48]
            ahash = {}
            for i in self.h.aRange():
                t = atype[i]
                if t is None:
                    continue
                ahash[t] = np.searchsorted(v1, float(t), 'left')
        self.initData(atype, atypes, ahash)

    def getChristou2019(self, tn=1):
        self.prepareData("GL29")
        atype = self.h.getSurvName('c circadian phase')
        atypes = ['D1', 'D2', 'D3', 'N1', 'N2', 'N3']
        v1 = [-4, 0, 6, 12, 18]
        ahash = {}
        for i in self.h.aRange():
            t = atype[i]
            if t is None:
                continue
            ahash[t] = np.searchsorted(v1, float(t), 'left')
        if (tn == 2):
            atypes = ['N', 'D']
            atype = [ahash[i] if i in ahash else None for i in atype]
            ahash = { 1: 0, 2: 0, 3:1 }
        self.initData(atype, atypes, ahash)

    def getZieker2010(self, tn=1):
        self.prepareData("GL30")
        atype = self.h.getSurvName('c time point')
        atypes = ['0h', '6h', '12h', '18h']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getSteidl2010(self, tn=1):
        self.prepareData("LM7")
        atype = self.h.getSurvName('c Title')
        atype = [ re.split(",", str(i))[0] for i in atype]
        atypes = ['class S', 'class F']
        if (tn == 2):
            atype = self.h.getSurvName('c disease stage')
            atypes = ['1', '2', '3', '4']
        if (tn == 3):
            stage = self.h.getSurvName('c disease stage')
            atype = self.h.getSurvName('c Title')
            atype = [ re.split(",", str(atype[i]))[0] if stage[i] == '4' \
                    else None for i in range(len(atype))]
            atypes = ['class S', 'class F']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getVicente2012(self, tn=1):
        self.prepareData("LM10")
        atype = self.h.getSurvName('c disease status')
        atypes = ['HC', 'FCL', 'SMZL', 'ML', 'DLBCL']
        ahash = {'MALT lymphoma':3,
                'DLBCL':4,
                'SMZL':2,
                'FCL':1,
                'healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBrech2020(self, tn=1):
        self.prepareData("MACV1")
        atype = self.h.getSurvName('c cell type')
        atypes = ['rccM', 'M1', 'M2', 'Mo', 'rcceD', 'Dc']
        ahash = {'macrophages':0,
                'M1 macrophages':1,
                'M2 macrophages':2,
                'Monocytes':3,
                'ercDCs':4,
                'CD1c+ DCs':5}
        self.initData(atype, atypes, ahash)

    def getMontoya2009(self, tn=1):
        self.prepareData("MACV2")
        atype = self.h.getSurvName('c src1')
        atypes = ['BT', 'LL', 'RR']
        ahash = {'lepromatous leprosy':1,
                'borderline tuberculoid leprosy':0,
                'reversal reaction leprosy':2}
        self.initData(atype, atypes, ahash)

    def getBurckstummer2009(self, tn=1):
        self.prepareData("MACV3")
        atype = self.h.getSurvName('c Title')
        atypes = ['U', 'S']
        ahash = {'L929 unstimulated':0,
                'NIH3T3 unstimulated':0,
                'RAW 264.7 unstimulated':0,
                'L929 + IFN-beta (4hrs)':1,
                'NIH3T3 + IFN-beta (4hr)':1,
                'RAW264.7 IFN-beta (4hrs)':1}
        self.initData(atype, atypes, ahash)

    def getMan2015(self, tn=1):
        self.prepareData("MACV4")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Irf1', 'Aim2', 'Ifnar1']
        ahash = {'wild-type':0, 'Irf1-/-':1, 'Aim2-/-':2, 'Ifnar1-/-':3}
        if (tn == 2):
            atypes = ['Wt', '', 'Irf1']
            ahash = {'wild-type':0, 'Irf1-/-':2}
        self.initData(atype, atypes, ahash)

    def getGray2016Hs(self, tn=1):
        self.prepareData("MACV5")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', '4', '12']
        ahash = {'AP1 dimerizer drug (12h)':2,
                'AP1 dimerizer drug (4h)':1,
                'Mock':0}
        self.initData(atype, atypes, ahash)

    def getGray2016Mm(self, tn=1):
        self.prepareData("MACV6")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'T']
        ahash = {'CT-DNA':1, 'mock':0}
        self.initData(atype, atypes, ahash)

    def getKarki2018(self, tn=1):
        self.prepareData("MACV7")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Irf8']
        ahash = {'Wild-type':0, 'IRF8-/-':1}
        self.initData(atype, atypes, ahash)

    def getPan2017(self, tn=1):
        self.prepareData("MACV8")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Tet2']
        ahash = {'wildtype':0, 'Tet2 knockout':1}
        if (tn == 2):
            ctype = self.h.getSurvName('c cell type')
            ahash = {'bone marrow derived macrophages (BMDMs)':0,
                    'tumor associated macrophages (TAMs)':1}
            rval = [ahash[i] if i in ahash else None for i in ctype]
            atype = self.h.getSurvName('c genotype/variation')
            atype = [atype[i] if rval[i] == 1 else None \
                    for i in range(len(atype))]
            atypes = ['Wt', 'Tet2']
            ahash = {'wildtype':0, 'Tet2 knockout':1}
        self.initData(atype, atypes, ahash)

    def getJura2008(self, tn=1):
        self.prepareData("MACV9")
        atype = self.h.getSurvName('c Title')
        atype = [ re.split(",", str(i))[0] for i in atype]
        atypes = ['Control', 'IL-1', 'IL-6']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getGustafsson2008(self, tn=1):
        self.prepareData("MACV10")
        atype = self.h.getSurvName('c tissue')
        atypes = ['Decidual', 'Blood']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getAndreu2017(self, tn=1):
        self.prepareData("MACV11")
        atype = self.h.getSurvName('c infection')
        atypes = ['U', 'D', 'L']
        ahash = {'uninfected':0, 'Dead (Irradiated) Mtb':1, 'Live Mtb':2}
        self.initData(atype, atypes, ahash)

    def getBurke2020(self, tn=1):
        self.prepareData("MACV12")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Irf3/7', 'Irf1', 'Irf5']
        ahash = {'WT BMDMs':0, 'Irf3/7-/- BMDMs':1,
                'Irf1-/- BMDMs':2, 'Irf5-/- BMDMs':3}
        self.initData(atype, atypes, ahash)

    def getAaronson2017(self, tn=1):
        self.prepareData("MACV13")
        atype = self.h.getSurvName('c src1')
        atypes = ['Bv', 'Bi', 'Xv', 'Xi', 'Sv', 'Si', 'Lv', 'Li']
        ahash = {'Cerebellum, Vehicle':0,
                'Cerebellum, CHDI-00340246, 10mg/kg':1,
                'Cortex, Vehicle':2,
                'Cortex, CHDI-00340246, 10mg/kg':3,
                'Striatum, Vehicle':4,
                'Striatum, CHDI-00340246, 10mg/kg':5,
                'Liver, Vehicle':6,
                'Liver, CHDI-00340246, 10mg/kg':7}
        self.initData(atype, atypes, ahash)

    def getFensterl2012(self, tn=1):
        self.prepareData("MACV14")
        atype = self.h.getSurvName('c Title')
        atypes = ['WV6', 'IV6', 'WV2', 'IV2']
        ahash = {'wtbrain-VSV-6d-rep1':0,
                'Ifit2KObrain-VSV-6d-rep1':1,
                'wtbrain-VSV-2d-rep1':2,
                'Ifit2KObrain-VSV-2d-rep1':3}
        if (tn == 2):
            atypes = ['W', '', 'Ifit2']
            ahash = {'wtbrain-VSV-2d-rep1':0,
                    'Ifit2KObrain-VSV-2d-rep1':2}
        self.initData(atype, atypes, ahash)

    def getHorsch2015(self, tn=1):
        self.prepareData("MACV15")
        atype = self.h.getSurvName('c genotype/treatment')
        atypes = ['Wt', 'Mut']
        ahash = {'homozygote':1, 'wild type':0}
        if (tn == 2):
            atype = self.h.getSurvName('c treatment protocol')
            atypes = ['C', 'T']
            ahash = {'control':0, 'OVA challenge':1}
        self.initData(atype, atypes, ahash)

    def getHorsch2015II(self, tn=1):
        self.prepareData("MACV15.2")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Mut']
        ahash = {'Cox4-2 mutant line':1, 'wild type':0}
        if (tn == 2):
            atype = self.h.getSurvName('c treatment')
            atype = [ re.split(" ", str(i))[0] for i in atype]
            atypes = ['C', 'T']
            ahash = {'challenged':1, 'none':0}
        self.initData(atype, atypes, ahash)

    def getHorsch2015III(self, tn=1):
        self.prepareData("MACV15.3")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Mut']
        ahash = {'Prdm11 knockout mice':1, 'wild type':0}
        if (tn == 2):
            atype = self.h.getSurvName('c treatment')
            atype = [ re.split(" ", str(i))[0] for i in atype]
            atypes = ['C', 'T']
            ahash = {'challenged':1, 'none':0}
        self.initData(atype, atypes, ahash)

    def getLee2012(self, tn=1):
        self.prepareData("MACV16")
        atype = self.h.getSurvName('c aim2 expression')
        atypes = ['Wt', 'Mut']
        ahash = {'persistent expression':1, 'absent expression':0}
        self.initData(atype, atypes, ahash)

    def getVasudevan2019(self, tn=1):
        self.prepareData("MACV17")
        atype = self.h.getSurvName('c genotype')
        atypes = ['Wt', 'Mut']
        ahash = {'control':0, 'Sp110 knockdown':1}
        self.initData(atype, atypes, ahash)

    def getIrey2019Hs(self, tn=1):
        self.prepareData("MACV18")
        atype = self.h.getSurvName('c Title')
        atype = [ re.split(",", str(i))[0] for i in atype]
        atypes = ['V', '7', '231']
        ahash = {'MCF7-conditioned-medium':1,
                'vehicle-conditioned-medium':0,
                'MDA-MB-231-conditioned-medium':2}
        media = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName('c Title')
            atype = [ re.split(", ", str(atype[i]))[1] \
                    if len(re.split(", ", str(atype[i]))) > 1 and \
                    media[i] != 2 \
                    else "" for i in range(len(atype))]
            atypes = ['V', 'I', 'I']
            ahash = {'vehicle':0, 'ruxolitinib':2}
        self.initData(atype, atypes, ahash)

    def getIrey2019Mm(self, tn=1):
        self.prepareData("MACV18.2")
        atype = self.h.getSurvName('c phenotype')
        atypes = ['Wt', '', 'STAT3']
        ahash = {'STAT3 wild type':0, 'STAT3 knockout':2}
        self.initData(atype, atypes, ahash)

    def getOakes2017Hs(self, tn=1):
        self.prepareData("MACV19")
        atype = self.h.getSurvName('c src1')
        atypes = ['W-D', 'W+D', 'M-D', 'M+D']
        ahash = {'T47D cell line, WT mouse Oas2, -DOX':0,
                'T47D cell line, MUT mouse Oas2, +DOX':3,
                 'T47D cell line, WT mouse Oas2, +DOX':1,
                 'T47D cell line, MUT mouse Oas2, -DOX':2}
        if (tn == 2):
            atypes = ['W-D', 'W+D']
            ahash = {'T47D cell line, WT mouse Oas2, -DOX':0,
                     'T47D cell line, WT mouse Oas2, +DOX':1}
        self.initData(atype, atypes, ahash)

    def getOakes2017Mm(self, tn=1):
        self.prepareData("MACV19.2")
        atype = self.h.getSurvName('c src1')
        atypes = ['W2', 'W18', 'M2', 'M18']
        ahash = {'Mammary gland, MUT Oas2, 2dpp':2,
                'Mammary gland, WT Oas2, 2dpp':0,
                'Mammary gland, WT Oas2, 18dpc':1,
                'Mammary gland, MUT Oas2, 18dpc':3}
        if (tn == 2):
            atypes = ['W2', '', 'M2']
            ahash = {'Mammary gland, MUT Oas2, 2dpp':2,
                    'Mammary gland, WT Oas2, 2dpp':0}
        self.initData(atype, atypes, ahash)

    def getLinke2017(self, tn=1):
        self.prepareData("MACV20")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Mut']
        ahash = {'Tsc2fl/fl LysM+/+':0, 'Tsc2fl/fl LysM+/cre':1}
        self.initData(atype, atypes, ahash)

    def getQi2017(self, tn=1):
        self.prepareData("MACV21")
        atype = self.h.getSurvName('c src1')
        atype = [ re.sub(".*\((.*)\).*", "\\1", str(i)) for i in atype]
        atypes = ['NC', 'KD']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLi2019(self, tn=1):
        self.prepareData("MACV22")
        atype = self.h.getSurvName('c mouse genotype')
        atypes = ['Wt', 'Rnf5']
        ahash = {'WT':0, 'Rnf5 KO':1}
        self.initData(atype, atypes, ahash)

    def getScortegagna2020(self, tn=1):
        self.prepareData("MACV23")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Siah2']
        ahash = {'Siah2 WT':0, 'Siah2 KO':1}
        self.initData(atype, atypes, ahash)

    def getGoldmann2015(self, tn=1):
        self.prepareData("MACV24")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Ifnar', 'Usp18', 'C61A', 'KD', 'NC', 'DKO']
        ahash = {'WT':0,
                'IFNARko':1,
                'Usp18_C61A':3,
                'Usp18 ko':2,
                'si RNA Usp18':4,
                'si RNA control':5,
                'USP18ko:IFNARko':6}
        if (tn == 2):
            atypes = ['Wt', '', 'DKO']
            ahash = {'WT':0,
                    'USP18ko:IFNARko':2}
        if (tn == 3):
            atypes = ['Ifnar', '', 'DKO']
            ahash = {'IFNARko':0,
                    'USP18ko:IFNARko':2}
        self.initData(atype, atypes, ahash)

    def getKurata2011(self, tn=1):
        self.prepareData("MACV25")
        atype = self.h.getSurvName('c Title')
        atypes = ['Wt', 'S', 'U']
        ahash = {'32Dcl3 Xbp1S':1, '32Dcl3 pMIG':0, '32Dcl3 Xbp1U':2}
        if (tn == 2):
            atypes = ['Wt', 'Xbp1']
            ahash = {'32Dcl3 Xbp1S':1, '32Dcl3 pMIG':0, '32Dcl3 Xbp1U':1}
        self.initData(atype, atypes, ahash)

    def getWang2019Mac(self, tn=1):
        self.prepareData("MACV26")
        atype = self.h.getSurvName('c treatment')
        atypes = ['None', 'IL-15', 'IL-4', 'Media']
        ahash = {}
        if (tn == 2):
            atypes = ['None', 'IL-15']
        self.initData(atype, atypes, ahash)

    def getUckelmann2020(self, tn=1):
        self.prepareData("MACV27")
        rtype = self.h.getSurvName('c cell line')
        ahash = {'IMS-M2':0, 'OCI-AML-3':1}
        line = [ahash[i] if i in ahash else None for i in rtype]
        ttype = self.h.getSurvName('c timepoint')
        ahash = {'day 3':3, 'day 5':5, 'day 7':7}
        time = [ahash[i] if i in ahash else None for i in ttype]
        atype = self.h.getSurvName('c treatment')
        ahash = {'':0, '330nM VTP50469':1}
        treat = [ahash[i] if i in ahash else None for i in atype]
        atype = ["-".join([str(line[i]), str(treat[i]), str(time[i])])
                                    for i in range(len(atype))]
        atypes = ['N', 'T']
        ahash = {'0-0-3':0, '0-0-5':0, '0-0-7':0,
                '1-0-3':0, '1-0-5':0, '1-0-7':0,
                '0-1-3':1, '0-1-5':1, '0-1-7':1,
                '1-1-3':1, '1-1-5':1, '1-1-7':1}
        if (tn == 2):
            ahash = {'1-0-3':0, '1-0-5':0, '1-0-7':0, '1-1-5':1}
        if (tn == 3):
            ahash = {'0-0-3':0, '0-0-5':0, '0-0-7':0,
                    '0-1-3':1, '0-1-5':1, '0-1-7':1}
        if (tn == 4):
            ahash = {'1-0-3':0, '1-0-5':0, '1-0-7':0,
                    '1-1-3':1, '1-1-5':1, '1-1-7':1}
        self.initData(atype, atypes, ahash)

    def getUckelmann2020Mm(self, tn=1):
        self.prepareData("MACV28")
        rtype = self.h.getSurvName('c cell type')
        rtype = [str(i).replace('mouse ', '').replace(' cells', '') \
                for i in rtype]
        ahash = {'':0, 'LSK':1, 'GMP':2, 'LSK-derived GMP':3,
                'GMP-derived GMP':4, 'LSK-derived LSK':5,
                'GMP-derived LSK':6, 'long-term GMP-derived GMP':7,
                'LSK-derived GMP-like':8, 'GMP-derived GMP-like':9}
        line = [ahash[i] if i in ahash else None for i in rtype]
        ttype = self.h.getSurvName('c timepoint')
        ahash = {'4 weeks':28, '':0, '9 months post transplant':270,
                'day 5':5, 'day 3':3}
        time = [ahash[i] if i in ahash else None for i in ttype]
        atype = self.h.getSurvName('c treatment')
        ahash = {'pIpC induction':1, '':0, '1% VTP50469 chow':2, 
                '30nM VTP50469':3}
        treat = [ahash[i] if i in ahash else None for i in atype]
        ctype = ["-".join([str(line[i]), str(treat[i]), str(time[i])])
                                    for i in range(len(atype))]
        atype = treat
        atypes = [0, 1, 2, 3]
        ahash = {}
        if (tn == 2):
            atype = [treat[i] if (line[i] == 0) else None
                    for i in range(len(atype))]
            atypes = [0, 3]
        if (tn == 3):
            atype = [treat[i] if (line[i] == 2) else None
                    for i in range(len(atype))]
            atypes = [0, 1, 2, 3]
        self.initData(atype, atypes, ahash)

    def getDong2019(self, tn=1):
        self.prepareData("MACV29")
        atype = self.h.getSurvName('c genotype')
        atypes = ['W', 'M']
        ahash = {'IRE1 KO':1, 'C57BL/6':0}
        self.initData(atype, atypes, ahash)

    def getGlmozzi2019 (self, tn=1):
        self.prepareData("MACV30")
        atype = self.h.getSurvName('c genotype')
        atypes = ['W', 'M']
        ahash = {'wild type':0, 'PGRMC2 adipose tissue knockout':1}
        self.initData(atype, atypes, ahash)

    def getElDahr2019 (self, tn=1):
        self.prepareData("MACV31")
        atype = self.h.getSurvName('c src1')
        atype = [re.sub("\s*[0-9]$", "", str(i)) for i in atype]
        atypes = ['W', '2', '21het', '21']
        ahash = {'WT':0,
                'EZH2-KO & EZH1-WT':1,
                'EZH2-KO & EZH1-hetKO':2,
                'EZH2-KO & EZH1-homoKO':3}
        if (tn == 2):
            atypes = ['W', 'M']
            ahash = {'WT':0,
                    'EZH2-KO & EZH1-WT':0,
                    'EZH2-KO & EZH1-hetKO':1,
                    'EZH2-KO & EZH1-homoKO':1}
        self.initData(atype, atypes, ahash)

    def getEncode2020 (self, tn=1):
        self.prepareData("MACV32")
        atype = self.h.getSurvName('c Type')
        ahash = {'K562':1, 'HepG2':2, '':0}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Target')
        atypes = ['W', 'M']
        ahash = {'':0, 'EEF2':1, 'NFX1':1, 'HMBOX1':1, 
                'HNRNPA1':1, 'NFYB':1, 'PCBP2':1}
        for k in atype:
            if k != '':
                ahash[k] = 1
        if (tn == 2):
            atype = [atype[i] if rval[i] == 1 else None
                    for i in range(len(atype))]
            ahash = {'':0, 'EEF2':1, 'NFX1':1, 'HMBOX1':1, 
                    'HNRNPA1':1, 'NFYB':1, 'PCBP2':1}
        self.initData(atype, atypes, ahash)

    def getBeheshti2015(self, tn=1):
        self.prepareData("MACV33")
        atype = self.h.getSurvName('c src1')
        atypes = ['C', 'T']
        ahash = {'Spleen from LLC tumor bearing mice':1,
                'Spleen from non-tumor bearing control mice':0}
        self.initData(atype, atypes, ahash)

    def getPlatten2020(self, tn=1):
        self.prepareData("MACV34")
        atype = self.h.getSurvName('c phenotype')
        atypes = ['R', 'NR']
        ahash = {'PD-1 and CTLA-4 non-responder':1,
                'PD-1 and CTLA-4 responder':0}
        self.initData(atype, atypes, ahash)

    def getVolkmer2020(self, tn=1):
        self.prepareData("MACV35")
        atype = self.h.getSurvName('c src1')
        atype = [ re.split(",", str(i))[0] for i in atype]
        ahash = {'MC38 whole tumor':0,
                'B16-OVA whole tumor':1,
                'PDA30364 cell line pellet':2,
                'PDA30364 whole tumor':3,
                'B16-OVA cell line pellet':4,
                'MC38 cell line pellet':5}
        tumor = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment groups')
        atypes = ['C1', 'T1', 'C2', 'T2', 'C3', 'T3', 'C4', 'T4', 'C5', 'T5']
        ahash = {'24h DMSO':0, '24h GDC-0623':1,
                '72h DMSO':2, '72h GDC-0623':3,
                'Vehicle+control IgG':4, 'Vehicle+CD40 mIgG1':5,
                'GEM+control IgG':6, 'GEM+CD40 mIgG1':7,
                'MEKi+control IgG':8, 'MEKi+CD40 mIgG1':9}
        treat = [ahash[i] if i in ahash else None for i in atype]
        if (tn >= 2):
            atype = ["-".join([str(tumor[i]), str(treat[i])])
                                    for i in range(len(atype))]
            atypes = ['C', 'T']
            list1 = ['0-4', '0-5', '0-8', '0-9',
                    '1-4', '1-5', '1-8', '1-9',
                    '2-0', '2-1', '2-2', '2-3',
                    '3-4', '3-5', '3-6', '3-7',
                    '4-0', '4-1', '4-2', '4-3',
                    '5-0', '5-1', '5-2', '5-3']
            index = (tn - 2) * 2;
            ahash = { list1[index] : 0, list1[index + 1]:1}
        self.initData(atype, atypes, ahash)

    def getVolkmer2020II(self, tn=1):
        self.prepareData("MACV36")
        atype = self.h.getSurvName('c src1')
        atype = [ re.split(",", str(i))[0] for i in atype]
        ahash = {'MC38 whole tumor':0,
                'B16-OVA whole tumor':1,
                'PDA30364 cell line pellet':2,
                'PDA30364 whole tumor':3,
                'B16-OVA cell line pellet':4,
                'MC38 cell line pellet':5}
        tumor = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment groups')
        atypes = ['C1', 'T1', 'C2', 'T2', 'C3', 'T3', 'C4', 'T4', 'C5', 'T5']
        ahash = {'24h DMSO':0, '24h GDC-0623':1,
                '72h DMSO':2, '72h GDC-0623':3,
                'Vehicle+control IgG':4, 'Vehicle+CD40 mIgG1':5,
                'GEM+control IgG':6, 'GEM+CD40 mIgG1':7,
                'MEKi+control IgG':8, 'MEKi+CD40 mIgG1':9}
        treat = [ahash[i] if i in ahash else None for i in atype]
        if (tn >= 2):
            atype = ["-".join([str(tumor[i]), str(treat[i])])
                                    for i in range(len(atype))]
            atypes = ['C', 'T']
            list1 = ['0-4', '0-5', '0-8', '0-9',
                    '1-4', '1-5', '1-8', '1-9',
                    '2-0', '2-1', '2-2', '2-3',
                    '3-4', '3-5', '3-6', '3-7',
                    '4-0', '4-1', '4-2', '4-3',
                    '5-0', '5-1', '5-2', '5-3']
            index = (tn - 2) * 2;
            ahash = { list1[index] : 0, list1[index + 1]:1}
        self.initData(atype, atypes, ahash)

    def getZhou2020(self, tn=1):
        self.prepareData("MACV37")
        atype = self.h.getSurvName('c treatment')
        atypes = ['T1', 'T2']
        ahash = {'anti-gp120':0, 'anti-MerTK':1}
        self.initData(atype, atypes, ahash)

    def getZhou2020II(self, tn=1):
        self.prepareData("MACV38")
        atype = self.h.getSurvName('c treatment')
        atypes = ['T1', 'T2']
        ahash = {'anti-gp120':0, 'anti-MerTK':1}
        self.initData(atype, atypes, ahash)

    def getSteenbrugge2019(self, tn=1):
        self.prepareData("MACV39")
        atype = self.h.getSurvName('c src1')
        atypes = ['N', 'T', 'N1', 'T1']
        ahash = {'C57BL/6-derived mammary gland':0,
                '4T1 tumor':1,
                'BALB/c-derived mammary gland':2,
                'Py230 tumor':3}
        if (tn == 2):
            atypes = ['N', 'T']
            ahash = {'C57BL/6-derived mammary gland':0,
                'Py230 tumor':1}
        if (tn == 3):
            atypes = ['N', 'T']
            ahash = {'BALB/c-derived mammary gland':0,
                    '4T1 tumor':1}
        if (tn == 4):
            atypes = ['BALB/c', 'C57BL/6']
            ahash = {'BALB/c-derived mammary gland':0,
                    'C57BL/6-derived mammary gland':1}
        self.initData(atype, atypes, ahash)

    def getHollern2019(self, tn=1):
        self.prepareData("MACV40")
        atype = self.h.getSurvName('c class')
        atypes = ['S', 'R']
        ahash = {'sensitive':0, 'resistant':1}
        self.initData(atype, atypes, ahash)

    def getDas2015(self, tn=1):
        self.prepareData("MACV41")
        atype = self.h.getSurvName('c agent')
        atypes = ['aCTLA4', 'Combo', 'Seq', 'aPD1']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName('c time point')
            atypes = ['pre', 'post', 'pre-s', 'post-s']
            ahash = {'post':1, 'pre':0,
                    'pre sequential therapy sample':2,
                    'post sequential therapy sample':3}
        self.initData(atype, atypes, ahash)

    def getTaube2015(self, tn=1):
        self.prepareData("MACV42")
        atype = self.h.getSurvName('c pd-l1 status')
        atypes = ['pos', 'neg']
        ahash = {'PD-L1 positive':0, 'PD-L1 negative':1}
        self.initData(atype, atypes, ahash)

    def getBalachandran2019(self, tn=1):
        self.prepareData("MACV43")
        atype = self.h.getSurvName('c genotype')
        atypes = ['W', 'M']
        ahash = {'wild type':0, 'IL33-/-':1}
        self.initData(atype, atypes, ahash)

    def getHwang2020(self, tn=1):
        self.prepareData("MACV44")
        atype = self.h.getSurvName('c tumor type')
        atypes = ['L', 'A', 'AS', 'S']
        ahash = {'Large cell neuroendocarine carcinoma':0,
                'adenocarcinoma':1,
                'adenocarcinoma-squamous cell carcinoma':2,
                'squamous cell carcinoma':3}
        self.initData(atype, atypes, ahash)

    def getSimpson2019(self, tn=1):
        self.prepareData("MACV45")
        atype = self.h.getSurvName('c patient outcome')
        atypes = ['C', 'I', 'F']
        ahash = {'control':0,
                'acute liver failure':2,
                'acute liver injury':1}
        if (tn == 2):
            atype = self.h.getSurvName('c survival')
            atypes = ['S', 'D']
            ahash = {'spontaneously survived':0, 'dead or transplanted':1}
        self.initData(atype, atypes, ahash)

    def getHuang2019(self, tn=1):
        self.prepareData("MACV46")
        atype = self.h.getSurvName('c fvc_progressor_10%')
        atypes = ['0', '1']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName('c dlco_progressor_15%')
            atypes = ['0', '1']
        self.initData(atype, atypes, ahash)

    def getYu2019(self, tn=1):
        self.prepareData("MACV47")
        course = self.h.getSurvName('c course')
        ahash = {'Acute myocarditis':0,
                'Acute myocardial infarction':1,
                'Aortic dissection':2,
                'Congestive heart failure':3,
                'Dilated cardiomyopathy':4,
                'Dilated cardiomyopathy, DCMP':5,
                'Arrhythmia':6}
        rval = [ahash[i] if i in ahash else None for i in course]
        atype = self.h.getSurvName('c outcome')
        atypes = ['S', 'F']
        ahash = {'Success':0, 'Failure':1, 'failure':1}
        if (tn == 2):
            atype = course
            atypes = ['Am', 'Ami', 'Ad', 'Chf', 'Dc', 'Dcmp', 'Ar']
            ahash = {'Acute myocarditis':0, 'Acute myocardial infarction':1,
                    'Aortic dissection':2, 'Congestive heart failure':3,
                    'Dilated cardiomyopathy':4, 'Dilated cardiomyopathy, DCMP':5,
                    'Arrhythmia':6}
        if (tn == 3):
            atype = [atype[i] if rval[i] != 4 and rval[i] != 5 else None
                    for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if rval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getVanhaverbeke2019(self, tn=1):
        self.prepareData("MACV48")
        time = self.h.getSurvName('c time point')
        status = self.h.getSurvName('c patient diagnosis')
        atype = [" ".join([str(status[i]), str(time[i])])
                for i in range(len(time))]
        atypes = ['MI D0', 'MI D30', 'MI Y1', 'CAD D0']
        ahash = {}
        if (tn == 2):
            atypes = ['CAD D0', 'MI D0']
        if (tn == 3):
            atypes = ['MI D30', 'MI D0']
        self.initData(atype, atypes, ahash)

    def getSuresh2014(self, tn=1):
        self.prepareData("MACV49")
        atype = self.h.getSurvName('c disease status')
        atypes = ['C', 'Nr', 'r']
        ahash = {'normal control':0,
                'patient without recurrent events':1,
                'patient with recurrent events':2}
        if (tn == 2):
            atypes = ['C', 'r']
            ahash = {'normal control':0,
                    'patient with recurrent events':1}
        self.initData(atype, atypes, ahash)

    def getPalmer2019(self, tn=1, tb=0):
        self.prepareData("MACV50")
        atype = self.h.getSurvName('c src1')
        ahash = {'colon':0, 'blood':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease type')
        atypes = ['C', 'IBD', 'UC', 'CD']
        ahash = {'Control':0,
                'Control - infect':0,
                'Control - celiac':0,
                'Control (E.coli) --> CD':0,
                'Control--> CD':0,
                'IBD w/ oral':1,
                'IBDU':1,
                'UC - pancolitis':2,
                'UC - L colitis':2,
                'UC - proctitis':2,
                "Crohn's Disease":3}
        if (tn == 2):
            atypes = ['C', 'IBD']
            ahash = {'Control':0,
                    'Control - infect':0,
                    'Control - celiac':0,
                    'Control (E.coli) --> CD':0,
                    'Control--> CD':0,
                    'UC - pancolitis':1,
                    'UC - L colitis':1,
                    'UC - proctitis':1,
                    "Crohn's Disease":1}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getOstrowski2019(self, tn=1):
        self.prepareData("MACV51")
        atype = self.h.getSurvName('c src1')
        atypes = ['C', 'UC', 'CD', 'PSC', 'PBC', 'UCc', 'CDc']
        ahash = {'Control, adult':0,
                'Ulcerative colitis, adult':1,
                'Crohn\xe2\x80\x99s disease, adult':2,
                'Primary sclerosing cholangitis, adult':3,
                'Primary biliary cholangitis, adult':4,
                'Ulcerative colitis, child':5,
                'Crohn\xe2\x80\x99s disease, child':6}
        if (tn == 2):
            atypes = ['C', 'UC', 'CD']
            ahash = {'Control, adult':0,
                    'Ulcerative colitis, adult':1,
                    'Crohn\xe2\x80\x99s disease, adult':2}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getJeffrey2006(self, tn=1):
        self.prepareData("MACV52")
        src = self.h.getSurvName('c src1')
        ahash = {'Cord blood':0, 'Peripheral blood':1}
        rval = [ahash[i] if i in ahash else None for i in src]
        title = self.h.getSurvName('c Title')
        atype = [1 if str(k).find("unstimulated") >= 0 or \
                str(k).find("control") >= 0 or \
                str(k).find("Immature") >= 0 else 0 for k in title]
        atypes = ['C', 'S']
        ahash = {0:1, 1:0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSippel2018(self, tn=1):
        self.prepareData("MACV53")
        atype = self.h.getSurvName('c treatment')
        atypes = ['V', 'IL5', 'PGD2']
        ahash = {'IL5':1, 'Vehicle':0, 'dkPGD2':2}
        if (tn == 2):
            atypes = ['V', 'T']
            ahash = {'IL5':1, 'Vehicle':0, 'dkPGD2':1}
        self.initData(atype, atypes, ahash)

    def getPuan2017(self, tn=1):
        self.prepareData("MACV54")
        state = self.h.getSurvName('c donor_state')
        ahash = {'reactive':0, 'anergic':1}
        rval = [ahash[i] if i in ahash else None for i in state]
        atype = self.h.getSurvName('c stimulation')
        atypes = ['C', 'S']
        ahash = {'unstimulated':0, 'Fc-epsilon receptor-crosslinking':1}
        self.pair = [ [2, 11], [3, 6], [7, 5], [4, 9], [10, 8]]
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKlocperk2020(self, tn=1):
        self.prepareData("MACV55")
        src = self.h.getSurvName('c src1')
        src = [re.sub("mo.*of ", "", str(k)) for k in src]
        treat = [re.sub(".* [1-5] *", "", str(k)) for k in src]
        treat = [re.sub("c.* with ", "", str(k)) for k in treat]
        ahash = {'':0,
                'autologous healthy NETs':1,
                'healthy NETs':2,
                'T1D NETs':3,
                'autologous T1D NETs':4}
        rval = [ahash[i] if i in ahash else None for i in treat]
        disease = [k.split(" ")[0] for k in src]
        atype = disease
        atypes = ['healthy', 'T1D']
        ahash = {}
        if (tn == 2):
            atypes = ['C', 'S']
            atype = rval
            ahash = {0:0, 1:1, 2:1, 3:1, 4:1}
            atype = [atype[i] if disease[i] == 'healthy' \
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'S']
            atype = rval
            ahash = {0:0, 1:1, 2:1, 3:1, 4:1}
            atype = [atype[i] if disease[i] == 'T1D' \
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLenardo2020(self, tn=1):
        self.prepareData("MACV58")
        atype = self.h.getSurvName('c time')
        atypes = ['D2', ' ', 'D5']
        ahash = {'Day2':0, 'Day5':2}
        self.initData(atype, atypes, ahash)

    def getNair2015(self, tn=1):
        self.prepareData("MACV59")
        atype = self.h.getSurvName('c protocol')
        atypes = ['C', 'S']
        ahash = {'unstimulated (control)':0, 'stimulated':1}
        self.initData(atype, atypes, ahash)

    def getJohnson2020(self, tn=1):
        self.prepareData("MACV60")
        atype = self.h.getSurvName("c treatment_code1")
        atypes = ['C', 'S']
        ahash = {'NS':0}
        for k in atype:
            if k != 'NS':
                ahash[k] = 1
        if (tn == 2):
            atypes = ['C', 'HIV']
            ahash = {'NS': 0, 'HIV2_WT':1, 'HIV2_P86HA':1}
        self.initData(atype, atypes, ahash)

    def getAbbas2005(self, tn=1):
        self.prepareData("MACV61")
        atype = self.h.getSurvName("c src1")
        ahash = {'NK cells from PBMC':0,
                'Plasma cells from bone marrow':1,
                'Monocytes from PBMC':2,
                'CD4+ T cells from PBMC':3,
                'B cells from PBMC':4,
                'Neutrophils from PBMC':5,
                'CD14+ cells from PBMC':6,
                'CD4+ CD45RO+ CD45RA- T cells from PBMC':7,
                'CD8+ T cells from PBMC':8,
                'Plasma cells from PBMC':9}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment agent')
        ahash = {'NA':0,
                'macrophage differentiation medium':1,
                'IL2':2,
                'LPS':3,
                'IL15':4,
                'aCD3/aCD28':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = tval
        atypes = ['C', 'S']
        ahash = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1}
        self.initData(atype, atypes, ahash)

    def getMetcalf2015(self, tn=1):
        self.prepareData("MACV62")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'S']
        ahash = {'Rig I':1, 'PolyIC':1, 'NoTx':0, 'LyoVec_only':1, 'LPS':1, 'CLO_97':1}
        self.initData(atype, atypes, ahash)

    def getBanchereau2014I(self, tn=1):
        self.prepareData("MACV63")
        atype = self.h.getSurvName('c cell population')
        ahash = {'IL4 DC':1, 'IFNa DC':0}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c stimulation')
        atypes = ['C', 'S']
        ahash = {'MDP':1, 'LPS':1, 'Poly I:C-LMW-Lyovec':1,
                'TNFa':1, 'CpG 2216':0, 'Poly I:C':1, 'R837':1,
                'CL097':1, 'IFNa':1, 'IL10':1, 'CpG 2006':0, 'Flagellin':1,
                'PAM3':1, 'IL15':1, 'IL1b':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            atype = rval
            ahash = {0:1, 1:2}
            atype = [atype[i] if aval[i] == 0 \
                    else None for i in range(len(atype))]
        if (tn == 3):
            ahash['A'] = 1
            atype = [atype[i] if rval[i] == 1 \
                    else 'A' for i in range(len(atype))]
        if (tn == 4):
            atypes = ['C', 'S']
            ahash = {'CpG 2216':0, 'CpG 2006':0, 'IL1b':1}
        self.initData(atype, atypes, ahash)

    def getBanchereau2014II(self, tn=1):
        self.prepareData("MACV64")
        atype1 = self.h.getSurvName('c culture conditions')
        atype2 = self.h.getSurvName('c culture condition')
        atype3 = self.h.getSurvName('c vaccine abbr.')
        atype = [ " ".join([str(k) for k in [atype1[i], atype2[i], atype3[i]]]) 
                         for i in range(len(atype1))]
        atypes = ['C', 'S']
        ahash = {'  RAB':1, 'LPS  ':1, ' HKSE ':1,
                ' Media ':0, '  Medium':0, 'Medium1  ':0,
                '  FZ':1, '  TDAP':1, ' H1N1 Brisbane ':1, ' HKSA ':1, 'Medium2  ':1,
                'HEPB  ':1, 'HPV  ':1, 'HIB  ':1, '  POL':1, '  VAR':1, 'VAR  ':1,
                'PVX  ':1, 'RAB  ':1, 'MGL  ':1, '  HIB':1, '  HER':1, '  HPV':1,
                'HER  ':1, '  PVX':1, '  JPE':1, '  HEPB':1, 'HEPA  ':1, 'JPE  ':1,
                '  HEPA':1, 'FZ  ':1, 'POL  ':1, '  MGL':1, 'TDAP  ':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName('c cell population')
            atypes = ['M0', 'M1', 'M2']
            ahash = {'IL4 DC':2, 'IFNa DC':1, 'Monocytes':0}
            atype = [atype[i] if aval[i] == 0 \
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHarfuddin2016(self, tn=1):
        self.prepareData("MACV65")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'S']
        ahash = {'None':0, 'CD137-Fc protein':1, 'GM-CSF + IL-4':1,
                'GM-CSF + IL-4; matured with LPS + IFNg':1,
                'Fc protein':1, 'M-CSF':1}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'None':0, 'GM-CSF + IL-4':2,
                    'GM-CSF + IL-4; matured with LPS + IFNg':1}
        self.initData(atype, atypes, ahash)

    def getIampietro2017(self, tn=1):
        self.prepareData("MACV66")
        atype = self.h.getSurvName('c infection')
        atypes = ['Mock', 'EBOV', 'LPS']
        ahash = {}
        if (tn == 2):
            atypes = ['C', 'S']
            ahash = {'Mock':0, 'EBOV':1, 'LPS':1}
        self.initData(atype, atypes, ahash)

    def getSinnaeve2009(self, tn=1):
        self.prepareData("MACV68")
        atype = self.h.getSurvName('c CADi')
        atypes = ['C', 'D']
        ahash = {'0':0}
        for k in atype:
            if k != '0':
                ahash[k] = 1
        self.initData(atype, atypes, ahash)

    def getMartinez2019(self, tn=1):
        self.prepareData("MACV69")
        atype = self.h.getSurvName('c outcome')
        atypes = ['M', 'F']
        ahash = {'Matured':0, 'Failed':1}
        self.initData(atype, atypes, ahash)

    def getHollander2013(self, tn=1):
        self.prepareData("MACV70")
        atype = self.h.getSurvName('c src1')
        atypes = ['non-AR', 'AR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBaron2007I(self, tn=1):
        self.prepareData("MACV71")
        atype = self.h.getSurvName('c Title')
        cells = [re.split("_", str(i))[1] if len(re.split("_", str(i))) > 1
                                else None for i in atype]
        acute = [re.split("_", str(i))[3] if len(re.split("_", str(i))) > 3
                                else None for i in atype]
        chronic = [re.split("_", str(i))[4] if len(re.split("_", str(i))) > 4
                                else None for i in atype]
        atype = acute
        atypes = ['C', 'D']
        ahash = {'aGVHD+':1, 'aGVHD-':0}
        if (tn == 2):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
        if (tn == 3):
            atype = [atype[i] if cells[i] == 'CD4+' \
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if cells[i] == 'CD8+' \
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
            atype = [atype[i] if cells[i] == 'CD4+' \
                    else None for i in range(len(atype))]
        if (tn == 6):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
            atype = [atype[i] if cells[i] == 'CD8+' \
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBaron2007II(self, tn=1):
        self.prepareData("MACV72")
        atype = self.h.getSurvName('c Title')
        cells = [re.split("_", str(i))[1] if len(re.split("_", str(i))) > 1
                                else None for i in atype]
        acute = [re.split("_", str(i))[3] if len(re.split("_", str(i))) > 3
                                else None for i in atype]
        chronic = [re.split("_", str(i))[4] if len(re.split("_", str(i))) > 4
                                else None for i in atype]
        atype = acute
        atypes = ['C', 'D']
        ahash = {'aGVHD+':1, 'aGVHD-':0}
        if (tn == 2):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
        if (tn == 3):
            atype = [atype[i] if cells[i] == 'CD4+' \
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if cells[i] == 'CD8+' \
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
            atype = [atype[i] if cells[i] == 'CD4+' \
                    else None for i in range(len(atype))]
        if (tn == 6):
            atype = chronic
            ahash = {'cGVHD+':1, 'cGVHD-':0}
            atype = [atype[i] if cells[i] == 'CD8+' \
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSagoo2010(self, tn=1):
        self.prepareData("MACV73")
        atypes = ["Tol-DF", "s-LP", "s-CNI", "s-nCNI", "CR", "HC"]
        atype = self.h.getSurvName("c Tol-DF")
        for k in atypes:
            g1 = self.h.getSurvName("c " + k)
            for i in range(len(g1)):
                if (g1[i] == '1'):
                    atype[i] = k
        ahash = {}
        if (tn == 2):
            atypes = ['HC', 'SD', 'CR']
            ahash = {"Tol-DF":1, "s-LP":1, "s-CNI":1, "s-nCNI":1, "CR":2, "HC":0}
        if (tn == 3):
            atypes = ['HC', 'CR', 'SD']
            ahash = {"Tol-DF":2, "CR":1, "HC":0}
        self.initData(atype, atypes, ahash)

    def getHerazoMaya2013(self, tn=1):
        self.prepareData("MACV75")
        atype = self.h.getSurvName("c outcome")
        atypes = ['0', '1']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLi2012I(self, tn=1):
        self.prepareData("MACV76")
        atype = self.h.getSurvName("c disease state")
        atype = [str(k)[0:2] for k in atype]
        atypes = ['ST', 'AR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLi2012II(self, tn=1):
        self.prepareData("MACV77")
        atype = self.h.getSurvName("c Disease State")
        atype = [str(k)[0:1] for k in atype]
        atypes = ['S', 'A']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getKurian2014(self, tn=1):
        self.prepareData("MACV79")
        atype = self.h.getSurvName("c phenotype")
        atypes = ['S', 'R']
        ahash = {'Acute Kidney Rejection':1,
                'Well-functioning kidney transplant':0}
        self.initData(atype, atypes, ahash)

    def getZhang2019(self, tn=1):
        self.prepareData("MACV80")
        atype = self.h.getSurvName("c acr before or at 6m")
        atypes = ['S', 'R']
        ahash = {'ACR or Borderline':1, 'None':0}
        if (tn == 2):
            atype = self.h.getSurvName("c acr after 6m")
            atypes = ['S', 'B', 'R']
            ahash = {'Borderline':1, 'None':0, 'ACR':2}
        if (tn == 3):
            atype = self.h.getSurvName("c acr after 6m")
            atypes = ['S', 'R']
            ahash = {'Borderline':0, 'None':0, 'ACR':1}
        self.initData(atype, atypes, ahash)

    def getKhatri2013(self, tn=1):
        self.prepareData("MACV81")
        atype = self.h.getSurvName("c patient group")
        atypes = ['S', 'R']
        ahash = {'stable patient (STA)':0,
                'patient with acute rejection (AR)':1}
        self.initData(atype, atypes, ahash)

    def getEinecke2010(self, tn=1):
        self.prepareData("MACV82")
        atype = self.h.getSurvName("c rejection/non rejection")
        atypes = ['S', 'R']
        ahash = {'rej':1, 'nonrej':0}
        self.initData(atype, atypes, ahash)

    def getReeve2013(self, tn=1):
        self.prepareData("MACV83")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['S', 'R']
        ahash = {'non-rejecting':0, 'MIXED':1, 'ABMR':1, 'TCMR':1, 'Nephrectomy':1}
        self.initData(atype, atypes, ahash)

    def getRay2007(self, tn=1):
        self.prepareData("MACV84")
        atype = self.h.getSurvName("c Title")
        atype = [re.split(" ", str(i))[2] if len(re.split(" ", str(i))) > 2
                                                else None for i in atype]
        atypes = ['S', 'R']
        ahash = {'not':0, 'PGD':1}
        self.initData(atype, atypes, ahash)

    def getVanLoon2019(self, tn=1):
        self.prepareData("MACV88")
        atype = self.h.getSurvName("c tissue")
        ahash = {'BLOOD':0, 'kidney allograft biopsy':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        tcmr = self.h.getSurvName("c tcmr (no:0_borderline:1_TCMR:2)")
        abmr = self.h.getSurvName("c abmr (no:0_Yes:1)")
        atype = [str(tcmr[i]) + " " + str(abmr[i]) for i in range(len(tcmr))]
        atypes = ['S', 'R']
        ahash = {'2 0':1, '0 1':1, '0 0':0, '1 0':1, '2 1':1, '1 1':1}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 \
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 \
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMorgun2006I(self, tn=1):
        self.prepareData("MACV89")
        atype = self.h.getSurvName("c Final Clinical diagnosis")
        atypes = ['N', 'Ch', 'Pre-Ch', 'R', 'Pre-R', 'Tox']
        ahash = {'toxoplasma myocarditis':5}
        if (tn == 2):
            atypes = ['S', 'R']
            ahash = {'R':1, 'N':0}
        self.initData(atype, atypes, ahash)

    def getMorgun2006II(self, tn=1):
        self.prepareData("MACV90")
        atype = self.h.getSurvName("c Final Clinical diagnosis")
        atypes = ['N', 'R', 'Pre-R']
        ahash = {}
        if (tn == 2):
            atypes = ['S', 'R']
            ahash = {'R':1, 'N':0}
        self.initData(atype, atypes, ahash)

    def getShannon2012(self, tn=1):
        self.prepareData("MACV91")
        atype = self.h.getSurvName("c rejection status")
        atypes = ['NR', 'AR']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getWells2003(self, tn=1):
        self.prepareData("MACV93")
        atype = self.h.getSurvName("c Title")
        strain = [str(k).split("_")[1] if len(str(k).split("_")) > 1 else None
                for k in atype]
        ahash = {'BalbC':0, 'C57BL6':3, 'C3H/ARC':2, 'C3H/HeJ':1, 'C57/BL6':4}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = [re.sub(".*time", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'7h':1, '21h':None, '0.5h':0, '2h':1, '0h':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] >= 3 else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if rval[i] == 2 else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 6):
            atype = strain
            atypes = ['BalbC', 'C3H/HeJ', 'C3H/ARC', 'C57/BL6']
            ahash = {'BalbC':0, 'C57BL6':3, 'C3H/ARC':2, 'C3H/HeJ':1, 'C57/BL6':3}
        if (tn == 7):
            atype = strain
            atypes = ['BalbC', 'C57/BL6']
            ahash = {'BalbC':0, 'C57BL6':1, 'C57/BL6':1}
        self.initData(atype, atypes, ahash)

    def getvanErp2006(self, tn=1):
        self.prepareData("MACV94")
        atype = self.h.getSurvName("c Title")
        strain = [str(k).split("_")[0] if len(str(k).split("_")) > 0 else None
                for k in atype]
        ahash = {'Bc':0, 'Bl6':1}
        rval = [ahash[i] if i in ahash else None for i in strain]
        treat = [str(k).split("_")[1] if len(str(k).split("_")) > 1 else None
                for k in atype]
        ahash = {'WAP':1, 'p60':2, 'MOCK':0, 'Mock':0}
        tval = [ahash[i] if i in ahash else None for i in treat]
        atype = [str(k).split("_")[2] if len(str(k).split("_")) > 2 else None
                for k in atype]
        atypes = ['M0', 'M1']
        ahash = {'IFNy':1, '1':0, '3':0, '2':0, '4':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atypes = ['Bc', 'Bl6']
            ahash = {}
        if (tn == 5):
            atype = strain
            atype = [atype[i] if tval[i] == 0 else None for i in range(len(atype))]
            atypes = ['Bc', 'Bl6']
            ahash = {}
        self.rval = [ahash[i] if i in ahash else None for i in strain]
        self.initData(atype, atypes, ahash)

    def getTang2020(self, tn=1):
        self.prepareData("MACV95")
        atype = self.h.getSurvName("c macrophage phenotype")
        ahash = {'Tissue resident, F4/80hi CD206-':1,
                'Monocyte-derived, F4/80int CD206+':0}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Title")
        strain = [str(k).split("_")[0] if len(str(k).split("_")) > 0 else None
                for k in atype]
        ahash = {'B6':1, 'Balbc':0}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = [str(k).split("_")[1] if len(str(k).split("_")) > 1 else None
                for k in atype]
        atypes = ['IL4c', 'ThioIL4c']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atype = [atype[i] if mval[i] == 0 else None for i in range(len(atype))]
            atypes = ['Balbc', 'B6']
            ahash = {}
        if (tn == 5):
            atype = strain
            atype = [atype[i] if mval[i] == 1 else None for i in range(len(atype))]
            atypes = ['Balbc', 'B6']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getSajti2020(self, tn=1):
        self.prepareData("MACV96.2")
        atype = self.h.getSurvName("c Title")
        group = [re.sub("(.*h[_in]*).*", "\\1", str(k)) for k in atype]
        atype = [str(k).split("_")[0] if len(str(k).split("_")) > 0 else None
                for k in atype]
        ahash = {'AM':0, 'IM':1, 'IMo':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment time")
        ahash = {'20h':20, '6h':6, '':0, '2h':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'LPSi', 'LPSn']
        ahash = {'nasal LPS':2, 'IP LPS':1, '':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if rval[i] == 2 else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if tval[i] <= 2 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSajti2020II(self, tn=1):
        self.prepareData("MACV96.3")
        strain = self.h.getSurvName("c strain")
        ahash = {'C57BL6/J':0, 'DBA2/J':1}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = self.h.getSurvName("c Title")
        atype = [str(k).split(" ")[0] if len(str(k).split(" ")) > 0 else None
                                for k in atype]
        atypes = ['AM', 'IM', 'iMo', 'pMo']
        ahash = {'AM':0, 'IM':1, 'iMo':2, 'pMo':3}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atypes = ['BL6', 'DBA']
            ahash = {'C57BL6/J':0, 'DBA2/J':1}
        if (tn >= 5):
            atype = strain
            atype = [atype[i] if aval[i]==(tn-5) else None for i in range(len(atype))]
            atypes = ['BL6', 'DBA']
            ahash = {'C57BL6/J':0, 'DBA2/J':1}
        self.initData(atype, atypes, ahash)

    def getShemer2018(self, tn=1):
        self.prepareData("MACV97.2")
        atype = self.h.getSurvName("c Title")
        atype = [str(k).split("_")[4] if len(str(k).split("_")) > 4 else None
                for k in atype]
        atypes = ['M0', 'M1']
        ahash = {'LPS':1, 'control':0}
        self.initData(atype, atypes, ahash)

    def getLink2018(self, tn=1):
        self.prepareData("MACV98")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_rep.*", "", str(k)) for k in atype]
        group = [re.sub("BMDM_RNA_polyA_", "", str(k)) for k in atype]
        strain = self.h.getSurvName("c strain")
        ahash = {'C57BL/6J':1, 'SPRET/EiJ':2, 'BALB/cJ':0, 'NOD/ShiLtJ':3, 'PWK/PhJ':4}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = self.h.getSurvName("c ligands in culture")
        atypes = ['M0', 'M1']
        ahash = {'no treatment':0, 'KLA 6h':1, 'KLA 1h':1}
        if (tn == 2):
            atype = group
            ahash = {'BALB_notx':0, 'BALB_KLA_1h':1}
        if (tn == 3):
            atype = group
            ahash = {'C57_notx_6h':0, 'C57_KLA_6h':1}
        if (tn == 4):
            atype = group
            atypes = ['Bl6', 'Bl6t', 'Bc', 'Bct']
            ahash = {'C57_notx_6h':0, 'C57_KLA_6h':1,
                    'BALB_notx':2, 'BALB_KLA_1h':3}
        if (tn == 5):
            atype = group
            atypes = ['Bc', 'Bl6']
            ahash = {'C57_notx_6h':1, 'C57_KLA_6h':1,
                    'BALB_notx':0, 'BALB_KLA_1h':0}
        self.initData(atype, atypes, ahash)

    def getMunck2019(self, tn=1):
        self.prepareData("MACV99")
        strain = self.h.getSurvName("c strain")
        ahash = {'C57BL/6':1, 'BALB/c':0}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = self.h.getSurvName("c infection status")
        atypes = ['M0', 'M1']
        ahash = {'infected':1, 'control':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atypes = ['Bc', 'B6']
            ahash = {'C57BL/6':1, 'BALB/c':0}
        self.initData(atype, atypes, ahash)

    def getElderman2018(self, tn=1):
        self.prepareData("MACV100")
        strain = self.h.getSurvName("c strain")
        ahash = {'Balb/c':0, 'C57bl/6':1}
        rval = [ahash[i] if i in ahash else None for i in strain]
        atype = self.h.getSurvName("c tissue")
        atypes = ['colon', 'ileum']
        ahash = {'distal ileum':1, 'proximal colon':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atypes = ['Bc', 'B6']
            ahash = {'Balb/c':0, 'C57bl/6':1}
        if (tn == 5):
            atype = strain
            atype = [atype[i] if aval[i] == 0 else None for i in range(len(atype))]
            atypes = ['Bc', 'B6']
            ahash = {'Balb/c':0, 'C57bl/6':1}
        self.initData(atype, atypes, ahash)

    def getHowes2016(self, tn=1):
        self.prepareData("MACV101")
        strain = self.h.getSurvName("c strain")
        ahash = {'C57BL/6':1, 'BALB/c':0}
        rval = [ahash[i] if i in ahash else None for i in strain]
        gtype = self.h.getSurvName("c genotype/variation")
        ahash = {'WT':0, 'IFNabRKO':1}
        gval = [ahash[i] if i in ahash else None for i in gtype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1']
        ahash = {'HkBps':1, 'media':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 1 else None for i in range(len(atype))]
        if (tn == 4):
            atype = strain
            atypes = ['Bc', 'B6']
            ahash = {'C57BL/6':1, 'BALB/c':0}
        if (tn == 5):
            atype = strain
            atype = [atype[i] if aval[i] == 0 and gval[i] == 0 \
                    else None for i in range(len(atype))]
            atypes = ['Bc', 'B6']
            ahash = {'C57BL/6':1, 'BALB/c':0}
        self.initData(atype, atypes, ahash)

    def getJochems2018(self, tn=1):
        self.prepareData("MACV102")
        atype = self.h.getSurvName("c src1")
        atypes = ['TIV', 'LAIV']
        ahash = {'LAIV_nasal cells_D-4':1,
                'TIV_nasal cells_D-4':0,
                'LAIV_nasal cells_D2':1,
                'TIV_nasal cells_D2':0,
                'LAIV_nasal cells_D9':1,
                'TIV_nasal cells_D9':0}
        if (tn == 2):
            atypes = ['TIV', 'LAIV']
            ahash = {'LAIV_nasal cells_D2':1,
                    'TIV_nasal cells_D2':0}
        if (tn == 3):
            atype = self.h.getSurvName("c carriage status")
            atypes = ["NEG", "POS"]
            ahash = {}
        if (tn == 4):
            ahash = {'LAIV_nasal cells_D2':1,
                    'TIV_nasal cells_D2':0}
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = self.h.getSurvName("c carriage status")
            atype = [atype[i] if aval[i] is not None else None for i in range(len(atype))]
            atypes = ["NEG", "POS"]
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getZhai2015(self, tn=1):
        self.prepareData("MACV103")
        atype = self.h.getSurvName("c time point")
        atypes = ['C', 'I']
        ahash = {'Baseline':0, 'Spring':0, 'Day0':1, 'Day6':1, 'Day21':1,
                'Day2':1, 'Day4':1}
        if (tn == 2):
            ahash = {'Baseline':0, 'Day0':1}
        if (tn == 3):
            atypes = ['CV', 'AV']
            ahash = {'Day0':1, 'Day21':0}
        self.initData(atype, atypes, ahash)

    def getMitchell2013(self, tn=1):
        self.prepareData("MACV104")
        time = self.h.getSurvName("c timepoint")
        atype = [re.sub("h.*", "", str(k)) for k in time]
        ahash = {'0':0, '18':3, '12':2, '6':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection code")
        atypes = ['C', 'I']
        ahash = {'BatSRBD':1, 'icSARS':1, 'Mock':0, 'dORF6':1, 'H1N1':1, 'mock':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c infection code")
            ahash = {'BatSRBD':1, 'icSARS':1, 'Mock':0, 'mock':0}
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['C' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName("c infection code")
            ahash = {'H1N1':1, 'Mock':0, 'mock':0}
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['C' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getWallihan2018(self, tn=1):
        self.prepareData("MACV105")
        atype = self.h.getSurvName('c viral organism')
        ahash = {'Coronavirus':1, 'RSV+Coronavirus':1,
                'Coronavirus+Bocavirus+Adenovirus':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c race')
        ahash = {'Other':3, 'Black or African American':1,
                 'White':0, 'Asian':2, 'Unknown':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c condition")
        atypes = ['C', 'I', 'CoV']
        ahash = {'Pneumonia':1, 'Healthy Control':0}
        if (tn == 2):
            atype = [atype[i] if rval[i] != 1 else 'CoV' for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = ['W', 'B', 'A']
            ahash = {0:0, 1:1, 2:2}
        self.initData(atype, atypes, ahash)

    def getPfaender2020(self, tn=1):
        self.prepareData("MACV106.2")
        atype = self.h.getSurvName("c conditional ly6e knock-out")
        ahash = {'wt':0, 'ko':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c tissue")
        ahash = {'spleen':0, 'liver':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c inoculation")
        atypes = ['PBS', 'MHV']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 and gval[i] == 0 
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if rval[i] == 0 and gval[i] == 1 
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if rval[i] == 1 and gval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if rval[i] == 1 and gval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getJones2019(self, tn=1):
        self.prepareData("MACV107")
        atype = self.h.getSurvName("c visit")
        ahash = {'AV':0, 'CV':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c src1")
        ahash = {'NMS':0, 'PBMC':1}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c virus positive at av (1=yes, 0=no, 9=not measured)")
        atypes = ['0', '1']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c human coronavirus at av (1=yes, 0=no, 9=not measured)")
            atype = [atype[i] if rval[i] == 0 and gval[i] == 0
                    else None for i in range(len(atype))]
        if (tn >= 3):
            atype = self.h.getSurvName("c visit")
            ahash = {'AV':1, 'CV':0}
            atypes = ['CV', 'AV']
        if (tn == 4):
            atype = [atype[i] if rval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if rval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBermejoMartin2010(self, tn=1):
        self.prepareData("MACV108")
        mtype = self.h.getSurvName("c mechanical ventilation")
        atype = self.h.getSurvName("c disease phase")
        mtype = [ " ".join([str(atype[i]), str(mtype[i])])
                for i in range(len(atype))]
        atypes = ['C', 'ENMV', 'LNMV', 'EMV', 'LMV']
        ahash = {'late period no':2, 'early period no':1,
                'early period yes':3, 'late period yes':4, 'control ':0}
        mval = [ahash[i] if i in ahash else None for i in mtype]
        ptype = self.h.getSurvName("c patient")
        ph = {'7':1, '8':1, '11':1, '17':1, '19':1}
        rval = [ph[i] if i in ph else None for i in ptype]
        atype = self.h.getSurvName("c disease phase");
        atypes = ['C', 'E', 'L']
        ahash = {'late period':2, 'early period':1, 'control':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c mechanical ventilation");
            atypes = ['NMV', 'MV']
            ahash = {'no':0, 'yes':1}
        if (tn == 3 or tn == 4):
            atype = mtype
            atypes = ['C', 'ENMV', 'LNMV', 'EMV', 'LMV']
            ahash = {'late period no':2, 'early period no':1,
                    'early period yes':3, 'late period yes':4, 'control ':0}
        if (tn == 4):
            atype = mtype
            atypes = ['C', 'ENMV', 'LNMV', 'EMV', 'LMV', 'LMVD']
            atype = [mtype[i] if mval[i] != 4 or rval[i] != 1 else "LMVD"
                    for i in range(len(atype))]
            ahash = {'late period no':2, 'early period no':1,
                    'early period yes':3, 'late period yes':4, 'control ':0}
        if (tn == 5):
            atype = mtype
            atypes = ['LMV', 'LMVD']
            atype = [mtype[i] if mval[i] != 4 or rval[i] != 1 else "LMVD"
                    for i in range(len(atype))]
            ahash = {'late period yes':0}
        if (tn == 6):
            atype = mtype
            atypes = ['C', 'A', 'D']
            atype = [mtype[i] if mval[i] != 4 or rval[i] != 1 else "D"
                    for i in range(len(atype))]
            ahash = {'late period no':1, 'early period no':1,
                    'early period yes':1, 'late period yes':1, 'control ':0}
        if (tn == 7):
            atype = mtype
            atypes = ['CV', 'AV']
            atype = [mtype[i] if mval[i] != 4 or rval[i] != 1 else "D"
                    for i in range(len(atype))]
            ahash = {'late period no':0, 'early period no':1,
                    'early period yes':1, 'late period yes':0}
        self.initData(atype, atypes, ahash)

    def getCameron2007(self, tn=1):
        self.prepareData("MACV109")
        atype = self.h.getSurvName("c Status");
        atypes = ['HC', 'C', 'Pre', 'Post']
        ahash = {'pre-pO2 nadir':2, 'post-pO2 nadir':3, 'healthy control':0,
                'convalescent':1}
        if (tn == 2):
            atypes = ['HC', 'I']
            ahash = {'pre-pO2 nadir':1, 'post-pO2 nadir':1, 'convalescent':1,
                    'healthy control':0}
        self.initData(atype, atypes, ahash)

    def getJosset2014(self, tn=1):
        self.prepareData("MACV110")
        atype = self.h.getSurvName("c virus");
        atypes = ['C', 'InfA', 'CoV'];
        ahash = {'MA15':2, 'MOCK':0, 'PR8':1}
        self.initData(atype, atypes, ahash)

    def getPrice2020(self, tn=1):
        self.prepareData("MACV111")
        mtype = self.h.getSurvName("c tissue")
        atype = self.h.getSurvName("c infection condtion")
        atype = [ " ".join([str(atype[i]), str(mtype[i])]) for i in
                range(len(atype))]
        atypes = ['S', 'SI', 'L', 'LI'];
        ahash = {'Mock Spleen':0, 'Mock Liver':2,
                'MA-EBOV infected Spleen':1, 'MA-EBOV infected Liver':3}
        if (tn == 2):
            atypes = ['C', 'I'];
            ahash = {'Mock Liver':0, 'MA-EBOV infected Liver':1}
        self.initData(atype, atypes, ahash)

    def getReynard2019(self, tn=1):
        self.prepareData("MACV112")
        atype = self.h.getSurvName("c group")
        atypes = ['HC', 'SR', 'VR', 'F'];
        ahash = {'Fatalities':3, 'Healthy controls':0,
                'Survivors in recovery phase':1, 'Viremic survivors':2}
        if (tn == 2):
            atypes = ['HC', 'R', 'I'];
            ahash = {'Fatalities':2, 'Healthy controls':0,
                    'Survivors in recovery phase':1, 'Viremic survivors':2}
        self.initData(atype, atypes, ahash)

    def getDunning2018(self, tn=1):
        self.prepareData("MACV113")
        atype = self.h.getSurvName("c t1severity")
        atypes = ['HC', '1', '2', '3'];
        ahash = {'HC':0, '1':1, '2':2, '3':3}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'I']
            ahash = {'1':1, '2':1, '3':1}
        if (tn == 3):
            atype = self.h.getSurvName("c ethnicity")
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'White':0, 'Other':3, 'Black':1, 'Asian':2}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBerdal2011(self, tn=1):
        self.prepareData("MACV114")
        atype = self.h.getSurvName("c control")
        expt = ["P" if k == "" else "" for k in atype]
        atype = self.h.getSurvName("c patient")
        ctrl = ["C" if k == "" else "" for k in atype]
        atype = [ str(expt[i]) + str(ctrl[i]) for i in range(len(atype)) ]
        atypes = ['C', 'P']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getXie2018(self, tn=1):
        self.prepareData("MACV115")
        atype = self.h.getSurvName("c disease interval")
        atypes = ['Con', 'Cr']
        ahash = {'convalescent':0, 'crisis':1}
        self.initData(atype, atypes, ahash)

    def getFriesenhagen2012(self, tn=1):
        self.prepareData("MACV116")
        atype = self.h.getSurvName("c desc")
        atype = [re.sub("Gene.*from ", "", str(k)) for k in atype]
        atype = [re.sub(" mac.*", "", str(k)) for k in atype]
        atypes = ['C', 'PR8', 'H5N1', 'FPV']
        ahash = {'FPV-infected':3, 'H5N1-infected':2, 'control':0, 'PR8-infected':1}
        self.initData(atype, atypes, ahash)

    def getGuan2018(self, tn=1):
        self.prepareData("MACV117")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(", .*", "", str(k)) for k in atype]
        atype = [ re.split(" ", str(k))[1] if len(re.split(" ", str(k))) > 1
                                else None for k in atype]
        ahash = {'2':2, 'control':0, '3':3, '8':8, '7':7, '4':4,
                '9':9, '6':6, '10':10}
        pval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c number of day post-infection")
        atype = [re.sub("NA", "0", str(k)) for k in atype]
        phash = {}
        for i in range(2, len(atype)):
            if pval[i] not in phash:
                phash[pval[i]] = []
            phash[pval[i]].append([i, int(atype[i])])
        before = []
        after = []
        for i in phash.keys():
            if i == 0:
                before += [k[0] for k in phash[i]]
                after += [k[0] for k in phash[i]]
            else:
                ll = sorted(phash[i], key = lambda k: k[1])
                before.append(ll[0][0])
                before.append(ll[1][0])
                after.append(ll[-1][0])
        beforehash = set(before)
        afterhash = set(after)
        ahash = {'2':0, '3':1, '8':1, '7':0, '4':1,
                '9':0, '6':0, '10':1}
        gender = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c subject category")
        atypes = ['C', 'P']
        ahash = {'Patient':1, 'Control':0}
        if (tn == 2):
            atype = [atype[i] if gender[i] == 0 or pval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            #Categories: 1) Control; 2) Mild - non-invasive ventilation (#4,#5);
            #3) Moderate- MV, but discharged within 2 mo (#1, 2, 7, 8);
            #4) Severe-  MV+prolonged hospitalization (#6 and 9);
            #5) Death after MV + ECMO = (#3 and 10)
            atype = pval
            atypes = ['C', 'Mi', 'Mo', 'S', 'D']
            ahash = {0:0, 4:1, 5:1, 1:2, 2:2, 7:2, 8:2, 6:3, 9:3, 3:4, 10:4}
            #atype = [atype[i] if gender[i] == 1 or pval[i] == 0
            #        else None for i in range(len(atype))]
        if (tn == 4):
            atype = pval
            atypes = ['C', 'Mi', 'MV', 'D']
            ahash = {0:0, 4:1, 5:1, 1:2, 2:2, 7:2, 8:2, 6:2, 9:2, 3:3, 10:3}
            atype = [atype[i] if i in beforehash
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = pval
            atypes = ['C', 'Mi', 'Mo', 'S', 'D']
            ahash = {0:0, 4:1, 5:1, 1:2, 2:2, 7:2, 8:2, 6:3, 9:3, 3:4, 10:4}
            atype = [atype[i] if i in afterhash
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKongchanagul2011(self, tn=1):
        self.prepareData("MACV118")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'H5N1']
        ahash = {'H5N1 influenza':1, 'normal':0}
        if (tn == 2):
            atype = self.h.getSurvName("c time of death")
            atypes = ['C', '6', '17']
            ahash = {'Day 17 of illness':2, 'Day 6 of illness':1, 'N/A':0}
        self.initData(atype, atypes, ahash)

    def getHo2013(self, tn=1):
        self.prepareData("MACV119")
        atype = self.h.getSurvName("c gender")
        atypes = ['F', 'M']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c age")
            atypes = ['C', '6', '17']
            ahash = {'Day 17 of illness':2, 'Day 6 of illness':1, 'N/A':0}
        self.initData(atype, atypes, ahash)

    def getKuparinen2013(self, tn=1):
        self.prepareData("MACV120")
        atype = self.h.getSurvName("c frailty index")
        atypes = ['nF', 'pF', 'F']
        ahash = {'non-frail':0, 'pre-frail':1, 'frail':2}
        if (tn == 2):
            atype = self.h.getSurvName("c age")
            aha = {'90':1, 'nonagenarian':1}
            age = ['O' if atype[i] in aha else 'Y' for i in range(len(atype))]
            atype = self.h.getSurvName("c Sex")
            ahash = {'Female':'F', 'female':'F', 'Male':'M', 'male':'M'}
            sex = [ahash[k] if k in ahash else None for k in atype]
            atype = [ str(age[i]) + str(sex[i]) for i in range(len(atype)) ]
            atypes = ['YF', 'YM', 'OF', 'OM']
            ahash = {}
        if (tn == 3):
            atype = self.h.getSurvName("c age")
            aha = {'90':1, 'nonagenarian':1}
            atype = ['O' if atype[i] in aha else 'Y' for i in range(len(atype))]
            atypes = ['Y', 'O']
            ahash = {}
        if (tn == 4):
            atype = self.h.getSurvName("c cmv serostatus")
            atypes = ['neg', 'pos']
            ahash = {'pos.':1, 'neg.':0}
        self.initData(atype, atypes, ahash)

    def getWagstaffe2020(self, tn=1):
        self.prepareData("MACV121")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("^._", "", str(k)) for k in atype]
        atypes = ['U-', 'U+', 'E-', 'E+']
        ahash = {'Med_CD14-':0, 'EBOV_CD14+':3, 'Med_CD14+':1, 'EBOV_CD14-':2}
        if (tn == 2):
            atypes = ['U', 'E']
            ahash = {'Med_CD14-':0, 'EBOV_CD14+':1, 'Med_CD14+':0, 'EBOV_CD14-':1}
        if (tn == 3):
            atypes = ['U', 'E']
            ahash = {'Med_CD14-':0, 'EBOV_CD14-':1}
        self.initData(atype, atypes, ahash)

    def getPeng2016I(self, tn=1):
        self.prepareData("MACV122")
        atype = self.h.getSurvName("c disease state")
        ahash = {'Chronic Obstructive Lung Disease':2,
                'Interstitial lung disease':1,
                'Control':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Sex")
        atypes = ['F', 'M']
        ahash = {'1-Male':1, '2-Female':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = ['C', 'ILD', 'COPD']
            ahash = {0:0, 1:1, 2:2}
        if (tn == 4):
            atype = self.h.getSurvName("c gold stage")
            atypes = ['R', 'Mi', 'Mo', 'S', 'VS']
            ahash = {'4-Very Severe COPD':4, '0-At Risk':0, '2-Moderate COPD':2,
                    '1-Mild COPD':1, '3-Severe COPD':3}
        if (tn == 5):
            atype = tval
            atypes = ['C', 'COPD']
            ahash = {0:0, 2:1}
        self.initData(atype, atypes, ahash)

    def getPeng2016II(self, tn=1):
        self.prepareData("MACV122.2")
        atype = self.h.getSurvName("c disease state")
        ahash = {'Chronic Obstructive Lung Disease':2,
                'Interstitial lung disease':1,
                'Control':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Sex")
        atypes = ['F', 'M']
        ahash = {'1-Male':1, '2-Female':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = ['C', 'ILD', 'COPD']
            ahash = {0:0, 1:1, 2:2}
        if (tn == 4):
            atype = self.h.getSurvName("c gold stage")
            atypes = ['R', 'Mi', 'Mo', 'S', 'VS']
            ahash = {'4-Very Severe COPD':4, '0-At Risk':0, '2-Moderate COPD':2,
                    '1-Mild COPD':1, '3-Severe COPD':3}
        if (tn == 5):
            atype = tval
            atypes = ['C', 'COPD']
            ahash = {0:0, 2:1}
        self.initData(atype, atypes, ahash)

    def getBosse2012(self, tn=1):
        self.prepareData("MACV123")
        atype = self.h.getSurvName("c src1")
        atypes = ['Lung']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMenachery2017(self, tn=1):
        self.prepareData("COV1.3")
        time1 = self.h.getSurvName("c time")
        time2 = self.h.getSurvName("c time-post-infection")
        time3 = self.h.getSurvName("c time-hrs post infection")
        time4 = self.h.getSurvName("c time point")
        atype = [ "".join([str(k) for k in
                               [time1[i], time2[i], time3[i], time4[i]]])
                                        for i in range(len(time1))]
        atype = [re.sub("[h ].*", "", k) for k in atype]
        ahash = {'00':'0'}
        tval = [ahash[i] if i in ahash else i for i in atype]
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(",.*", "", str(k)) for k in atype]
        ahash = {'Primary human fibroblasts':0,
                'Primary human airway epithelial cells':1,
                'Primary human microvascular endothelial cells':2,
                'Primary human dendritic cells':3}
        rval = [ahash[i] if i in ahash else None for i in atype]
        v1 = self.h.getSurvName("c virus")
        v2 = self.h.getSurvName("c virus infection")
        v3 = self.h.getSurvName("c infection")
        v4 = self.h.getSurvName("c infected with")
        atype = [ "".join([str(k) for k in [v1[i], v2[i], v3[i], v4[i]]])
                for i in range(len(atype))]
        atypes = ['C', 'I']
        ahash = { 'Mock':0, 'icMERS':1, 'RFP-MERS':1, 'mock':0, 'd4B-MERS':1,
                'MOCK':0, 'Mockulum':0, 'dNSP16-MERS':1, 'd3-5-MERS':1,
                'MERS-coronavirus (icMERS)':1}
        if (tn >= 2 and tn <= 5):
            atype = [atype[i] if rval[i] == (tn - 2) else None
                    for i in range(len(atype))]
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['C' if aval[i] == 1 and tval[i] == '0' else atype[i]
                    for i in range(len(atype))]
        if (tn == 6):
            atype = rval
            atypes = ['FI', 'AE', 'ME', 'DC']
            ahash = {0:0, 1:1, 2:2, 3:3}
        if (tn == 7):
            atype = self.h.getSurvName("c Series")
            atypes = ['FI', 'AE', 'ME', 'DC']
            ahash = {'GSE100496':0, 'GSE100504':1, 'GSE100509':2, 'GSE79172':3}
        if (tn == 8):
            ctype = self.h.getSurvName("c cell type")
            atypes = ['C', 'I']
            ahash = { 'Mock':0, 'mock':0,
                    'MOCK':0, 'Mockulum':0, 'EBOV-WT':1}
            atype = [atype[i] if ctype[i] == 'Immortalized Human Hepatocytes (IHH)'
                    else None for i in range(len(atype))]
        self.rval = rval
        self.tval = tval
        self.initData(atype, atypes, ahash)

    def getYoshikawa2010(self, tn=1):
        self.prepareData("COV2")
        atype = self.h.getSurvName("c time")
        ahash = {'48 hours post infection':3,
                '24 hours post infection':2,
                 '12 hours post infection':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'CoV', 'DoV']
        ahash = {'Mock-infected':0,
                'SARS-CoV-infected (MOI=0.1)':1,
                'DOHV-infected (MOI=0.1)':2}
        if (tn == 2):
            atypes = ['C', 'CoV']
            ahash = {'Mock-infected':0,
                    'SARS-CoV-infected (MOI=0.1)':1}
            atype = [atype[i] if tval[i] == 3 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKelvin2014(self, tn=1):
        self.prepareData("COV3")
        atype = self.h.getSurvName("c time point")
        ahash = {'Day 3':3, 'Day 2':2, 'Day 1':1, 'Day 5':5,
                'Day 0':0, 'Day 28':28, 'Day 14':14}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'SARS-CoV (TOR-2)':1, 'Uninfected':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] is not None and 
                    (tval[i] == 0 or tval[i] <= 3) else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDeDiego2011(self, tn=1):
        self.prepareData("COV4")
        atype = self.h.getSurvName("c hours post infection")
        ahash = {'15 hpi':15, '24 hpi':24, '65 hpi':65, '7 hpi':7, '0 hpi':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'SARS CoV':1, 'SARS CoV DeltaE':1, 'Mock infected':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 or tval[i] == 65 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSims2013(self, tn=1):
        self.prepareData("COV5")
        atype = self.h.getSurvName("c time")
        ahash = {'30h':30, '36h':36, '7h':7, '60h':60, '48h':48, '0h':0,
                '12h':12, '72h':72, '24h':24, '54h':54, '3h':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['U', 'CoV']
        ahash = {'mock infected':0,
                'SARS CoV Urbani infected':1,
                'SARS delta ORF6 infected':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] is not None and 
                    (tval[i] == 0 or tval[i] >= 64) else None
                    for i in range(len(atype))]
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['U' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] is not None and tval[i] > 40 else None
                    for i in range(len(atype))]
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['U' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        self.tval = tval
        self.initData(atype, atypes, ahash)

    def getKatze2012(self, tn=1):
        self.prepareData("COV6")
        atype = self.h.getSurvName("c sample collection time post virus infection")
        ahash = {'36h':36, '72h':72, '7h':7, '60h':60, '54h':54,
                '30h':30, '12h':12, '24h':24, '0h':0, '48h':48}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'SARS CoV Urbani infected':1,
                'SARS Bat SRBD infected':1,
                'mock infected':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] is not None and 
                    (tval[i] == 0 or tval[i] >= 18) else None
                    for i in range(len(atype))]
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['U' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getJosset2013(self, tn=1):
        self.prepareData("COV7")
        atype = self.h.getSurvName("c time point")
        ahash = {'18 hpi':18, '12 hpi':12, '3 hpi':3,
                '24 hpi':24, '7 hpi':7, '0 hpi':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infected with")
        atypes = ['U', 'CoV']
        ahash = {'Human Coronavirus EMC 2012 (HCoV-EMC)':1, 'Mock':0}
        if (tn == 2):
            atype = [atype[i] if (tval[i] == 12 and atype[i] == 'Mock') 
                    or tval[i] == 18 else None
                    for i in range(len(atype))]
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = ['U' if aval[i] == 1 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKatze2014(self, tn=1):
        self.prepareData("COV8")
        atype = self.h.getSurvName("c time")
        ahash = {'d7':7, 'd2':2, 'd4':4, 'd1':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'Mock':0, 'SARS MA15':1, 'SARS CoV':1, 'SARS BatSRBD mutant':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] >= 4 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getJimenezGuardeno2014(self, tn=1):
        self.prepareData("COV9")
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'Mock':0, 'SARS-CoV-wt':1, 'SARS-CoV-mutPBM':1}
        self.initData(atype, atypes, ahash)

    def getSelinger2014(self, tn=1):
        self.prepareData("COV10")
        atype = self.h.getSurvName("c timepoint")
        ahash = {'18h':18, '7h':7, '0h':0, '12h':12, '24h':24, '3h':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'Mock  Infected':0, 'LoCoV':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 18 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getTotura2015(self, tn=1):
        self.prepareData("COV11")
        atype = self.h.getSurvName("c time")
        ahash = {'4 dpi':4, '7 dpi':7, '2 dpi':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'CoV']
        ahash = {'MA15 virus':1, 'mockulum':0}
        self.initData(atype, atypes, ahash)

    def getFerris2017(self, tn=1):
        self.prepareData("COV12")
        atype = self.h.getSurvName("c src1")
        atypes = ['I']
        ahash = {'lung tissue, 4 days post SARS-CoV infection':0}
        self.initData(atype, atypes, ahash)

    def getMoodley2016(self, tn=1):
        self.prepareData("COV13")
        v1 = self.h.getSurvName("c jak inhinitor")
        v2 = self.h.getSurvName("c jak inhibitor")
        atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                        for i in range(len(v1))]
        atypes = ['U', 'T']
        ahash = {'IL2 only':2, 'PAN':1, 'JAK1/2':1, 'JAK1':1, 'Untreated':0, 'JAK3':1}
        if (tn == 2):
            ahash = {'PAN':1, 'JAK1/2':1, 'Untreated':0}
        self.initData(atype, atypes, ahash)

    def getPardanani2013(self, tn=1):
        self.prepareData("COV14")
        atype = self.h.getSurvName("c pre or post-treatment")
        atypes = ['U', 'T']
        ahash = {'post':1, 'pre':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c anemia response to treatment")
            atypes = ['R', 'NR']
            ahash = {'responder':0, 'Non-responder':1}
            atype = [atype[i] if aval[i] == 1 else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKuo2019(self, tn=1):
        self.prepareData("COV15")
        atype = self.h.getSurvName("c treatment")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['U', 'F', 'T']
        ahash = {'none':0, 'TNF':2, 'Fib':1}
        if (tn == 2):
            atype = self.h.getSurvName("c cultured with")
            atypes = ['U', 'F', 'T', 'TF']
            ahash = {'none (alone)':0,
                    'fibroblasts':1,
                    'tumor necrosis factor (TNF)':2,
                    'tumor necrosis factor (TNF) +fibroblasts':3}
        if (tn == 3):
            atype = self.h.getSurvName("c treatment")
            atypes = ['U', 'T', 'T+Q', 'T+T']
            ahash = {'none':0, 'TNF':1, 'TNF + Hydrox':2, 'TNF + tofa':3}
        if (tn == 4):
            atype = self.h.getSurvName("c Title")
            atypes = ['MT', 'MTaIL6', 'MTF', 'MTFaIL6']
            ahash = {"P2_MT_aIL6":1, "P2_MTF_aIL6":3, "P1_MTF_aIL6":3,
                    "P1_MT_aIL6":1, 'P1_MT':0, 'P1_MTF':2,
                    'P2_MT':0, 'P2_MTF':2, 'P3_MT':0, 'P3_MTF':2, 
                    'P4_MT':0, 'P4_MTF':2}
        if (tn == 5):
            atype = self.h.getSurvName("c Title")
            atypes = ['MT', 'MTFaIL6']
            ahash = {"P2_MTF_aIL6":1, "P1_MTF_aIL6":1,
                    'P1_MT':0, 'P2_MT':0, 'P3_MT':0, 'P4_MT':0}
        self.initData(atype, atypes, ahash)

    def getSmyth2016(self, tn=1):
        self.prepareData("COV16")
        atype = self.h.getSurvName("c treated with")
        atypes = ['U', 'Q', 'S', 'S+Q']
        ahash = {'none (untreated)':0,
                '20 \xc2\xb5M hydroxychloroquine (HCQ)':1,
                'group A streptococcus':2,
                'group A streptococcus and hydroxychloroquine':3}
        if (tn == 2):
            atypes = ['S', 'S+Q']
            ahash = {'group A streptococcus':0,
                    'group A streptococcus and hydroxychloroquine':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getWhyte2007(self, tn=1):
        self.prepareData("COV17")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub("rol, .*", "rol", str(k)) for k in atype]
        atype = [re.sub(".*h, ", "", str(k)) for k in atype]
        atypes = ['U', 'R']
        ahash = {'25 microM resveratrol':1, 'ethanol control':0}
        self.initData(atype, atypes, ahash)

    def getBaty2006(self, tn=1):
        self.prepareData("COV18")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("HC .. ", "", str(k)) for k in atype]
        atype = [re.sub("^. ", "", str(k)) for k in atype]
        atypes = ['C', 'W', 'A', 'G']
        ahash = {'12 wine':1, '12 alcohol':2, '12 grape.juice':3,
                '12 water':0}
        self.initData(atype, atypes, ahash)

    def getAuerbach2014(self, tn=1, tv=0, dg=''):
        self.prepareData("COV19")
        atype = self.h.getSurvName("c tissue")
        ahash = {'Liver':0, 'Kidney':1, 'Bone marrow':2, 'Intestine':3,
                'Brain':4, 'Heart':5, 'Spleen':6, 'Skeletal muscle':7,
                'Primary rat hepatocytes':8}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c time")
        ahash = { '0 d':0, '0.25 d':0.25, '0.67 d':0.67, '1 d':1,
                '3 d':3, '4 d':4, '5 d':5, '6 d':6, '7 d':7,
                '14 d':14, '28 d':28, '30 d':30, '91 d':91}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = ['C' if dval[i] is None or dval[i] <= 3
                else 'T' for i in range(len(atype))]
        atypes = ['C', 'T']
        if (tn == 2):
            atype = self.h.getSurvName("c compound")
            atypes = ['C', 'D']
            ahash = {'':0, 'Captopril':1}
            atype = [atype[i] if tval[i] == 1 else None for i in range(len(atype))]
        if (tn == 3):
            ahash = {'Benazepril':1, 'Captopril':1, 'Enalapril':1,
                    'Quinapril':1, 'Ramipril':1, 'Lisinopril':1, 'Flutamide':1}
            atype = self.h.getSurvName("c compound")
            ahash = {'':0, 'Captopril':1}
            pval = [ahash[i] if i in ahash else None for i in atype]
            atype = dval
            atype = [0 if pval[i] == 0 else dval[i]
                    for i in range(len(atype))]
            if (dg == ''):
                atype = [atype[i] if pval[i] is not None and tval[i] == 1 else None
                        for i in range(len(atype))]
            else:
                atype = [atype[i] if pval[i] is not None and tval[i] == 1 and
                        (dval[i] == 0 or dval[i] == dg)
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        if (tn == 4):
            atype = self.h.getSurvName("c compound")
            ahash = {'':0, 'Chlorpromazine':1}
            pval = [ahash[i] if i in ahash else None for i in atype]
            atype = dval
            atype = [0 if pval[i] == 0 else dval[i]
                    for i in range(len(atype))]
            if (dg == ''):
                atype = [atype[i] if pval[i] is not None and tval[i] == 0 
                        else None for i in range(len(atype))]
            else:
                atype = [atype[i] if pval[i] is not None and tval[i] == 0 and
                        (dval[i] == 0 or dval[i] == dg)
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            #atypes = [0, 0.25]
            ahash = {}
        if (tn == 5):
            atype = self.h.getSurvName("c compound")
            atypes = ['C', 'D']
            ahash = {'':0, dg:1}
            atype = [atype[i] if tval[i] == tv else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getRuiz2016(self, tn=1):
        self.prepareData("COV20")
        atype = self.h.getSurvName("c treatment")
        atypes = ['U', 'O', 'O+C']
        ahash = {'basal condition, untreated':0,
                'combined treatment, oxaliplatin plus curcumin':2,
                'single treatment, oxaliplatin':1}
        self.initData(atype, atypes, ahash)

    def getMeja2008(self, tn=1):
        self.prepareData("COV21")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*cells, ", "", str(k)) for k in atype]
        atypes = ['U', 'C', 'R', 'R+C']
        ahash = {'untreated, 18h':0,
                'ROS exposed, 18h':2,
                'ROS exposed, 4h':2,
                'ROS exposed, 1uM curcumin treated, 4h':3,
                '1uM curcumin treated, 18h':1,
                'untreated, 4h':0,
                'ROS exposed, 1uM curcumin treated, 18h':3,
                '1uM curcumin treated, 4h':1}
        if (tn == 2):
            atypes = ['U', 'C']
            ahash = {'untreated, 18h':0,
                    '1uM curcumin treated, 18h':1,
                    'untreated, 4h':0,
                    '1uM curcumin treated, 4h':1}
        self.initData(atype, atypes, ahash)

    def getGarland2019(self, tn=1):
        self.prepareData("COV22")
        v1 = self.h.getSurvName("c timepoint")
        v2 = self.h.getSurvName("c dosing regimen")
        atype = [ " ".join([str(k) for k in [v1[i], v2[i]]])
                                for i in range(len(v1))]
        atypes = ['B', '12', '13']
        ahash = {'13 weeks intermittent':2,
                '12 weeks continuous':1,
                'baseline continuous':0,
                'baseline intermittent':0,
                '12 weeks intermittent':1,
                '13 weeks continuous':2}
        self.initData(atype, atypes, ahash)

    def getGuo2016(self, tn=1):
        self.prepareData("COV23")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'T']
        ahash = {'Aspirin treated for 48 hours':1, 'DMSO treated for 48 hours':0}
        self.initData(atype, atypes, ahash)

    def getPavuluri2014(self, tn=1):
        self.prepareData("COV24")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'T']
        ahash = {'Treated with 2.0 mM Aspirin':1, 'Untreated with Aspirin':0}
        self.initData(atype, atypes, ahash)

    def getFallahi2013(self, tn=1):
        self.prepareData("COV25")
        atype = self.h.getSurvName("c group")
        atypes = ['N', 'S', 'R']
        ahash = {'aspirin sensitive':1, 'high normal':0, 'aspirin resistant':2}
        if (tn == 2):
            atypes = ['N', 'A']
            ahash = {'aspirin sensitive':1, 'high normal':0}
        self.initData(atype, atypes, ahash)

    def getLewis2020(self, tn=1):
        self.prepareData("PLP113")
        atype = self.h.getSurvName("c Sex")
        atypes = ['F', 'M']
        ahash = {'Male':1, 'Female':0}
        self.initData(atype, atypes, ahash)

    def getLv2017(self, tn=1):
        self.prepareData("COV26")
        atype = self.h.getSurvName("c group")
        ahash = {'treatment':1, 'control':0}
        atypes = ['C', 'T']
        if (tn == 2):
            atype = self.h.getSurvName("c perturbagen")
            ahash = {'DMSO':0, 'Resveratrol':1}
            atypes = ['C', 'R']
        if (tn == 3):
            atype = self.h.getSurvName("c perturbagen")
            ahash = {'DMSO':0, 'Hyodeoxycholic acid':1,
                    'Ursodeoxycholic acid':1, 'Deoxycholic acid':1,
                    'Chenodeoxycholic acid':1, 'Artemisinin':3, 'Resveratrol':2}
            atypes = ['C', 'B', 'R', 'A']
        if (tn == 4):
            atype = self.h.getSurvName("c perturbagen")
            ahash = {'DMSO':0, 'Hyodeoxycholic acid':1,
                    'Ursodeoxycholic acid':1, 'Deoxycholic acid':1,
                    'Chenodeoxycholic acid':1}
            atypes = ['C', 'B']
        if (tn == 5):
            atype = self.h.getSurvName("c perturbagen")
            ahash = {'DMSO':0, 'Artemisinin':1}
            atypes = ['C', 'A']
        self.initData(atype, atypes, ahash)

    def getCMAP(self, tn=1):
        self.prepareData("COV27")
        atype = self.h.getSurvName("c type")
        ahash = {'treatment':1, 'control':0}
        atypes = ['C', 'T']
        if (tn == 2):
            atype = self.h.getSurvName("c name")
            ahash = {'null':0, 'chlorpromazine':1}
            atypes = ['C', 'CPZ']
        if (tn == 3):
            atype = self.h.getSurvName("c name")
            ahash = {'null':0, 'resveratrol':1}
            atypes = ['C', 'R']
        if (tn == 4):
            atype = self.h.getSurvName("c name")
            ahash = {'null':0, 'sirolimus':1}
            atypes = ['C', 'S']
        self.initData(atype, atypes, ahash)

    def getWoo2015(self, tn=1):
        self.prepareData("COV28")
        atype = self.h.getSurvName("c src1")
        atypes = ['OCI-Ly3', 'OCI-Ly7', 'U2932', 'HeLa']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c compound treated")
            ahash = {'DMSO':0, 'ESTRADIOL':1, 'Docetaxel': 2}
            atypes = ['C', 'E', 'D']
        self.initData(atype, atypes, ahash)

    def getNiculescu2019(self, tn=1):
        self.prepareData("COV29")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['None', 'BP', 'SZA', 'MDD', 'SZ', 'PSYCH', 'PTSD', 'MOOD']
        ahash = {'':0}
        self.initData(atype, atypes, ahash)

    def getBush2017(self, tn=1):
        self.prepareData("COV30")
        atype = self.h.getSurvName("c src1")
        atypes = ['B', 'L', 'U']
        ahash = {'BT20 cell line':0, 'LnCAP cell line':1, 'U87 cell line':2}
        if (tn == 2):
            atype = self.h.getSurvName("c drug")
            atypes = ['C', 'T']
            ahash = {'DMSO':0, 'untreated':0, 'Temsirolimus':1}
        self.initData(atype, atypes, ahash)

    def getPlauth2015(self, tn=1):
        self.prepareData("COV31")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'R']
        ahash = {'vehicle':0, '16 hours 50 \xc2\xb5M resveratrol':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getTomeCarneiro2013(self, tn=1):
        self.prepareData("COV32")
        atype = self.h.getSurvName("c time point")
        ahash = {'12 months':12, '6 months':6, 'day 0 (basal)':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c dietary group")
        atypes = ['C', 'G', 'R']
        ahash = {'placebo (A)':0,
                'grape extract (B)':1,
                'resveratrol-enriched grape extract (C)':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = ['C' if aval[i] != 0 and tval[i] == 0 else atype[i]
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 12 else None
                    for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if tval[i] == 6 else None
                    for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if tval[i] == 12 else None
                    for i in range(len(atype))]
            atypes = ['C', 'R']
            ahash = {'placebo (A)':0,
                    'resveratrol-enriched grape extract (C)':1}
        if (tn == 6):
            atype = [atype[i] if tval[i] == 12 else None
                    for i in range(len(atype))]
            atypes = ['C', 'G']
            ahash = {'placebo (A)':0,
                    'grape extract (B)':1}
        self.initData(atype, atypes, ahash)

    def getUribe2011(self, tn=1):
        self.prepareData("COV33")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'R1', 'R2']
        ahash = {'0.03% ethanol (control)':0,
                'resveratrol_150mM':1,
                'resveratrol_250mM':2}
        self.initData(atype, atypes, ahash)

    def getEuba2015(self, tn=1):
        self.prepareData("COV34")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'I']
        ahash = {'Haemophilus influenzae strain NTHi375':1, 'none':0}
        self.initData(atype, atypes, ahash)

    def getDavis2019(self, tn=1):
        self.prepareData("COV35")
        atype = self.h.getSurvName("c tissue")
        atypes = ['L', 'BM', 'S', 'K', 'H']
        ahash = {'Liver':0, 'Bone Marrow':1, 'Skin':2, 'Kidney':3, 'Heart':4}
        self.initData(atype, atypes, ahash)

    def getPodtelezhnikov2020(self, tn=1):
        self.prepareData("COV36")
        atype = self.h.getSurvName("c Sex")
        atypes = ['F', 'M']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c treatment")
            atypes = ['C', 'X', 'C', 'A', 'Q', 'CP']
            ahash = {'-control-':0, 'amoxicillin':1, 'caffeine':2,
                    'acetaminophen':3, 'quinidine':4, 'captopril':5}
        if (tn == 3):
            atype = self.h.getSurvName("c treatment")
            atypes = ['C', 'CP']
            ahash = {'-control-':0, 'captopril':1}
        if (tn == 4):
            atype = self.h.getSurvName("c treatment")
            atypes = ['C', 'F']
            ahash = {'-control-':0, 'flutamide':1}
        if (tn == 5):
            atype = self.h.getSurvName("c treatment")
            atypes = ['C', 'Q']
            ahash = {'-control-':0, 'quinidine':1}
        self.initData(atype, atypes, ahash)

    def getMonks2018(self, tn=1):
        self.prepareData("GL15")
        atype = self.h.getSurvName("c Title")
        atype = [ re.split("_", str(k))[2] if len(re.split("_", str(k))) > 2
                else None for k in atype]
        conc = [int(re.sub("nM", "", k)) if k is not None else None for k in atype]
        atype = self.h.getSurvName("c tissue")
        ahash = {'Renal':0, 'CNS':1, 'Melanoma':2, 'Lung':3, 'Breast':4,
                'Ovarian':5, 'Colon':6, 'Leukemia':7, 'Prostate':8}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Time")
        ahash = {'2h':2, '6h':6, '24h':24}
        mval = [ahash[i] if i in ahash else None for i in atype]
        drug = self.h.getSurvName("c Drug")
        atype = self.h.getSurvName("c Time")
        atypes = ['2h', '6h', '24h']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c Drug")
            atypes = ['S', 'C', 'SN']
            ahash = {'sunitinib':2, 'sirolimus':0, 'cisplatin':1}
            atype = [atype[i] if tval[i] == 3 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 3 else None
                    for i in range(len(atype))]
        if (tn == 4):
            atype = self.h.getSurvName("c Drug")
            atypes = ['sunitinib', 'dasatinib', 'sirolimus',
                    'lapatinib', 'doxorubicin', 'sorafenib',
                    'bortezomib', 'cisplatin', 'erlotinib']
            atype = [atype[i] if tval[i] == 3 and conc[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 5):
            atype = conc
            atype = [atype[i] if tval[i] == 6 and drug[i] == 'sirolimus' and
                    mval[i] == 24 else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([k for k in atype if k is not None]))
        self.conc = conc
        self.mval = mval
        self.tval = tval
        self.drug = drug
        self.initData(atype, atypes, ahash)

    def getKornakiewicz2018(self, tn=1):
        self.prepareData("COV37")
        atype = self.h.getSurvName("c src1")
        atypes = ['SC', 'pSC']
        ahash = {'Human Kidney Cancer Stem Cells treated with everolimus':0,
                'Parental treated with everolimus':1}
        self.initData(atype, atypes, ahash)

    def getYu2018(self, tn=1):
        self.prepareData("COV38")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'R']
        ahash = {'control':0, 'rapamycin':1}
        self.initData(atype, atypes, ahash)

    def getSabine2010(self, tn=1):
        self.prepareData("COV39")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*\.P", "P", str(k)) for k in atype]
        atype = [re.sub("_.*", "", str(k)) for k in atype]
        atypes = ['Pre', 'Post']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getDreschers2019(self, tn=1):
        self.prepareData("COV40")
        atype = self.h.getSurvName("c agent")
        atypes = ['1', '2', '3', '4', '5']
        ahash = {'IL10':0, 'IFN-\xce\xb3':1, 'M(IFN-gamma)':2,
                'M (IL-10)':3, 'M(IFN-gamma) + Rapamycin':4,
                'M (IL-10) + Rapamycin':5}
        if (tn == 2):
            atypes = ['C', 'C+R']
            ahash = {'M (IL-10)':0, 'M (IL-10) + Rapamycin':1}
        if (tn == 3):
            atypes = ['C', 'C+R']
            ahash = {'M(IFN-gamma)':0, 'M(IFN-gamma) + Rapamycin':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getCardoso2016(self, tn=1):
        self.prepareData("COV41")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'S', 'S+D']
        ahash = {'sunitinib and docetaxel combined treatment':2,
                'prior to any treatment':0,
                'sunitinib':1}
        if (tn == 2):
            atype = self.h.getSurvName("c neosu_response")
            atypes = ['good', 'bad']
            ahash = {}
        if (tn == 3):
            atype = self.h.getSurvName("c docetaxel_response")
            atypes = ['good', 'bad']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getKrishnaSubramanian2012(self, tn=1):
        self.prepareData("COV42")
        atype = self.h.getSurvName("c Title")
        atypes = ['N', 'T']
        ahash = {'N1+2':0, 'T3+4':1, 'N3+4':0, 'N5+6':0, 'T5+6':1, 'T1+2':1}
        self.initData(atype, atypes, ahash)

    def getAuerbach2020(self, tn=1, dg=""):
        self.prepareData("COV43")
        atype = self.h.getSurvName("c cell type")
        atypes = ['HepaRG cells']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c chemical")
            atypes = ['D', 'S', 'A', 'C', 'B']
            ahash = {'DMSO':0,'aspirin':2,'sucrose':1,'caffeine':3, 'CDCA':4}
        if (tn == 3):
            drug = self.h.getSurvName("c chemical")
            dose = self.h.getSurvName("c dose")
            dose = [re.sub(" nM", "", str(k)) for k in dose]
            atype = [dose[i] if drug[i] == dg or drug[i] == 'DMSO' else None
                    for i in range(len(drug))]
            atype = ['0' if drug[i] == 'DMSO' else atype[i]
                    for i in range(len(drug))]
            atype = [float(atype[i]) if atype[i] is not None else None
                    for i in range(len(drug))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
        self.initData(atype, atypes, ahash)

    def getWigger2019(self, tn=1):
        self.prepareData("COV44")
        atype = self.h.getSurvName("c time point")
        ahash = {'24h':24, '4h':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c ligand treatment")
        atypes = ['Control', 'CDCA', 'FXR-L', 'PPARa', 'LXR-L']
        ahash = {}
        if (tn == 2):
            atypes = ['Control', 'CDCA', 'FXR-L']
            atype = [atype[i] if tval[i] == 24 else None
                    for i in range(len(atype))]
        if (tn == 3):
            drug = self.h.getSurvName("c ligand treatment")
            atype = self.h.getSurvName("c time point")
            atypes = ['4h', '24h']
            ahash = {}
            atype = [atype[i] if drug[i] == 'CDCA' else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getIjssennagger2016(self, tn=1):
        self.prepareData("COV45")
        atype = self.h.getSurvName("c treatment")
        atypes = ['V', 'OCA']
        ahash = {'vehicle (0.1% DMSO)':0, '1 uM OCA (INT-747)':1}
        self.initData(atype, atypes, ahash)

    def getIjssennagger2016Mm(self, tn=1):
        self.prepareData("COV45.2")
        atype = self.h.getSurvName("c genotype/variation")
        ahash = {'FXR KO':1, 'wild type':0}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['V', 'OCA']
        ahash = {'OCA (INT-747) (10 mg/kg/day, dissolved in 1% methyl cellulose':1,
                'vehicle (1% methyl cellulose)':0}
        if (tn == 2):
            atype = [atype[i] if gval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if gval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getShen2020(self, tn=1):
        self.prepareData("COV46")
        atype = self.h.getSurvName("c disase state")
        atypes = ['H', 'AIDS']
        ahash = {'AIDS':1, 'healthy':0}
        self.initData(atype, atypes, ahash)

    def getLiao2020(self, tn=1):
        self.prepareData("COV47")
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'Zika']
        ahash = {'control':0, 'Zika virus':1}
        self.initData(atype, atypes, ahash)

    def getSuthar2019(self, tn=1):
        self.prepareData("COV48")
        atype = self.h.getSurvName("c Title")
        time = [ re.split("_", str(k))[2] if len(re.split("_", str(k))) > 2
                else None for k in atype]
        ahash = {'12H':12, '24H':24}
        tval = [ahash[i] if i in ahash else None for i in time]
        atype = [ re.split("_", str(k))[1] if len(re.split("_", str(k))) > 1
                else None for k in atype]
        atypes = ['Mock', 'WNV', 'RIG-I', 'MDA5', 'IFNb']
        ahash = {}
        if (tn == 2):
            atypes = ['Mock', 'WNV']
            atype = [None if atype[i] == 'WNV' and tval[i] == 12 else atype[i]
                    for i in range(len(atype))]
        if (tn == 3):
            atypes = ['Mock', 'IFNb']
        self.initData(atype, atypes, ahash)

    def getScott2019(self, tn=1):
        self.prepareData("COV49")
        atype = self.h.getSurvName("c culture condition")
        atypes = ['None', 'EGF', 'RAFT']
        ahash = {'':0, 'mouse EGF':1, '3D raft culture':2}
        if (tn == 2):
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = self.h.getSurvName("c infection")
            atype = [atype[i] if aval[i] == 1 else None
                    for i in range(len(atype))]
            atypes = ['C', 'HPV']
            ahash = {'ECM only':0,
                    '7d ECM only':0,
                    '7dpi with HPV16':1,
                    'transfected/immortalized with HPV16 genome':1,
                    '27-33 d ECM only':0,
                    '27-33 dpi with HPV16 WT':1,
                    '36-55 dpi with HPV16 WT':1,
                    '43-63 dpi with HPV16 WT':1,
                    'uninfected control':0,
                    'HPV16 WT':1}
        self.initData(atype, atypes, ahash)

    def getvanTol2020(self, tn=1):
        self.prepareData("COV50")
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'WNV']
        ahash = {'Mock with PBS':0, 'WNV at MOI = 5':1}
        self.initData(atype, atypes, ahash)

    def getDeutschman2019(self, tn=1):
        self.prepareData("COV51")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'Non':0, 'CAP-D3':1, 'CAP-H2':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['DMSO', 'VitK']
        ahash = {'DMSO':0, '50uM Menadione':1}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDeferme2013(self, tn=1):
        self.prepareData("COV52")
        atype = self.h.getSurvName("c time")
        tval = [float(atype[i]) if i > 1 else None for i in range(len(atype))]
        atype = self.h.getSurvName("c compound, dose")
        atypes = ['C', 'Men', 'H2O2', 'TBH']
        ahash = {'100\xc2\xb5M Men':1,
                '50\xc2\xb5M H2O2':2,
                'Control':0,
                '200\xc2\xb5M TBH':3}
        if (tn == 2):
            atypes = ['C', 'VitK']
            ahash = {'100\xc2\xb5M Men':1,
                    'Control':0}
            atype = [atype[i] if tval[i] == 8 else None
                    for i in range(len(atype))]
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getBell2017(self, tn=1):
        self.prepareData("COV53")
        atype = self.h.getSurvName("c treatment time")
        ahash = {'48h':0, '7d':1, '14d':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'CPZ', 'Am', 'Af']
        ahash = {'DMSO':0, 'Chlorpromazine':1, 'Amiodarone':2, 'Aflatoxin B1':3}
        if (tn == 2):
            atypes = ['C', 'CPZ']
            ahash = {'DMSO':0, 'Chlorpromazine':1}
            atype = [atype[i] if tval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDeAbrew2015(self, tn=1):
        self.prepareData("COV54")
        drug = self.h.getSurvName("c agent")
        drugs = hu.uniq(drug[2:])
        time = self.h.getSurvName("c time")
        conc = self.h.getSurvName("c concentration")
        comb = [str(time[i]) + " " + str(conc[i]) for i in range(len(time))]
        atype = self.h.getSurvName("c time")
        ahash = {'24 hours':24, '48 hours':48}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['24', '48']
        ahash = {'24 hours':0, '48 hours':1}
        if (tn == 2):
            atype = [comb[i] if drug[i] == 'Chlorpromazine HCl' or
                    drug[i] == 'DMSO' or drug[i] == 'Water'
                    else None for i in range(len(atype))]
            atype = ['0 ' + drug[i] + ' ' + comb[i] if 
                    drug[i] == 'DMSO' or drug[i] == 'Water'
                    else atype[i] for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            atypes = ['C', 'CPZ']
            ahash = {'0 DMSO 24 hours 1%':0, '48 hours 0.8 uM':1}
        self.initData(atype, atypes, ahash)

    def getVandenHof2014(self, tn=1):
        self.prepareData("COV55")
        atype = self.h.getSurvName("c treatment time")
        ahash = {'24h':24}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['24']
        ahash = {'24h':0}
        if (tn == 2):
            atype = self.h.getSurvName("c compound, dose")
            atypes = ['C', 'CPZ']
            ahash = {'DMSO, 0.5 %':0, 'CLP, 20 \xc2\xb5M':1}
            atype = [atype[i] if tval[i] == 24 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName("c compound, dose")
            atypes = ['C', 'CP']
            ahash = {'DMSO, 0.5 %':0, 'CP, 2000 \xc2\xb5M':1}
            atype = [atype[i] if tval[i] == 24 else None
                    for i in range(len(atype))]
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getVitins2017(self, tn=1):
        self.prepareData("COV56")
        atype = self.h.getSurvName("c time point")
        ahash = {'25 days':25}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C1', 'C2', 'CPZ', 'EE2']
        ahash = {'EE2':3, 'CPZ':2, 'CPZ vehicle (PBS)':0,
                'EE2 vehicle (sunflower oil)':1}
        if (tn == 2):
            atype = self.h.getSurvName("c treatment")
            atypes = ['C', 'CPZ']
            ahash = {'CPZ vehicle (PBS)':0, 'CPZ':1}
            atype = [atype[i] if tval[i] == 25 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBroin2016(self, tn=1):
        self.prepareData("COV57")
        atype = self.h.getSurvName("c treatment")
        atypes = ['control', 'ACEI', 'ARB']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMatsuuraHachiya2015mm(self, tn=1):
        self.prepareData("COV58")
        atype = self.h.getSurvName("c treatment")
        atype = [re.sub("l .*", "l", str(k)) for k in atype]
        atypes = ['control', 'ACEI']
        ahash = {'applied 30% ethanol':0, 'applied 1% enalapril':1}
        self.initData(atype, atypes, ahash)

    def getAbdAlla2010mm(self, tn=1):
        self.prepareData("COV59")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*from ", "", str(k)) for k in atype]
        atypes = ['C', 'Ath', 'Ath+ACEI']
        ahash = {'non-transgenic C57BL/6J control mice':0,
                'captopril-treated APOE-deficient mice':2,
                 'atherosclerotic APOE-deficient mice':1}
        self.initData(atype, atypes, ahash)

    def getEun2013rat(self, tn=1):
        self.prepareData("COV60")
        atype = self.h.getSurvName("c treatment")
        time = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'7':7, '2':2, '3':3, '10':10}
        tval = [ahash[i] if i in ahash else None for i in time]
        atype = [re.sub(".*after ", "", str(k)) for k in atype]
        atype = [re.sub(" treatment ", "", str(k)) for k in atype]
        atype = [re.sub(" dose", "", str(k)) for k in atype]
        group = [re.sub(".*from ", "", str(k)) for k in atype]
        drug = [re.sub("(.*)\\((.*)\\)", "\\1", str(k)) for k in group]
        dose = [re.sub(".*\\((.*)\\)", "\\1", str(k)) for k in group]
        ahash = {'high':3, 'middle':2, 'low':1, 'corn oil':0}
        dval = [ahash[i] if i in ahash else None for i in dose]
        atype = [time[i] + " " + dose[i] for i in range(len(atype))]
        atypes = sorted(hu.uniq(atype[2:]), reverse=1)
        atype = group
        atypes = ['C', 'T']
        ahash = {'RAN(high)':1, 'PZA(middle)':1, 'RAN(low)':1, 'CPZ(low)':1,
                'CPZ(high)':1, 'CBZ(high)':1, 'RAN(middle)':1, 'PZA(low)':1,
                'CBZ(middle)':1, 'PZA(high)':1, 'CPZ(middle)':1,
                'vehicle (corn oil)':0, 'CBZ(low)':1}
        self.initData(atype, atypes, ahash)

    def getDelVecchio2014(self, tn=1):
        self.prepareData("COV61")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'VitK', 'PERKi']
        ahash = {'PERKi':2, 'Menadione':1, 'DMSO':0}
        self.initData(atype, atypes, ahash)

    def getBriede2010I(self, tn=1):
        self.prepareData("COV62")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*FD .*", "B", str(k)) for k in atype]
        group = [re.sub(".* bi.*", "A", str(k)) for k in atype]
        atype = self.h.getSurvName("c Title")
        drug = [re.sub("T.*", "", str(k)) for k in atype]
        atype = self.h.getSurvName("c time")
        atype = [re.sub(" h", "", str(k)) for k in atype]
        time = [float(atype[i]) if i > 1 else None for i in range(len(atype))]
        atypes = sorted(hu.uniq(time[2:]))
        atype = time
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if drug[i] == 'Menadione'  and group[i] == 'B' 
                    else None for i in range(len(atype))]
            atypes = ['C', 'VitK', 'T2']
            ahash = {0.08:0, 16.0:2, 0.5:0, 0.25:0, 4.0:1, 
                    8.0:2, 2.0:1, 24.0:2, 1.0:1}
        if (tn == 3):
            atype = [atype[i] if drug[i] == 'H2O2' else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBriede2010II(self, tn=1):
        self.prepareData("COV63")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*FD .*", "B", str(k)) for k in atype]
        group = [re.sub(".* bi.*", "A", str(k)) for k in atype]
        atype = self.h.getSurvName("c Title")
        drug = [re.sub("T.*", "", str(k)) for k in atype]
        atype = self.h.getSurvName("c time")
        atype = [re.sub(" h", "", str(k)) for k in atype]
        time = [float(atype[i]) if i > 1 else None for i in range(len(atype))]
        atypes = sorted(hu.uniq(time[2:]))
        atype = time
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if drug[i] == 'Menadione'  and group[i] == 'B' 
                    else None for i in range(len(atype))]
            atypes = ['C', 'VitK', 'T2']
            ahash = {0.08:0, 16.0:2, 0.5:0, 0.25:0, 4.0:1, 
                    8.0:2, 2.0:1, 24.0:2, 1.0:1}
        if (tn == 3):
            atype = [atype[i] if drug[i] == 'H2O2' else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getYau2008(self, tn=1):
        self.prepareData("COV64")
        atype = self.h.getSurvName("c src1")
        atypes = ['U', 'M', 'H', 'D', 'E', 'i']
        ahash = {'MCF7, E2 deprivation, 72hr':4,
                'MCF7, anti-ERa siRNA treatment, 72hr':5,
                'MCF7, 0.5mM H2O2, 8hr':2,
                'MCF7, 275mM diamide, 8hr':3,
                'MCF7,untreated':0,
                'MCF7, 10mM menadione, 8hr':1}
        self.initData(atype, atypes, ahash)

    def getGusenleitner2014(self, tn=1):
        self.prepareData("COV65")
        tissue = self.h.getSurvName("c tissue")
        time = self.h.getSurvName("c time")
        conc = self.h.getSurvName("c dose")
        comb = [str(time[i]) + " " + str(conc[i]) for i in range(len(time))]
        drug = self.h.getSurvName("c compound")
        atype = self.h.getSurvName("c tissue")
        atypes = ['K', 'L', 'H', 'PH', 'TM']
        ahash = {'Kidney':0,
                'Liver':1,
                'Heart':2,
                'Primary rat hepatocytes':3,
                'Thigh muscle':4}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [drug[i] if aval[i] == 3 and (drug[i] == 'Chloroquine'
                    or drug[i] == '') else None
                    for i in range(len(drug))]
            atypes = ['Neg', 'Q']
            ahash = {'':0, 'Chloroquine': 1}
        if (tn == 3):
            atype = [drug[i] if aval[i] == 3 and (drug[i] == 'Chlorpromazine'
                    or drug[i] == '') else None
                    for i in range(len(drug))]
            atypes = ['Neg', 'CPZ']
            ahash = {'':0, 'Chlorpromazine': 1}
        self.initData(atype, atypes, ahash)

    def getReadhead2018I(self, tn=1):
        self.prepareData("COV66")
        atype = self.h.getSurvName("c treatment")
        atypes = ['DMSO', 'LOX', 'MPB']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getReadhead2018II(self, tn=1):
        self.prepareData("COV67")
        drug = self.h.getSurvName("c perturbagen")
        atype = self.h.getSurvName("c perturbation type")
        ahash = {'vehicle':0, 'test':1, 'poscon':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c src1")
        atypes = ['Ccontrol', 'Cancer', 'SZ']
        ahash = {'hiPSC_control':0, 'cancerCells':1, 'hiPSC_SZ':2}
        if (tn == 2):
            atype = [drug[i] if drug[i] == 'hydroquinine'
                    or tval[i] == 0 or tval[i] == 2 else None
                    for i in range(len(drug))]
            for i in range(len(atype)):
                if (tval[i] == 0):
                    atype[i] = 'Neg'
                if (tval[i] == 2):
                    atype[i] = 'Pos'
            atypes = ['Neg', 'Q']
            ahash = {'hydroquinine': 1}
        if (tn == 3):
            atype = [drug[i] if drug[i] == 'Chlorpromazine'
                    or tval[i] == 0 or tval[i] == 2 else None
                    for i in range(len(drug))]
            for i in range(len(atype)):
                if (tval[i] == 0):
                    atype[i] = 'Neg'
                if (tval[i] == 2):
                    atype[i] = 'Pos'
            atypes = ['Neg', 'CPZ']
            ahash = {'Chlorpromazine': 1}
        self.initData(atype, atypes, ahash)

    def getDeGottardi2016(self, tn=1):
        self.prepareData("COV73")
        atype = self.h.getSurvName("c treatment")
        ahash = {'IgG1 Control Ab':0, 'Anti-IL15 Ab':1}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c timepoint")
        atypes = ['Pre', 'Post']
        ahash = {'Post Treatment':1, 'Pre Treatment':0, 'Prost Treatment':1}
        if (tn == 2):
            atype = [atype[i] if dval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDeGottardi2016Hs(self, tn=1):
        self.prepareData("COV73.2")
        atype = self.h.getSurvName("c treatment")
        ahash = {'IgG1 Control Ab':0, 'Anti-IL15 Ab':1}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c timepoint")
        atypes = ['Pre', 'Post']
        ahash = {'Post Treatment':1, 'Pre Treatment':0, 'Prost Treatment':1}
        if (tn == 2):
            atype = [atype[i] if dval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLaszlo2016(self, tn=1):
        self.prepareData("COV74")
        atype = self.h.getSurvName('c stimulated with')
        atypes = ['WT IL2', 'F42K IL2', 'IL15']
        ahash = {'0.4nM F42K IL-2 for 24 hr':1, '0.4nM IL-15 for 24 hr':2,
                '0.4nM WT IL-2 for 24 hr':0}
        if (tn == 2):
            atypes = ['IL2', 'IL15']
            ahash = {'0.4nM IL-15 for 24 hr':1,
                    '0.4nM WT IL-2 for 24 hr':0}
        self.initData(atype, atypes, ahash)

    def getShao2013(self, tn=1):
        self.prepareData("COV75")
        atype = self.h.getSurvName("c passage")
        atypes = ['P19', 'P18', 'P17', 'P16']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c agent")
            atypes = ['C', 'S1PR']
            ahash = {'Untreated cells':0, 'Fingolimod 4uM':1}
        if (tn == 3):
            atype = self.h.getSurvName("c agent")
            atypes = ['U', 'C']
            ahash = {'Untreated cells':0, 'Cyclophosphamide 5mM':1}
        if (tn == 4):
            atype = self.h.getSurvName("c agent")
            atypes = ['U', 'C']
            ahash = {'Untreated cells':0, 'Cyclophosphamide S9 treated 3mM':1}
        self.initData(atype, atypes, ahash)

    def getMelo2020(self, tn=1):
        self.prepareData("COV76.3")
        atype = self.h.getSurvName("c src1")
        atypes = ['C1', 'C2', 'RSV', 'IAV', 'CoV1', 'CoV2']
        ahash = {'Mock treated NHBE cells':0,
                'SARS-CoV-2 infected NHBE cells':4,
                'Mock treated A549 cells':1,
                'SARS-CoV-2 infected A549 cells':5,
                'RSV infected A549 cells':2,
                'IAV infected A549 cells':3}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['C', 'CoV']
            ahash = {'SARS-CoV-2 infected A549 cells':1,
                    'Mock treated A549 cells':0}
        if (tn == 3):
            atypes = ['C', 'CoV']
            ahash = {'Mock treated NHBE cells':0,
                    'SARS-CoV-2 infected NHBE cells':1}
        if (tn == 4):
            atype = aval
            atypes = ['C', 'RSV', 'IAV', 'CoV']
            ahash = {1:0, 2:1, 3:2, 5:3}
        if (tn == 5):
            atype = self.h.getSurvName("c subject status")
            atypes = ['C', 'CoV']
            ahash = {'No treatment - healthy 72 years old, male':0,
                    'No treatment - healthy 77 years old, male':0,
                    'No treatment; >60 years old male COVID-19 deceased patient':1}
        if (tn == 6):
            atypes = ['C', 'CoV']
            ahash = {'SARS-CoV-2 infected A549 cells':1,
                    'Mock treated A549 cells':0,
                    'Mock treated NHBE cells':0,
                    'SARS-CoV-2 infected NHBE cells':1}
        self.initData(atype, atypes, ahash)

    def getMelo2020II(self, tn=1, ta=0, tb=1):
        self.prepareData("COV76.2")
        h = self.h
        atype = h.getSurvName("c tissue/cell type")
        ahash = {'Nasal Wash':0, 'Trachea':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName("c time after treatment")
        ahash = {'1 day':1, '3 days':3, '7 days':7, '14 days':14}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c src1')
        atypes = ['C', 'CoV', 'IAV']
        ahash = {'Mock treated 4 month old Ferret':0,
                'SARS-CoV-2 infected 4 month old Ferret':1,
                'IAV infected 4 month old Ferret':2}
        if (tn == 2):
            atype = [atype[i] if gval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if gval[i] == 1 else None
                    for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if gval[i] == 0 and 
                    (tval[i] == 3 or tval[i] == 7) else None
                    for i in range(len(atype))]
        if (tn == 5):
            atypes = ['C', 'CoV']
            ahash = {'Mock treated 4 month old Ferret':0,
                    'SARS-CoV-2 infected 4 month old Ferret':1}
            atype = [atype[i] if gval[i] == ta and tval[i] == tb else None
                    for i in range(len(atype))]
        if (tn == 6):
            atypes = ['C', 'CoV']
            ahash = {'Mock treated 4 month old Ferret':0,
                    'SARS-CoV-2 infected 4 month old Ferret':1}
            atype = [atype[i] if gval[i] == 0 and 
                    (tval[i] == 3 or tval[i] == 7) else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPalmieri2017(self, tn=1):
        self.prepareData("COV78")
        atype = self.h.getSurvName("c treatment")
        atypes = ['U', 'T']
        ahash = {'100 mM of trehalose':1, 'Untreated':0}
        self.initData(atype, atypes, ahash)

    def getOzgyin2018(self, tn=1):
        self.prepareData("COV79")
        atype = self.h.getSurvName("c treatment")
        atypes = ['U', 'T']
        ahash = {'non-lyophilized (control)':0, 'Lyophilized':1}
        self.initData(atype, atypes, ahash)

    def getMeugnier2011(self, tn=1):
        self.prepareData("COV80")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*after (.*) treatment", "\\1", str(k)) for k in atype]
        atypes = ['E', 'A']
        ahash = {'Etanercept':0, 'Adalimumab':1}
        self.initData(atype, atypes, ahash)

    def getZaba2007(self, tn=1):
        self.prepareData("COV81")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'T']
        ahash = {'Day 5 DC control':0, 'Day 5 DC etanercept':1}
        self.initData(atype, atypes, ahash)

    def getSeelbinder2020(self, tn=1):
        self.prepareData("COV83")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'AF', 'E', 'EAF']
        ahash = {'none':0,
                'A. fumigatus ATCC 46645':1,
                'Etanercept 2 ug/mL':2,
                'Etanercept 2 ug/mL; A. fumigatus ATCC 46645':3}
        self.initData(atype, atypes, ahash)

    def getBolenhcv(self, tn=1):
        self.prepareData("COV84")
        atype = self.h.getSurvName("c infection (ch1)")
        atypes = ['healthy', 'infected']
        ahash = {'Healthy':0, 'HCV':1}
        self.initData(atype, atypes, ahash)
        
    def getMoalhev(self, tn=1):
        self.prepareData("COV86")
        atype = self.h.getSurvName("c patient type (ch1)")
        atypes = ['healthy', 'infected']
        ahash = {'Control':0, 'Infected':1}
        self.initData(atype, atypes, ahash)

    def getBrandes2013(self, tn=1):
        self.prepareData("COV87")
        atype = self.h.getSurvName("c cell type")
        ahash = {'':0, 'alveolar macrophage':1, 'lymphocyte (BC, TC, NK)':2,
                'Ly6Chi mononuclear myeloid cell':3, 'neutrophil':4,
                'pulmonary CD45neg':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c perturbation")
        atypes = ['C', 'A', 'T', 'P']
        ahash = {'H1N1 influenza A PR8 (100LD50)':3,
                'H1N1 influenza A PR8 (0.2LD50)':3,
                'H1N1 influenza A PR8 (0.6LD50)':3,
                'H1N1 influenza A PR8 (10LD50)':3,
                'H1N1 influenza A 0.6LD50 PR8':3,
                'ALUM (162g)':1,
                'H1N1 influenza A TX91 (10^6PFU)':2,
                'H1N1 influenza A TX91 (10^6 PFU)':2,
                'sham':0}
        if (tn >= 2):
            atypes = ['NL', 'SL', 'L']
            ahash = {'H1N1 influenza A PR8 (100LD50)':2,
                    'H1N1 influenza A PR8 (10LD50)':2,
                    'H1N1 influenza A PR8 (0.2LD50)':1,
                    'H1N1 influenza A PR8 (0.6LD50)':1,
                    'H1N1 influenza A 0.6LD50 PR8':1,
                    'H1N1 influenza A TX91 (10^6PFU)':0,
                    'H1N1 influenza A TX91 (10^6 PFU)':0}
            atype = [atype[i] if tval[i] == (tn - 1) else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getJung2016(self, tn=1):
        self.prepareData("COV88")
        atype = self.h.getSurvName("c bmi")
        atypes = ['N', 'Obese', 'Overweight']
        ahash = {'18.5~23 kg/m2':0, '27.5~30 kg/m2':2, '25~27.5 kg/m2':1}
        if (tn == 2):
            atypes = ['N', 'Obese']
            ahash = {'18.5~23 kg/m2':0, '25~27.5 kg/m2':1}
        if (tn == 3):
            atypes = ['N', 'Overweight']
            ahash = {'18.5~23 kg/m2':0, '27.5~30 kg/m2':1}
        self.initData(atype, atypes, ahash)

    def getEckle2013(self, tn=1):
        self.prepareData("COV89")
        atype = self.h.getSurvName("c treatment")
        atypes = ['NS', 'S']
        ahash = {'non-strained':0, 'strained':1}
        self.initData(atype, atypes, ahash)

    def getOmura2018(self, tn=1):
        self.prepareData("COV90")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("([1a]) .*", "\\1", str(k)) for k in atype]
        atypes = ['CN', 'IN', 'CH', 'IH', 'UV']
        ahash = {'Mock-infected Neuro-2a':0,
                'Mock-infected HL-1':2,
                'HL-1':4,
                'TMEV-infected HL-1':3,
                'TMEV-infected Neuro-2a':1}
        if (tn == 2):
            atypes = ['C', 'I']
            ahash = {'Mock-infected HL-1':0, 'TMEV-infected HL-1':1}
        self.initData(atype, atypes, ahash)

    def getOmura2014(self, tn=1):
        self.prepareData("COV91")
        atype = self.h.getSurvName('c time point')
        ahash = {'4 days post infection':4, '7 days post infection':7,
                '60 days post infection':60}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infected with")
        atypes = ['C', 'I']
        ahash = {"Theiler's murine encephalomyelitis virus (TMEV)":1,
                'none (naïve control)':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 4 else None
                for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getCoronado2012(self, tn=1):
        self.prepareData("COV92")
        atype = self.h.getSurvName('c time')
        ahash = {'10 dpi':0, '90 dpi':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c gender')
        ahash = {'Female':0, 'Male':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'I']
        ahash = {'PBS':0, 'CVB3':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 and gval[i] == 1 else None
                for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'FI', 'MI']
            atype = [atype[i] if tval[i] == 0 else None
                for i in range(len(atype))]
            atype = ['MI' if aval[i] == 1 and tval[i] == 0 and
                    gval[i] == 1 else atype[i] for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getClementeCasares2017(self, tn=1):
        self.prepareData("COV93")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(" rep.*", "", str(k)) for k in atype]
        atypes = ['DC103', 'DC11b', 'MF']
        ahash = {'Cardiac CD103+ DC':0,
                'Cardiac CD11b+ DC':1,
                'Cardiac MHC-IIhi MF':2}
        self.initData(atype, atypes, ahash)

    def getSchinke2004(self, tn=1):
        self.prepareData("COV95")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("PGA-(.*)-.*", "\\1", str(k)) for k in atype]
        atypes = ['CCMP', 'N', 'AS']
        ahash = {}
        if (tn == 2):
            atypes = ['N', 'CCMP']
        if (tn == 3):
            atypes = ['N', 'AS']
        self.initData(atype, atypes, ahash)

    def getSchinke2004II(self, tn=1):
        self.prepareData("COV96")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("PGA[-_](.*)[-_][0-9]+", "\\1", str(k)) for k in atype]
        atype = [re.sub("_", "-", str(k)) for k in atype]
        atypes = ['Hs-V', 'Hs-S', 'PA-D', 'Hs-D', 'PA-N', 'Hs-F',
                'Hs-H', 'PA-S', 'Hs-P']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getClarelli2017(self, tn=1):
        self.prepareData("COV97")
        atype = self.h.getSurvName("c agent")
        atypes = ['C', 'IFNb']
        ahash = {'IFN-beta':1, 'control':0}
        self.initData(atype, atypes, ahash)

    def getHu2012(self, tn=1):
        self.prepareData("COV98")
        atype = self.h.getSurvName("c src1")
        atypes = ['C', 'LPS', 'TNFa', 'S', 'S+L', 'S+Ta']
        ahash = {'BEAS-2B - control':0,
                'BEAS-2B - mechanical stretch plus TNF-\xce\xb1':5,
                'BEAS-2B - mechanical stretch plus LPS':4,
                'BEAS-2B - LPS':1,
                'BEAS-2B - mechanical stretch':3,
                'BEAS-2B - TNF-\xce\xb1':2}
        if (tn == 2):
            atypes = ['C', 'S']
            ahash = {'BEAS-2B - control':0,
                    'BEAS-2B - mechanical stretch':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getdosSantos2004(self, tn=1):
        self.prepareData("COV99")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*from ", "", str(k)) for k in atype]
        atype = [re.sub("RNA ", "", str(k)) for k in atype]
        atype = [re.sub("hr .*", "", str(k)) for k in atype]
        atype = [re.sub(".ells[ \\-]*", "", str(k)) for k in atype]
        atype = [re.sub(".tatic ", "", str(k)) for k in atype]
        atype = [re.sub(" [JO][ac][nt].*", "", str(k)) for k in atype]
        atype = [re.sub(" [14]", "", str(k)) for k in atype]
        atype = [re.sub("A549 ", "", str(k)) for k in atype]
        atypes = ['C', 'LPS', 'TNFa', 'S', 'S+Ta']
        ahash = {'Control':0, 'LPS':1, 'LPSh':1, 'TNF':2,
                'Stretch':3, 'stretch':3, 'TNF+Stretch':4}
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atypes = ['C', 'S']
            ahash = {'A549 TNF+Stretch 1hr Jan 31 2003':1,
                    'mRNA from A549 Static Control 1hr Jan 31 2003':0,
                    'A549 TNF+Stretch 4hr Jan 31 2003':1,
                    'RNA A549 cells Static Control 4hr Replicate 1':0,
                    'A549 stretch 4hr Jan 31 2003':1,
                    'A549 Stretch 1hr Jan 31 2003':1,
                    'RNA from A549 Cells - Static Control 1hr Replicate 1':0,
                    'A549 RNA Static Control 4hr Jan 31 2003':0}
        self.initData(atype, atypes, ahash)

    def getSwanson2012(self, tn=1):
        self.prepareData("COV100")
        atype = self.h.getSurvName("c src1")
        atypes = ['noVAP', 'VAP']
        ahash = {'patients without VAP':0, 'patients with VAP':1}
        self.initData(atype, atypes, ahash)

    def getGharib2014(self, tn=1):
        self.prepareData("COV101")
        atype = self.h.getSurvName("c tissue type")
        atypes = ['Kidney', 'Lung', 'Liver']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub("_.$", "", str(k)) for k in atype]
            atypes = ['C', 'MV']
            ahash = {'Kidney_Control':0,
                    'Lung_MV+SA':1,
                    'Kidney_MV+SA':1,
                    'Lung_Control':0}
        self.initData(atype, atypes, ahash)

    def getStaudt2018(self, tn=1):
        self.prepareData("COV103")
        atype = self.h.getSurvName("c tissue")
        atypes = ['AM', 'SAE']
        ahash = {'alveolar macrophage':0, 'small airway epithelium brushing':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c smoking status")
            atypes = ['NS', 'ENN', 'EN']
            ahash = {'nonsmoker':0, 'Ecig_no-nicotine':1, 'Ecig+nicotine':2}
            atype = [atype[i] if aval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName("c smoking status")
            atypes = ['NS', 'ENN', 'EN']
            ahash = {'nonsmoker':0, 'Ecig_no-nicotine':1, 'Ecig+nicotine':2}
            atype = [atype[i] if aval[i] == 1 else None
                    for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getQuast2019(self, tn=1):
        self.prepareData("COV104")
        sex = self.h.getSurvName("c Sex")
        atype = self.h.getSurvName("c disease state")
        atypes = ['H', 'COPD']
        ahash = {'healthy':0, 'COPD':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c smoker")
            atypes = ['N', 'Y']
            ahash = {}
        if (tn == 3):
            atype = sex
            atypes = ['F', 'M']
            ahash = {}
        if (tn == 4):
            atype = self.h.getSurvName("c treatment")
            atypes = ['0', '4']
            ahash = {}
            atype = [atype[i] if aval[i] == 0 else None
                    for i in range(len(atype))]
        if (tn == 5):
            atype = self.h.getSurvName("c treatment")
            atypes = ['Air', 'CS']
            ahash = {'0':0, '4':1}
        self.initData(atype, atypes, ahash)

    def getYu2011(self, tn=1):
        self.prepareData("COV105")
        atype = self.h.getSurvName("c infection")
        atypes = ['U', 'EV71']
        ahash = {'uninfected':0, 'EV71':1}
        self.initData(atype, atypes, ahash)

    def getMolinaNavarro2013(self, tn=1):
        self.prepareData("COV106")
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'DC', 'IC']
        ahash = {'Ischemic cardiomyopathy':2,
                'Dilated cardiomyopathy':1,
                'Normal heart':0}
        if (tn == 2):
            atypes = ['N', 'DC']
            ahash = {'Dilated cardiomyopathy':1,
                    'Normal heart':0}
        if (tn == 3):
            atypes = ['N', 'IC']
            ahash = {'Ischemic cardiomyopathy':1,
                    'Normal heart':0}
        self.initData(atype, atypes, ahash)

    def getLi2020(self, tn=1):
        self.prepareData("COV107")
        atype = self.h.getSurvName("c src1")
        atypes = ['LV', 'ARV']
        ahash = {"ARVC patients' heart LV":0, 'heart RV tissue':1}
        self.initData(atype, atypes, ahash)

    def getRen2020(self, tn=1):
        self.prepareData("COV108")
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'HF', 'HCM']
        ahash = {'heart failure':1, 'hypertrophic cardiomyopathy':2,
                'normal heart tissue':0}
        if (tn == 2):
            atypes = ['N', 'HF']
            ahash = {'heart failure':1, 'normal heart tissue':0}
        if (tn == 3):
            atypes = ['N', 'HCM']
            ahash = {'hypertrophic cardiomyopathy':1, 'normal heart tissue':0}
        self.initData(atype, atypes, ahash)

    def getRen2020Mm(self, tn=1):
        self.prepareData("COV108.2")
        atype = self.h.getSurvName("c src1")
        atypes = ['N', 'C', 'CM']
        ahash = {'TAC sham CM':1, 'TAC2W CM':2, 'TAC5W CM':2,
                'TAC8W CM':2, 'TAC11W CM':2, 'normal heart CM':0}
        if (tn == 2):
            atypes = ['C', 'CM']
            ahash = {'TAC sham CM':0, 'TAC2W CM':1, 'TAC5W CM':1,
                    'TAC8W CM':1, 'TAC11W CM':1, 'normal heart CM':0}
        self.initData(atype, atypes, ahash)

    def getMorley2019(self, tn=1):
        self.prepareData("COV109")
        atype = self.h.getSurvName("c etiology")
        atypes = ['N', 'DCM', 'HCM', 'PPCM']
        ahash = {'Dilated cardiomyopathy (DCM)':1,
                'Non-Failing Donor':0,
                'Hypertrophic cardiomyopathy (HCM)':2,
                'Peripartum cardiomyopathy (PPCM)':3}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['H', 'CM']
            ahash = {'Dilated cardiomyopathy (DCM)':1,
                    'Non-Failing Donor':0,
                    'Hypertrophic cardiomyopathy (HCM)':1,
                    'Peripartum cardiomyopathy (PPCM)':1}
        if (tn == 3):
            atype = self.h.getSurvName("c race")
            atypes = ['W', 'B']
            ahash = {'Caucasian':0, 'African American':1}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLiu2015(self, tn=1):
        self.prepareData("COV110")
        age = self.h.getSurvName("c age")
        age = [int(age[i]) if i > 1 else None for i in range(len(age))]
        sex = self.h.getSurvName("c gender")
        atype = self.h.getSurvName("c disease status")
        atypes = ['NF', 'ICM', 'CMP']
        ahash = {'ischemic':1, 'non-failing':0, 'idiopathic dilated CMP':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if sex[i] == 'male' and age[i] < 40 
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if sex[i] == 'female' and age[i] < 60
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = age
            ahash = {}
            atypes = ['Y', 'O']
            for i in range(len(atype)):
                if age[i] < 30 and age[i] < 10:
                    ahash[atype[i]] = 0
                if age[i] < 30 and age[i] > 10:
                    ahash[atype[i]] = 1
            atype = [atype[i] if aval[i] == 0 
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = self.h.getSurvName("c heart failure")
            atypes = ['no', 'yes']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getAkat2014(self, tn=1):
        self.prepareData("COV111")
        h = self.h
        atype = h.getSurvName("c disease")
        atypes = ['NF', 'ICM', 'DCM']
        ahash = {'None':0, 'Ischemic Cardiomyopathy':1, 'Dilated Cardiomyopathy':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = h.getSurvName('c treatment')
            atypes = ['N', 'E', 'I']
            ahash = {'None':0, 'LVAD Explantation':1, 'LVAD Implantation':2}
        if (tn == 3):
            atype = h.getSurvName('c treatment')
            atypes = ['N', 'E', 'I']
            ahash = {'None':0, 'LVAD Explantation':1, 'LVAD Implantation':2}
            atype = [atype[i] if aval[i] == 2 or atype[i] == 'None'
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = h.getSurvName('c treatment')
            atypes = ['N', 'E', 'I']
            ahash = {'None':0, 'LVAD Explantation':1, 'LVAD Implantation':2}
            atype = [atype[i] if aval[i] == 1 or atype[i] == 'None'
                    else None for i in range(len(atype))]
        if (tn == 5):
            atypes = ['NF', 'CM']
            ahash = {'None':0, 'Ischemic Cardiomyopathy':1, 'Dilated Cardiomyopathy':1}
        self.initData(atype, atypes, ahash)

    def getCasey2016(self, tn=1):
        self.prepareData("COV112")
        age = self.h.getSurvName("c age")
        age = [int(age[i]) if i > 1 else None for i in range(len(age))]
        sex = self.h.getSurvName("c Sex")
        atype = sex
        atypes = ['F', 'M']
        ahash = {}
        if (tn == 2):
            atype = age
            ahash = {}
            atypes = ['Y', 'O']
            for i in range(len(atype)):
                if age[i] is None:
                    continue
                if age[i] > 0 and age[i] < 55:
                    ahash[atype[i]] = 0
                if age[i] > 0 and age[i] > 75:
                    ahash[atype[i]] = 1
        self.initData(atype, atypes, ahash)

    def getHannenhalli2006(self, tn=1):
        self.prepareData("COV113")
        atype = self.h.getSurvName("c src1")
        atypes = ['NF', 'LV']
        ahash = {'explanted heart tissue at time of cardiac transplantation':1,
                 'unused donor heart with normal LV function':0}
        self.initData(atype, atypes, ahash)

    def getJacobson2016(self, tn=1, ar=None):
        self.prepareData("COV114")
        time = self.h.getSurvName("c timepoint")
        ahash = {'Week12':12, '12wk':12, '0wk':0, 'Week0':0}
        time = [ahash[i] if i in ahash else None for i in time]
        atype = self.h.getSurvName("c src1")
        atypes = ['U', 'T']
        ahash = {'Blood_untreated_chronic_HIV_placebo':0,
                'Blood_antiretroviral_therapy_placebo':1,
                'Blood_antiretroviral_therapy_Chloroquine':1,
                'Blood_untreated_chronic_HIV_Chloroq':0,
                'Blood_untreated_chronic_HIV_Chloroq_placebo':0}
        if (tn == 2):
            ahash = {'Blood_untreated_chronic_HIV_placebo': 'P',
                    'Blood_untreated_chronic_HIV_Chloroq': 'Q',
                    'Blood_untreated_chronic_HIV_Chloroq_placebo': 'P'}
            treat = [ahash[i] if i in ahash else None for i in atype]
            atype = [ahash[atype[i]] + ' ' + str(time[i]) if atype[i] in ahash
                    else None for i in range(len(atype))]
            if (ar is not None):
                atype = [atype[i] if time[i] == ar
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        if (tn == 3):
            ahash = {'Blood_antiretroviral_therapy_placebo': 'P',
                    'Blood_antiretroviral_therapy_Chloroquine': 'Q'}
            treat = [ahash[i] if i in ahash else None for i in atype]
            atype = [ahash[atype[i]] + ' ' + str(time[i]) if atype[i] in ahash
                    else None for i in range(len(atype))]
            if (ar is not None):
                atype = [atype[i] if time[i] == ar
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        if (tn == 4):
            ahash = {'Blood_untreated_chronic_HIV_placebo': 'P',
                    'Blood_untreated_chronic_HIV_Chloroq': 'Q',
                    'Blood_untreated_chronic_HIV_Chloroq_placebo': 'P'}
            treat = [ahash[i] if i in ahash else None for i in atype]
            atype = [ahash[atype[i]] + ' ' + str(time[i]) if atype[i] in ahash
                    else None for i in range(len(atype))]
            if (ar is not None):
                atype = [atype[i] if treat[i] == ar
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        if (tn == 5):
            ahash = {'Blood_antiretroviral_therapy_placebo': 'P',
                    'Blood_antiretroviral_therapy_Chloroquine': 'Q'}
            treat = [ahash[i] if i in ahash else None for i in atype]
            atype = [ahash[atype[i]] + ' ' + str(time[i]) if atype[i] in ahash
                    else None for i in range(len(atype))]
            if (ar is not None):
                atype = [atype[i] if treat[i] == ar
                        else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getLhakhang2014(self, tn=1):
        self.prepareData("COV115")
        atype = self.h.getSurvName("c treatment")
        atypes = ['U', 'T']
        ahash = {'untreated':0, 'hY3 treated':1}
        self.initData(atype, atypes, ahash)

    def getTakeda2018(self, tn=1):
        self.prepareData("COV116")
        atype = self.h.getSurvName("c treatment")
        atypes = ['V', 'Q', 'M', 'M+O']
        ahash = {'mefloquine with trans-l-diaminocyclohexane oxalatoplatinum':3,
                'vehicle':0,
                'chloroquine':1,
                'mefloquine':2}
        if (tn == 2):
            atypes = ['V', 'M']
            ahash = {'vehicle':0,
                    'mefloquine':1}
        self.initData(atype, atypes, ahash)

    def getBargiela2019(self, tn=1):
        self.prepareData("COV117")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".$", "", str(k)) for k in atype]
        atypes = ['MC', 'MD', 'Q 0.1M', 'Q 10M']
        ahash = {'Myoblast_Control':0,
                'Myoblast_Treated_01_S':2,
                'Myoblast_Treated_10_S':3,
                'Myoblast_Disease':1}
        if (tn == 2):
            atypes = ['MC', 'Q']
            ahash = {'Myoblast_Control':0,
                    'Myoblast_Treated_01_S':1,
                    'Myoblast_Treated_10_S':1}
        self.initData(atype, atypes, ahash)

    def getGoldberg2018(self, tn=1):
        self.prepareData("COV118")
        meds = self.h.survhdrs[15:24]
        atype = self.h.getSurvName(meds[0])
        for k in meds[1:]:
            atype1 = self.h.getSurvName(k)
            atype = [atype[i] + atype1[i] for i in range(len(atype))]
        drugs = atype
        atype = self.h.getSurvName('c gender')
        atypes = ['F', 'M']
        ahash = {}
        if (tn == 2):
            atype = drugs
            atypes = ['C', 'M', 'I']
            ahash = {'--Celebrex--------------':0,
                    'Methotrexate----------------':1,
                    '------Infliximab----------':2}
        self.initData(atype, atypes, ahash)

    def getGrinman2019(self, tn=1):
        self.prepareData("COV119")
        atype = self.h.getSurvName('c stage of mammary development')
        atypes = ['A', 'D']
        ahash = {'Secretory Activation':0,
                'Secretory Differentiation':1}
        self.initData(atype, atypes, ahash)

    def getThomas2018(self, tn=1):
        self.prepareData("COV120")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'LPS', 'T0+LPS']
        ahash = {'DMSO':0, 'DMSO+LPS':1, 'T0+LPS':2}
        self.initData(atype, atypes, ahash)

    def getSallam2018(self, tn=1):
        self.prepareData("COV121")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'T']
        ahash = {'Macrophage (DMSO)':0, 'Macrophage (GW3965)':1}
        self.initData(atype, atypes, ahash)

    def getDun2013(self, tn=1):
        self.prepareData("COV122")
        atype = self.h.getSurvName('c src1')
        atype = [re.sub(".*, (.*)h .*", "\\1", str(k)) for k in atype]
        atypes = ['C', 'T']
        ahash = {'72':1, '0':0, '24':1, '12':1, '48':1}
        self.initData(atype, atypes, ahash)

    def getWood2016(self, tn=1):
        self.prepareData("COV123")
        atype = self.h.getSurvName('c disease state')
        atypes = ['C', 'SSc', 'Mor']
        ahash = {'Systemic Sclerosis':1, 'SSc patient':1,
                '':0, 'normal control':0, 'normal':0, 'morphea patient':2}
        if (tn == 2):
            atype = self.h.getSurvName('c treatment')
            atypes = ['U', 'Q']
            ahash = {'':0, 'NA':0, 'Plaquenil':1}
        if (tn == 3):
            atype = self.h.getSurvName('c treatment')
            atypes = ['U', 'MMF']
            ahash = {'':0, 'NA':0, 'mycophenolate mofetil':1, 'MMF':1}
        self.initData(atype, atypes, ahash)

    def getFribourg2020(self, tn=1):
        self.prepareData("COV124")
        atype = self.h.getSurvName('c tacrolimus withdrawal')
        atypes = ['NW', 'WS', 'WR']
        ahash = {'Withdrawal Rejection (WR)':2,
                'Withdrawal Stable (WS)':1,
                'No withdrawal (NW)':0}
        self.initData(atype, atypes, ahash)

    def getMunz2020(self, tn=1):
        self.prepareData("COV125")
        atype = self.h.getSurvName('c treatment group')
        atypes = ['M', 'T']
        ahash = {'Mock (FK506\xe2\x80\x93)':0, 'FK506 (FK506+)':1}
        ahash = asciiNorm(ahash)
        self.initData(atype, atypes, ahash)

    def getDorr2015(self, tn=1):
        self.prepareData("COV126")
        atype = self.h.getSurvName('c time of blood draw')
        atypes = ['Pre', '1w', '3m', '6m']
        ahash = {'3 months post-transplant':2,
                'pre-transplant':0,
                '1 week post-transplant':1,
                '6 months post-transplant':3}
        if (tn == 2):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'AA', 'AI']
            ahash = {'American Indian orAlaskaNative':2,
                    'Caucasian / White':0,
                    'Black or African American':1}
        self.initData(atype, atypes, ahash)

    def getCamus2012(self, tn=1):
        self.prepareData("COV127")
        h = self.h
        atype = h.getSurvName('c Title')
        atype = [re.sub("\xc2\xb5", "u", str(k)) for k in atype]
        atype = [re.sub("\xb5", "u", str(k)) for k in atype]
        time = [re.sub(".*([0-9]+)h.*", "\\1", str(k)) for k in atype]
        treat = [re.sub(".*(CQ).*", "CQd", str(k)) for k in atype]
        treat = [re.sub(".*(uninfe).*", "U", str(k)) for k in treat]
        treat = [re.sub(".*(uninfe).*", "U", str(k)) for k in treat]
        treat = [re.sub("^H.*", "U", str(k)) for k in treat]
        infection = [re.sub(".*(JFH1).*", "I", str(k)) for k in atype]
        infection = [re.sub("^H.*", "U", str(k)) for k in infection]
        atype = [treat[i] + " " + infection[i] for i in range(len(atype))]
        atypes = ['U U', 'CQd U', 'CQd I', 'U I']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getAlao2018(self, tn=1):
        self.prepareData("COV128")
        h = self.h
        atype = h.getSurvName('c treatment')
        atypes = ['U', 'T']
        ahash = {'Base':0, 'Wk2':1, 'Wk4':1}
        self.initData(atype, atypes, ahash)

    def getMeissner2014I(self, tn=1):
        self.prepareData("COV129")
        h = self.h
        atype = h.getSurvName('c time point')
        atypes = ['U', 'T']
        ahash = {'day 24':1, 'day 10':1,
                'end of treatment (post)':1,
                'pre treatment':0, 'day 0':0,
                'end of treatment (post-C)':1,
                'pre-C treatment':0}
        if (tn == 2):
            ahash = {'day 24':1, 'day 10':1,
                    'end of treatment (post)':1,
                    'pre treatment':0, 'day 0':0,
                    'end of treatment (post-C)':0,
                    'pre-C treatment':0}
        self.initData(atype, atypes, ahash)

    def getMeissner2014II(self, tn=1):
        self.prepareData("COV130")
        h = self.h
        atype = h.getSurvName('c tissue')
        atypes = ['S', 'R']
        ahash = {'unpaired liver biopsy with sustained virologic response':0,
                'unpaired liver biopsy with relapse':1}
        self.initData(atype, atypes, ahash)

    def getCostarelli2017(self, tn=1):
        self.prepareData("COV131")
        atype = self.h.getSurvName('c treatment')
        atypes = ['U', 'OMP', 'LSP']
        ahash = {'lansoprazole':2, 'untreated':0, 'omeprazole':1}
        if (tn == 2):
            atype = self.h.getSurvName('c src1')
            atype = [re.sub(".*cells, ", "", str(k)) for k in atype]
            ahash = {'young, control':0,
                    'young, omeprazole':1, 'young,omeprazole':1}
            atypes = ['U', 'OMP']
        if (tn == 3):
            atype = self.h.getSurvName('c src1')
            atype = [re.sub(".*cells, ", "", str(k)) for k in atype]
            ahash = {'old, control':0, 'old, omeprazole':1}
            atypes = ['U', 'OMP']
        if (tn == 4):
            atype = self.h.getSurvName('c src1')
            atype = [re.sub(".*cells, ", "", str(k)) for k in atype]
            ahash = {'young, control':0,
                    'young, omeprazole':0, 'young,omeprazole':0,
                    'old, control':1, 'old, omeprazole':1}
            atypes = ['Y', 'O']
        if (tn == 5):
            atype = self.h.getSurvName('c src1')
            atype = [re.sub(".*cells, ", "", str(k)) for k in atype]
            ahash = {'young, control':0,
                    'young, omeprazole':1, 'young,omeprazole':1,
                    'old, control':2, 'old, omeprazole':3}
            atypes = ['YU', 'YOMP', 'OU', 'OOMP']
        self.initData(atype, atypes, ahash)

    def getAbdelAziz2016(self, tn=1):
        self.prepareData("COV132")
        h = self.h
        atype = h.getSurvName('c Title')
        tissue = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'whole':0, 'esophageal':1}
        tval = [ahash[i] if i in ahash else None for i in tissue]
        treat = [re.sub("whole blood ", "", str(k)) for k in atype]
        treat = [re.sub("esophageal tissue ", "", str(k)) for k in treat]
        treat = [re.sub(" .*", "", str(k)) for k in treat]
        atype = treat
        atypes = ['U', 'Sh', 'ST', 'O']
        ahash = {'STW5':2, 'sham':1, 'untreated':0, 'omeprazole':3}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
            atypes = ['U', 'OMP']
            ahash = {'untreated':0, 'omeprazole':1}
        if (tn == 3):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atypes = ['U', 'OMP']
            ahash = {'untreated':0, 'omeprazole':1}
        self.initData(atype, atypes, ahash)

    def getWickramasinghe2015(self, tn=1):
        self.prepareData("COV133")
        h = self.h
        atype = h.getSurvName('c incubated with')
        atypes = ['GLU', 'HMO', 'LAC']
        ahash = {'glucose-grown B. infantis':0,
                'HMO-grown B. infantis':1,
                'lactose-grown B. infantis':2}
        if (tn == 2):
            atypes = ['C', 'HMO']
            ahash = {'glucose-grown B. infantis':0,
                    'HMO-grown B. infantis':1,
                    'lactose-grown B. infantis':0}
        self.initData(atype, atypes, ahash)

    def getHead2011(self, tn=1):
        self.prepareData("COV134")
        h = self.h
        atype = h.getSurvName('c cell type')
        atype = [re.sub("Caco2 cells exposed to ", "", str(k)) for k in atype]
        atype = [re.sub("human milk oligosacharides", "HMO", str(k)) for k in atype]
        atypes = ['C', 'HMO', 'HMO+Bi', 'Bi+Lac']
        ahash = {'B. infantis pre-grown on lactose.':3, 'HMO':1,
                'HMO and to B. infantis pre-grown on HMO.':2,
                'Caco2 cells with no B. infantis and no HMO':0}
        if (tn == 2):
            atypes = ['C', 'HMO']
            ahash = {'HMO':1, 'Caco2 cells with no B. infantis and no HMO':0}
        self.initData(atype, atypes, ahash)

    def getChen2017(self, tn=1):
        self.prepareData("COV135")
        h = self.h
        atype = h.getSurvName('c infection status')
        atypes = ['C', 'Zika']
        ahash = {'mock':0, 'ZIKA-infected':1}
        if (tn == 2):
            atype = h.getSurvName('c src1')
            atypes = ['DMSO', 'AQ', 'HH']
            ahash = {'ZIKA infection, AQ drug':1,
                    'ZIKA infection, HH drug':2,
                    'ZIKA infection, DMSO vehicle':0}
        if (tn == 3):
            atype = h.getSurvName('c src1')
            atypes = ['DMSO', 'AQ']
            ahash = {'ZIKA infection, AQ drug':1,
                    'ZIKA infection, DMSO vehicle':0}
        if (tn == 4):
            atype = h.getSurvName('c src1')
            atypes = ['DMSO', 'HH']
            ahash = {'ZIKA infection, HH drug':1,
                    'ZIKA infection, DMSO vehicle':0}
        self.initData(atype, atypes, ahash)

    def getRialdi2016(self, tn=1):
        self.prepareData("COV136")
        h = self.h
        atype = h.getSurvName('c infection')
        ahash = {'no infection':0, 'A/PR/8/34(\xce\x94NS1) Infection':1}
        ahash = asciiNorm(ahash)
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c treatment')
        atypes = ['C', 'iTop']
        ahash = {'Top1 siRNA':1, 'Control siRNA':0, 'no siRNA':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getTheken2019(self, tn=1):
        self.prepareData("COV137")
        h = self.h
        atype = h.getSurvName('c drug treatment')
        ahash = {'ibuprofen sodium':1, 'Placebo':0}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c timepoint')
        ahash = {'baseline':0, 'post-surgery 1':1, 'post-surgery 2':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c response group')
        atypes = ['P', 'PR', 'CR']
        ahash = {'Partial responder':1, 'Placebo':0, 'Full responder':2}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 1 or tval[i] == 2
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = h.getSurvName('c drug treatment')
            atypes = ['P', 'IBU']
            ahash = {'ibuprofen sodium':1, 'Placebo':0}
        self.initData(atype, atypes, ahash)

    def getFerretti2018(self, tn=1):
        self.prepareData("COV138")
        h = self.h
        atype = h.getSurvName('c Title')
        atype = [re.sub("H.* ileum (.*)_.*", "\\1", str(k)) for k in atype]
        atypes = ['control', 'IBU']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getJabbari2015(self, tn=1):
        self.prepareData("COV139")
        h = self.h
        atype = h.getSurvName('c treatment')
        atypes = ['C', 'B']
        ahash = {'vehiclecontrol':0, 'baricitinib':1, 'vehicle control':0}
        if (tn == 2):
            atype = h.getSurvName('c src1')
            ahash = {'skin, topical baricitinib after disease establishment, week 0':0,
                    'skin, topical baricitinib after disease establishment, week 12':1}
        self.initData(atype, atypes, ahash)

    def getDrawnel2017(self, tn=1):
        self.prepareData("COV140")
        h = self.h
        atype = h.getSurvName('c cell type')
        atypes = ['CM']
        ahash = {'iPS-derived cardiomyocytes':0}
        if (tn == 2):
            atype = h.getSurvName('c compound name')
            atypes = ['C', 'R', 'Q']
            ahash = {'media_Ctrl':0, 'Resveratrol':1, 'Quinine-HCl-2H2O':2}
        if (tn == 3):
            atype = h.getSurvName('c compound name')
            atypes = ['C', 'Q']
            ahash = {'media_Ctrl':0, 'Quinine-HCl-2H2O':1}
        self.initData(atype, atypes, ahash)

    def getTakeshita2019(self, tn=1):
        self.prepareData("COV141")
        h = self.h
        atype = h.getSurvName('c disease status')
        atypes = ['H', 'SN', 'Non', 'TCZ', 'MTX', 'IFX']
        ahash = {'RA TCZ treatment':3, 'RA non treatment':2, 'RA MTX treatment':4,
                'RA IFX treatment':5, 'healthy':0, 'RA synovial fluid':1}
        if (tn == 2):
            atypes = ['C', 'TCZ']
            ahash = {'RA non treatment':0, 'RA TCZ treatment':1}
        self.initData(atype, atypes, ahash)

    def getNakamura2016(self, tn=1):
        self.prepareData("COV142")
        h = self.h
        atype = h.getSurvName('c sampling point')
        ahash = {'Before abatacept administration':0,
                'Before infliximab administration':1,
                'Before tocilizumab administration':2}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c clinical outcome')
        atypes = ['R', 'NR']
        ahash = {'Remission':0, 'Non-remission':1}
        if (tn == 2):
            atype = [atype[i] if dval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getNishimoto2014(self, tn=1):
        self.prepareData("COV143")
        h = self.h
        atype = h.getSurvName('c treatment')
        atypes = ['bT', 'aT', 'bM', 'aM']
        ahash = {'after Tocilizumab/MRA':1, 'before methotrexate/MTX':2,
                'after methotrexate/MTX':3, 'before Tocilizumab/MRA':0}
        self.initData(atype, atypes, ahash)

    def getGaertner2012(self, tn=1):
        self.prepareData("COV144")
        h = self.h
        atype = h.getSurvName('c ventricle')
        ahash = {'right':1, 'left':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c indication')
        atypes = ['NF', 'DCM', 'ARVC']
        ahash = {'Dilated cardiomyopathy':1,
                'Arrhythmogenic right ventricular cardiomyopathy':2,
                'Non-Failing':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = h.getSurvName('c ventricle')
            atype = [atype[i] if aval[i] == 2
                    else None for i in range(len(atype))]
            atypes = ['left', 'right']
            ahash = {}
        if (tn == 3):
            atypes = ['NF', 'HF']
            ahash = {'Dilated cardiomyopathy':1,
                    'Arrhythmogenic right ventricular cardiomyopathy':1,
                    'Non-Failing':0}
        self.initData(atype, atypes, ahash)

    def getvandenBerg2018(self, tn=1):
        self.prepareData("COV145")
        h = self.h
        atype = h.getSurvName('c tissue')
        ahash = {'Peripheral Blood Mononuclear Cells (PBMC)':0,
                'Whole Blood (WB)':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c sample day')
        atypes = ['D0', 'D40', 'D44', 'D47', 'D31', 'D37', 'D30']
        ahash = {}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['D0', 'D31', 'D44']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLemoine2018(self, tn=1):
        self.prepareData("COV146")
        h = self.h
        atype = h.getSurvName('c tissue')
        ahash = {'Peripheral blood':0, 'Cord Blood':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c stimulation')
        atypes = ['U', 'BCG', 'TLR', 'RSV']
        ahash = {'Unstimulated':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBerry2010(self, tn=1):
        self.prepareData("COV147")
        h = self.h
        atype = h.getSurvName('c src1')
        atype = [re.sub("Human ", "", str(k)) for k in atype]
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'whole':0, 'Whole':0, 'Neutrophils':2, 'CD8+':4,
                'Monocytes':1, 'CD4+':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c illness')
        atypes = ['C', 'LTB', 'PTB']
        ahash = {'LATENT TB':1, 'Control':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = h.getSurvName('c bcg vaccinated')
            atypes = ['Yes', 'No']
            ahash = {}
        if (tn == 3):
            atype = h.getSurvName('c ethnicity')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'Caucasian':0, 'African American':1, 'Hispanic':3,
                    'South Asian':2, 'Asian Other':2, 'Asian':2, 'White':0,
                    'Native American':3, 'Black':2, 'Asian other':2,
                    'Afican American':1, 'Caucasian/Asian':0, 'Other':3}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMatsumiya2014(self, tn=1):
        self.prepareData("COV148")
        h = self.h
        atype = h.getSurvName('c time')
        ahash = {'0':0, '14':14, '2':2, '7':7}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c group')
        atypes = ['noBCG', 'BCG']
        ahash = {'A':0, 'C':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLoxton2017(self, tn=1):
        self.prepareData("COV149")
        h = self.h
        atype = h.getSurvName('c timepoint')
        ahash = {'6':6, '0':0, '2':2, '26':26, '12':12, '18':18}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c group')
        atypes = ['VPM1002', 'BCG']
        ahash = {'VPM1002':0, 'BCG':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = h.getSurvName('c timepoint')
            atypes = ['0', '2-6-12', '18-26']
            ahash = {'2':1, '6':1, '12':1, '18':2, '26':2}
        self.initData(atype, atypes, ahash)

    def getKaushal2015(self, tn=1):
        self.prepareData("COV150")
        h = self.h
        atype = h.getSurvName('c vaccinated with')
        atypes = ['MtbDsigH', 'BCG']
        ahash = {'aerosols with attenuated MtbDsigH mutant':0,
                'aerosols with BCG':1}
        self.initData(atype, atypes, ahash)

    def getDarrah2020(self, tn=1):
        self.prepareData("COV151")
        h = self.h
        atype = h.getSurvName("c time after bcg vaccination")
        ahash = {'Week 13':13, 'Week 25':25}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName("c stimulation")
        ahash = {'Unstimulated cells':0, 'PPD-Stimulated Cells':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c vaccination route')
        atypes = ['noBCG', 'BCG']
        ahash = {'Intradermal':1, 'Naive':0, 'Aerosol':1, 'Intravenous':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 25 and gval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atype = [atype[i] if gval[i] == 0
                    else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getDarrah2020II(self, tn=1, ta=13, tb=0):
        self.prepareData("COV151.2")
        h = self.h
        atype = h.getSurvName("c Cell Type")
        ahash = {'Epithelial':0, 'Proliferating':1, 'Mac':2, 'Eo':3,
                'Neutrophils':4, 'Mast':5, 'B':6, 'T':7, 'NA':8}
        bval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName("c time after bcg vaccination")
        ahash = {'Week 13':13, 'Week 25':25}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName("c stimulation")
        ahash = {'Unstimulated cells':0, 'PPD-Stimulated Cells':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName('c vaccination route')
        atypes = ['noBCG', 'BCG']
        ahash = {'Intradermal':1, 'Naive':0, 'Aerosol':1, 'Intravenous':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if gval[i] == 0 and tval[i] == 25 and bval[i] == 8
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = gval
            phash = {6:1, 7:1}
            atype = [atype[i] if tval[i] == ta and bval[i] in phash and aval[i] == tb
                    else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        if (tn == 4):
            atype = gval
            phash = {2:1, 5:1}
            atype = [atype[i] if tval[i] == ta and bval[i] in phash and aval[i] == tb
                    else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getHatae2020(self, tn=1):
        self.prepareData("LU14")
        h = self.h
        atype = h.getSurvName("c treatment")
        atypes = ['Pre', 'Post']
        ahash = {'Pre-treatment':0, 'Post-treatment (nivolumab)':1}
        self.initData(atype, atypes, ahash)

    def getKadara2013(self, tn=1):
        self.prepareData("LU18")
        h = self.h
        atype = h.getSurvName("c airway site")
        ahash = {'Contralateral':4, 'Adjacent':3, 'Non-adjacent':2,
                'Main carina':1, 'NA':0}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = h.getSurvName("c time point")
        ahash = {'0':0, '12':12, '24':24, '36':36}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['0', '12', '24', '36']
        ahash = {}
        if (tn == 2):
            atype = [gval[i] if tval[i] == 12 or tval[i] == 36
                    else None for i in range(len(atype))]
            atypes = ['A', 'Na', 'C', 'Mc']
            ahash = {3:0, 2:1, 1:3, 4:2}
        self.initData(atype, atypes, ahash)

    def getNorenHooten2019(self, tn=1):
        self.prepareData("COV152")
        h = self.h
        atype = h.getSurvName("c age group")
        atypes = ['Y', 'O']
        ahash = {}
        if (tn == 2):
            atype = h.getSurvName("c race")
            atypes = ['W', 'AA']
            ahash = {'White':0, 'African American':1}
        if (tn == 3):
            atype = h.getSurvName("c poverty")
            atypes = ['A', 'B']
            ahash = {'Above':0, 'Below':1}
        self.initData(atype, atypes, ahash)

    def getPrince2019(self, tn=1):
        self.prepareData("COV153")
        h = self.h
        atype = h.getSurvName("c race")
        atypes = ['W', 'AA']
        ahash = {'White':0, 'African American':1}
        if (tn == 2):
            atype = h.getSurvName("c frailty status")
            ahash = {'Non-Frail':0, 'Frail':1}
            atypes = ['NF', 'F']
        if (tn == 3):
            group = h.getSurvName("c frailty status")
            atype = [atype[i] if group[i] == 'Frail'
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDluzen2016(self, tn=1):
        self.prepareData("COV154")
        h = self.h
        atype = h.getSurvName("c race")
        atypes = ['W', 'AA']
        ahash = {'African American':1, 'Caucasian':0}
        if (tn == 2):
            atype = h.getSurvName("c src1")
            atypes = ['WN', 'WH', 'AAN', 'AAH']
            ahash = {'African American_hypertensives':3,
                    'African American_normotensives':2,
                    'white normotensives':0, 'white hypertensives':1,
                    'white_hypertensives':1}
        if (tn == 3):
            atype = h.getSurvName("c src1")
            atypes = ['WN', 'WH']
            ahash = {'white normotensives':0, 'white hypertensives':1,
                    'white_hypertensives':1}
        if (tn == 4):
            atype = h.getSurvName("c src1")
            atypes = ['AAN', 'AAH']
            ahash = {'African American_hypertensives':1,
                    'African American_normotensives':0}
        self.initData(atype, atypes, ahash)

    def getHubal2017(self, tn=1):
        self.prepareData("COV155")
        h = self.h
        group = h.getSurvName("c group")
        src1 = h.getSurvName("c src1")
        atype = [" ".join([str(group[i]), str(src1[i])])
                for i in range(len(group))]
        atypes = ['HB', 'PreDB', 'HP', 'PreDP']
        ahash = {'Healthy PBMC_baseline':0, 'Healthy PBMC_post-ex':2,
                'Prediabetic PBMC_baseline':1, 'Prediabetic PBMC_post-ex':3}
        if (tn == 2):
            atypes = ['HB', 'PreDB']
            ahash = {'Healthy PBMC_baseline':0, 'Prediabetic PBMC_baseline':1}
        if (tn == 3):
            atypes = ['B', 'Ex']
            ahash = {'Healthy PBMC_baseline':0, 'Healthy PBMC_post-ex':1,
                    'Prediabetic PBMC_baseline':0, 'Prediabetic PBMC_post-ex':1}
        self.initData(atype, atypes, ahash)

    def getSilva2018(self, tn=1):
        self.prepareData("COV156")
        h = self.h
        atype = h.getSurvName("c race")
        atypes = ['brown', 'white', 'black']
        ahash = {}
        if (tn == 2):
            atypes = ['white', 'black']
        if (tn == 3):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub(".*, ", "", str(k)) for k in atype]
            atype = [re.sub(" .*", "", str(k)) for k in atype]
            atypes = ['C', 'UnC']
            ahash = {'uncontrolled':1, 'controlled':0}
        self.initData(atype, atypes, ahash)

    def getLee2014I(self, tn=1):
        self.prepareData("COV157")
        h = self.h
        atype = h.getSurvName("c ethnicity")
        atypes = ['W', 'AA', 'A', 'M']
        ahash = {'Caucasian':0, 'African American':1, 'Asian':2,
                'African-American':1, 'East Asian':2, 'MULTI-RACIAL':3}
        if (tn == 2):
            group = h.getSurvName("c stimulation")
            atype = [atype[i] if group[i] == 'Unstim'
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = h.getSurvName("c stimulation")
            atypes = ['Unstim', 'LPS', 'dNS1']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getLee2014II(self, tn=1):
        self.prepareData("COV157.2")
        h = self.h
        atype = h.getSurvName("c ethnicity")
        atypes = ['W', 'AA', 'A', 'M']
        ahash = {'Caucasian':0, 'African American':1, 'Asian':2,
                'African-American':1, 'East Asian':2, 'MULTI-RACIAL':3}
        if (tn == 2):
            group = h.getSurvName("c stimulation")
            atype = [atype[i] if group[i] == 'unstim'
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHuang2015(self, tn=1):
        self.prepareData("LU17")
        atype = self.h.getSurvName("c study")
        ahash = {'ACCURACY STUDY':0, 'EQUIVALENCE STUDY':1, 'PRECISION STUDY':2,
                'SPECIFICITY STUDY':3, 'SENSITIVITY STUDY':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c risk")
        atypes = ['low', 'high']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 2
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getFavreau2008(self, tn=1):
        self.prepareData("COV164")
        atype = self.h.getSurvName("c src1")
        time = [re.sub(".*, (.*)hou.*", "\\1", str(k)) for k in atype]
        ahash = {'48':48, '72':72, '24':24}
        tval = [ahash[i] if i in ahash else None for i in time]
        atype = [re.sub(".*, (.*) infe.*", "\\1", str(k)) for k in atype]
        atypes = ['C', 'I']
        ahash = {'HCoV-OC43':1, 'mock':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 72
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMatualatupauw2019(self, tn=1):
        self.prepareData("MACV124")
        atype = self.h.getSurvName("c day")
        ahash = {'day29':29, 'day1':1}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c time point")
        ahash = {'120min':2, '360min':6, '0min':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease")
        atypes = ['H', 'MetS']
        ahash = {'Healty':0, 'Metabolic Syndrome':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if dval[i] == 1 and tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = [0, 2, 6]
            ahash = {}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = tval
            atypes = [0, 2, 6]
            ahash = {}
            atype = [atype[i] if aval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if tval[i] == 2
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPaczkowskaAbdulsalam2020(self, tn=1):
        self.prepareData("MACV125")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*blood, (.*), sam.*", "\\1", str(k)) for k in atype]
        atypes = ['H, O', 'H, L', 'D, L', 'D, O']
        ahash = {}
        if (tn == 2):
            atypes = ['H, O', 'D, O']
        if (tn == 3):
            atypes = ['H, L', 'D, L']
        self.initData(atype, atypes, ahash)

    def getRouet2018(self, tn=1):
        self.prepareData("HRT1")
        atype = self.h.getSurvName("c subject status")
        atypes = ['C', 'HN', 'HR']
        ahash = {'hypertensive patient with left ventricular remodeling':2,
                'hypertensive patient with normal left ventricular size':1,
                 'control individual':0}
        if (tn == 2):
            atypes = ['C', 'HN']
            ahash = {'hypertensive patient with normal left ventricular size':1,
                     'control individual':0}
        if (tn == 3):
            atypes = ['C', 'HR']
            ahash = {'hypertensive patient with left ventricular remodeling':1,
                     'control individual':0}
        if (tn == 4):
            atypes = ['HN', 'HR']
            ahash = {'hypertensive patient with left ventricular remodeling':1,
                    'hypertensive patient with normal left ventricular size':0}
        self.initData(atype, atypes, ahash)

    def getEsser2018(self, tn=1):
        self.prepareData("HRT2")
        atype = self.h.getSurvName("c treatment")
        ahash = {'placebo':0, 'epicatechin (100mg/d)':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c before/after supplementation")
        atypes = ['B', 'A']
        ahash = {'after':1, 'before':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLi2016Mm(self, tn=1):
        self.prepareData("HRT3")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("D.*", "", str(k)) for k in atype]
        atypes = ['0', 'AngII7', 'AngII28']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getNelson2018Mm(self, tn=1):
        self.prepareData("HRT4")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("BP.*", "", str(k)) for k in atype]
        atypes = ['young', 'old']
        ahash = {}
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub(".*(BP.).*", "\\1", str(k)) for k in atype]
            atypes = ['BPN', 'BPH']
        self.initData(atype, atypes, ahash)

    def getMahata2018(self, tn=1):
        self.prepareData("HRT5")
        atype = self.h.getSurvName("c genotype")
        atypes = ['Wt', 'Chga', 'CST']
        ahash = {'CST knockout':2, 'Wild-type':0, 'CHGA knockout':1}
        self.initData(atype, atypes, ahash)

    def getPeng2020(self, tn=1):
        self.prepareData("HRT6")
        atype = self.h.getSurvName("c group")
        atypes = ['N', 'H', 'Q']
        ahash = {'Hypertension':1, 'QDG Treatment':2, 'Normal':0}
        if (tn == 2):
            atypes = ['N', 'H']
            ahash = {'Hypertension':1, 'Normal':0}
        if (tn == 3):
            atypes = ['H', 'QDG']
            ahash = {'Hypertension':0, 'QDG Treatment':1}
        self.initData(atype, atypes, ahash)

    def getSweeney2015(self, tn=1):
        self.prepareData("MACV130")
        atype = self.h.getSurvName("c disease")
        atypes = ['C', 'S', 'SS', 'SIRS']
        ahash = {'SepticShock':2, 'SIRS':3, 'Sepsis':1, 'Control':0}
        if (tn == 2):
            atypes = ['C', 'SIRS']
            ahash = {'SIRS':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getMcHugh2015(self, tn=1):
        self.prepareData("MACV131.2")
        atype = self.h.getSurvName("c group")
        atypes = ['C', 'S']
        ahash = {'post-surgical':0, 'Sepsis':1}
        self.initData(atype, atypes, ahash)

    def getLance2017(self, tn=1):
        self.prepareData("LP3")
        atype = self.h.getSurvName('c Time               (Weeks)')
        ahash = {'0.142857':0, '1':1, '8':8, '3':3, '6':6, '2':2, '7':7,
                '10':10, '22':22, '23':23, '4':4, '5':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c BPD severity                             ')
        ahash = {'Severe':3, 'Mild':1, 'N/A':4, 'Moderate':2, 'None':0}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Treatment")
        atypes = ['CTRL', 'LPS']
        ahash = {'CTRL':0, 'LPS':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if dval[i] == 2
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName('c BPD severity                             ')
            atypes = ['N', 'S']
            ahash = {'Severe':1, 'Mild':0, 'N/A':0, 'Moderate':0, 'None':0}
            atype = [atype[i] if aval[i] == 0 and tval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getCohen2007(self, tn=1):
        self.prepareData("LP4")
        atype = self.h.getSurvName("c Disease")
        atypes = ['nobpd', 'bpd']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getCai2018(self, tn=1):
        self.prepareData("LP5")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['N', 'B']
        ahash = {'normal':0, 'Bronchopulmonary dysplasia (BPD)':1}
        self.initData(atype, atypes, ahash)

    def getDavidson2013(self, tn=1):
        self.prepareData("LP6")
        atype = self.h.getSurvName("c Treatment")
        atypes = ['L', 'LI', 'LD']
        ahash = {' LPS+IL-10':1, ' LPS+DEX':2, ' LPS':0}
        self.initData(atype, atypes, ahash)

    def getPietrzyk2013(self, tn=1):
        self.prepareData("LP7")
        atype = self.h.getSurvName('c bronchopulmonary dysplasia (bpd) group')
        atypes = ['N', 'Mi', 'Mo', 'S', 'A']
        ahash = {'#N/A!':4, '0 (no BPD)':0, '1 (mild BPD)':1,
                '3 (severe BPD)':3, '2 (moderate BPD)':2}
        if (tn == 2):
            atype = self.h.getSurvName('c retinopathy of prematurity (rop) group')
            atypes = ['N', 'NT', 'LT', 'NA']
            ahash = {'#N/A!':3, '0 (no ROP)':0,
                    '2 (ROP which need laser therapy)':2,
                    '1 (ROP not requiring treatment)':1}
        self.initData(atype, atypes, ahash)

    def getCho2012Mm(self, tn=1):
        self.prepareData("LP8")
        atype = self.h.getSurvName("c treatment")
        ahash = {'air':0, 'hyperoxia':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c genotype variation")
        atypes = ['W', 'Nrf']
        ahash = {'Nrf+/+':0, 'Nrf-/-':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getRittirsch2016(self, tn=1, tb=0):
        self.prepareData("MACV160")
        atype = self.h.getSurvName("c time point")
        atype = [re.sub(".*day (.*)[^0-9]", "\\1", str(k)) for k in atype]
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'0':0, '1':1, '2':2, '3':3, '5':5, '7':7, '10':10, '14':14,
                '21':21, '26':26, '28':28}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c subject subgroup")
        atypes = ['C', 'S']
        ahash = {'patients with systemic inflammation without infection':0,
                'patients with secondary sepsis after trauma':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLoren2014(self, tn=1):
        self.prepareData("MACV161")
        atype = self.h.getSurvName("c patient prognosis")
        atypes = ['G', 'M', 'P']
        ahash = {'Good prognosis':0,
                'Poor prognosis':2,
                'Moderate prognosis':1}
        self.initData(atype, atypes, ahash)

    def getStapels2018(self, tn=1):
        self.prepareData("MACV162")
        atype = self.h.getSurvName("c macrophage subtype")
        atypes = ['U', 'iHK', 'By', 'iG', 'iNG']
        ahash = {'Infected, with growing Salmonella':3,
                'Infected, with non-growing Salmonella':4,
                'Bystander':2,
                'Uninfected':0,
                'Infected, with non-viable Salmonella':1}
        self.initData(atype, atypes, ahash)

    def getStammet2012(self, tn=1):
        self.prepareData("MACV163")
        atype = self.h.getSurvName("c cpc")
        atypes = ['1', '2', '3', '4', '5']
        ahash = {}
        if (tn == 2):
            atypes = ['G', 'B']
            ahash = {'1':0, '2':0, '3':1, '4':1, '5':1}
        self.initData(atype, atypes, ahash)

    def getMoyer2010(self, tn=1):
        self.prepareData("MACV164")
        atype = self.h.getSurvName("c molecular group")
        atypes = ['U', 'I', 'F']
        ahash = {'Fibrosis':2, 'Inflammation':1, 'Unclassified':0}
        self.initData(atype, atypes, ahash)

    def getHunter2010(self, tn=1):
        self.prepareData("MACV165")
        atype = self.h.getSurvName("c outcome at one year")
        atypes = ['P', 'E']
        ahash = {'persistent':0, 'extended':1}
        self.initData(atype, atypes, ahash)

    def getKlapper2008(self, tn=1):
        self.prepareData("MACV166")
        atype = self.h.getSurvName("c Ann Arbour Stage")
        atypes = ['NA', 'I', 'II', 'III', 'IV']
        atype = self.h.getSurvName("c Gene expression")
        atype = [re.sub(".*: ", "", str(k)) for k in atype]
        atypes = ['N', 'I', 'M']
        ahash = {'intermediate':1, 'mBL':2, 'non-mBL':0}
        self.initData(atype, atypes, ahash)

    def getGuthridge2020(self, tn=1):
        self.prepareData("MACV169")
        atype = self.h.getSurvName("c case/control")
        atypes = ['C', 'SLE']
        ahash = {'SLE Case':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getBohne2012I(self, tn=1):
        self.prepareData("MACV170")
        atype = self.h.getSurvName("c immunotolerance group")
        atypes = ['NT', 'TP', 'T']
        ahash = {'TOL POST':1, 'Non TOL':0, 'TOL':2}
        self.initData(atype, atypes, ahash)

    def getBohne2012II(self, tn=1):
        self.prepareData("MACV171")
        atype = self.h.getSurvName("c liver sample group")
        atypes = ['Cont', 'Non-TOL', 'HEPC', 'Non-TOL REJ', 'TOL', 'Cont-Tx', 'REJ']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getZander2011I(self, tn=1):
        self.prepareData("MACV172")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*: ", "", str(k)) for k in atype]
        atypes = ['C', 'BC']
        ahash = {'bronchial carcinoma':1, 'control':0}
        self.initData(atype, atypes, ahash)

    def getZander2011II(self, tn=1):
        self.prepareData("MACV173")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(".*: ", "", str(k)) for k in atype]
        atype = [re.sub("X.*", "X", str(k)) for k in atype]
        atypes = ['C', 'BC', 'X']
        ahash = {'bronchial carcinoma':1, 'control':0}
        self.initData(atype, atypes, ahash)

    def getPennycuick2020(self, tn=1):
        self.prepareData("COV165")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['PA', 'AA', 'PF', 'AF', 'PC', 'AC']
        ahash = {'Paediatric_Airway':0, 'Adult_Airway':1, 'Paediatric_FACS':2,
                'Adult_FACS':3, 'Paediatric_Cultured':4, 'Adult_Cultured':5}
        if (tn == 2):
            atypes = ['P', 'A']
            ahash = {'Paediatric_FACS':0, 'Adult_FACS':1,
                    'Paediatric_Cultured':0, 'Adult_Cultured':1}
        if (tn == 3):
            atypes = ['P', 'A']
            ahash = {'Paediatric_Airway':0, 'Adult_Airway':1}
        self.initData(atype, atypes, ahash)

    def getAulicino2018(self, tn=1):
        self.prepareData("MACV174")
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'LT', 'D']
        ahash = {'STM-D23580':2, 'STM-LT2':1, 'Mock':0}
        self.initData(atype, atypes, ahash)

    def getAulicino2018Sc(self, tn=1):
        self.prepareData("MACV174.3")
        atype = self.h.getSurvName("c infection")
        atypes = ['M', 'LT', 'D']
        ahash = {'D23580':2, 'Mock':0, 'LT2':1, 'Blank':0}
        self.initData(atype, atypes, ahash)

    def getAvital2017(self, tn=1):
        self.prepareData("MACV175")
        atype = self.h.getSurvName("c agent")
        atypes = ['N', 'U', 'S', 'D']
        ahash = {'':3, 'none':0, 'Salmonella typhimurium SL1344':2, 'unexposed':1}
        self.initData(atype, atypes, ahash)

    def getYe2018(self, tn=1):
        self.prepareData("MACV181")
        atype = self.h.getSurvName("c condition")
        atypes = ['B', 'InfA', 'IFNB']
        ahash = {'baseline':0, 'influenza stimulated':1, 'IFN-beta stimulated':2}
        self.initData(atype, atypes, ahash)

    def getKash2017(self, tn=1):
        self.prepareData("COV166")
        atype = self.h.getSurvName("c virus strain")
        atypes = ['none', 'Ebola']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getTang2019(self, tn=1):
        self.prepareData("COV167")
        sex = self.h.getSurvName("c Sex")
        ahash = {'f':0, 'm':1}
        sex = [ahash[i] if i in ahash else None for i in sex]
        age = self.h.getSurvName("c age")
        age = [int(age[i]) if i > 1 and age[i] != 'NA'
                else None for i in range(len(age))]
        atype = self.h.getSurvName("c severity")
        atypes = ['C', 'M', 'S']
        ahash = {'flu_mod':1, 'flu_svre':2, 'hlty_ctrl':0}
        if (tn == 2):
            atypes = ['M', 'S']
            ahash = {'flu_mod':0, 'flu_svre':1}
        if (tn == 3):
            atype = [atype[i] if age[i] > 50
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getZhang2020(self, tn=1):
        self.prepareData("COV169")
        atype = self.h.getSurvName("c patient group")
        atypes = ['H', 'CoV']
        ahash = {'mild':1, 'severe':1, 'healthy control':0,
                'severe COVID-19 patient':1}
        if (tn == 2):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
        self.initData(atype, atypes, ahash)

    def getZhang2020Mac(self, tn=1):
        self.prepareData("COV169.2")
        atype = self.h.getSurvName("c patient group")
        atypes = ['H', 'CoV']
        ahash = {'mild':1, 'severe':1, 'healthy control':0,
                'severe COVID-19 patient':1}
        if (tn == 2):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
        self.initData(atype, atypes, ahash)

    def getZhang2020CB(self, tn=1, tb=0):
        self.prepareData("COV169.4")
        atype = self.h.getSurvName("c Cell Type")
        atype = [re.sub("[_-][c1].*", "", str(k)) for k in atype]
        ahash = {'B':0, 'T':1, 'CD4_T':2, 'CD8_T':3, 'Natural_killer':4,
                'Macs_Monos_DCs':5, 'Epithelial':6}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c patient group")
        atypes = ['H', 'CoV']
        ahash = {'mild':1, 'severe':1, 'healthy control':0,
                'severe COVID-19 patient':1}
        if (tn == 2):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
        if (tn == 3):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 4):
            atypes = ['H', 'CoV']
            ahash = {'mild':1, 'severe':1, 'healthy control':0,
                    'severe COVID-19 patient':1}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getZhang2020Epi(self, tn=1):
        self.prepareData("COV169.5")
        atype = self.h.getSurvName("c patient group")
        atypes = ['H', 'CoV']
        ahash = {'mild':1, 'severe':1, 'healthy control':0,
                'severe COVID-19 patient':1}
        if (tn == 2):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
        self.initData(atype, atypes, ahash)

    def getZhang2020CD8(self, tn=1):
        self.prepareData("COV169.6")
        atype = self.h.getSurvName("c patient group")
        atypes = ['H', 'CoV']
        ahash = {'mild':1, 'severe':1, 'healthy control':0,
                'severe COVID-19 patient':1}
        if (tn == 2):
            atypes = ['H', 'M', 'S']
            ahash = {'mild':1, 'severe':2, 'healthy control':0,
                    'severe COVID-19 patient':2}
        self.initData(atype, atypes, ahash)

    def getButler2011(self, tn=1, ta=0):
        self.prepareData("COV170")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'alveolar':0, 'airway':1}
        self.tissue = [ahash[i] if i in ahash else None for i in atype]
        v1 = self.h.getSurvName("c Age")
        v2 = self.h.getSurvName("c age")
        atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                for i in range(len(v1))]
        self.age = [int(atype[i]) if i > 1 else None for i in range(len(atype))]
        v1 = self.h.getSurvName("c Ancestry")
        v2 = self.h.getSurvName("c ancestry")
        v3 = self.h.getSurvName("c Ethnic group")
        v4 = self.h.getSurvName("c ethnic group")
        atype = [ "".join([str(k) for k in [v1[i], v2[i], v3[i], v4[i]]])
                                        for i in range(len(v1))]
        ahash = {'African':1, 'hispanic':2, 'white':0,
                'black':1, 'European':0, 'Hispanic':2}
        self.race = [ahash[i] if i in ahash else None for i in atype]
        v1 = self.h.getSurvName("c Sex")
        v2 = self.h.getSurvName("c sex")
        self.sex = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                for i in range(len(v1))]
        v1 = self.h.getSurvName("c smoking status")
        v2 = self.h.getSurvName("c Smoking Status")
        v3 = self.h.getSurvName("c Smoking status")
        atype = [ "".join([str(k) for k in [v1[i], v2[i], v3[i]]])
                                for i in range(len(v1))]
        atype = [re.sub(".*, ", "", str(k)) for k in atype]
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atype = [re.sub("non-smoker", "0", str(k)) for k in atype]
        self.packs = [float(atype[i]) if i > 1 else None for i in range(len(atype))]
        atype = [ "".join([str(k) for k in [v1[i], v2[i], v3[i]]])
                                for i in range(len(v1))]
        self.ss = [re.sub(",.*", "", str(k)) for k in atype]
        atype = self.ss
        atypes = ['NS', 'S']
        ahash = {'non-smoker':0, 'smoker':1}
        if (tn == 2):
            atype = ['Y' if self.age[i] is not None and 
                    self.age[i] < 40 else 'O'
                    for i in range(len(atype))]
            atype = [atype[i] if self.age[i] is not None
                    else None for i in range(len(atype))]
            atypes = ['Y', 'O']
            ahash = {}
        if (tn == 3):
            atype = self.race
            atypes = ['W', 'B', 'H']
            ahash = {0:0,1:1, 2:2}
        if (tn == 4):
            atype = self.sex
            atypes = ['F', 'M']
            ahash = {}
        if (tn == 5):
            atype = self.race
            atypes = ['W', 'B']
            ahash = {0:0, 1:1}
        atype = [atype[i] if self.tissue[i] == ta
                else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getTilley(self, tn=1):
        if ("c ethnicity" in self.h.survhdrs):
            v1 = self.h.getSurvName("c ethnicity")
            v2 = self.h.getSurvName("c ethnic group")
            self.eg = [ "".join([str(k) for k in [v1[i], v2[i]]])
                    for i in range(len(v1))]
        else:
            self.eg = self.h.getSurvName("c ethnic group")
        atype = self.h.getSurvName("c smoking status")
        atype = [re.sub(", .*", "", str(k)) for k in atype]
        atypes = ['NS', 'S']
        ahash = {'smoker':1, 'S':1, 'nonsmoker':0, 'NS':0, 'non-smoker':0}
        self.ss = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 6):
            atype = self.h.getSurvName("c copd status")
            atypes = ['H', 'COPD']
            ahash = {'':0, 'yes':1}
        if (tn == 5):
            atype = self.eg
            atypes = ['W', 'B']
            ahash = {'Afr':1, 'Eur':0, 'black':1, 'white':0}
        if (tn == 3):
            atype = self.eg
            atypes = ['His', 'W', 'B', 'As']
            ahash = {'Afr':2, 'Eur':1, 'black':2, 'hispanic':0, 'white':1,
                    'asian':3}
        if (tn == 2):
            atype = self.h.getSurvName("c age")
            self.age = [int(atype[i]) if i > 1 and atype[i] != ''
                    else None for i in range(len(atype))]
            atype = ['Y' if  self.age[i] is not None and self.age[i] < 40
                    else 'O' for i in range(len(atype))]
            atype = [atype[i] if self.age[i] is not None
                    else None for i in range(len(atype))]
            atypes = ['Y', 'O']
            ahash = {}
        if (tn == 4):
            atype = self.h.getSurvName("c sex")
            atypes = ['F', 'M']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getTilley2016(self, tn=1):
        self.prepareData("COV171")
        self.getTilley(tn)

    def getWang2010(self, tn=1):
        atype = self.h.getSurvName("c Age")
        if ("c age" in self.h.survhdrs):
            v1 = atype
            v2 = self.h.getSurvName("c age")
            atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        self.age = [int(atype[i]) if i > 1 and atype[i] != ''
                else None for i in range(len(atype))]
        atype = self.h.getSurvName("c Ethnic group")
        if ("c ethnic group" in self.h.survhdrs):
            v1 = atype
            v2 = self.h.getSurvName("c ethnic group")
            atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        if ("c Ethnicity" in self.h.survhdrs):
            v1 = atype
            v2 = self.h.getSurvName("c Ethnicity")
            atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        ahash = {'white':1, 'hispanic':0, 'black/hispanic':2,
                'black':2, 'hispnaic':0, 'asian':3}
        self.eg = [ahash[i] if i in ahash else None for i in atype]
        self.sex = self.h.getSurvName("c Sex")
        if ("c sex" in self.h.survhdrs):
            v1 = self.sex
            v2 = self.h.getSurvName("c sex")
            self.sex = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        if ("c Gender" in self.h.survhdrs):
            v1 = self.sex
            v2 = self.h.getSurvName("c Gender")
            ahash = {'':'', 'Male':'M', 'male':'M'}
            v2 = [ahash[i] if i in ahash else None for i in v2]
            self.sex = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        atype = ["" for k in atype]
        if "c Smoking Status" in self.h.survhdrs:
            atype = self.h.getSurvName("c Smoking Status")
        if "c smoking status" in self.h.survhdrs:
            v1 = atype
            v2 = self.h.getSurvName("c smoking status")
            atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        if "c Smoking status" in self.h.survhdrs:
            v1 = atype
            v2 = self.h.getSurvName("c Smoking status")
            atype = [ "".join([str(k) for k in [v1[i], v2[i]]])
                                    for i in range(len(v1))]
        self.ss = atype
        atype = [re.sub(".*, ", "", str(k)) for k in atype]
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atype = [re.sub("non-smoker", "0", str(k)) for k in atype]
        self.packs = [float(atype[i]) if i > 1 and atype[i] != ''
                else None for i in range(len(atype))]
        atype = self.ss
        self.ss = [re.sub(",.*", "", str(k)) for k in atype]
        atype = self.ss
        atypes = ['NS', 'S']
        ahash = {'non-smoker':0, 'smoker':1}
        if (tn == 5):
            atype = self.eg
            atypes = ['W', 'B']
            ahash = {1:0, 2:1}
        if (tn == 4):
            atype = self.sex
            atypes = ['F', 'M']
            ahash = {}
        if (tn == 3):
            atype = self.eg
            atypes = ['His', 'W', 'B', 'As']
            ahash = {0:0, 1:1, 2:2, 3:3}
        if (tn == 2):
            atype = ['Y' if self.age[i] is not None and self.age[i] < 40
                    else 'O' for i in range(len(atype))]
            atype = [atype[i] if self.age[i] is not None
                    else None for i in range(len(atype))]
            atypes = ['Y', 'O']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getWang2010I(self, tn=1):
        self.prepareData("COV172")
        self.getWang2010(tn)
    def getWang2010II(self, tn=1):
        self.prepareData("COV173")
        self.getWang2010(tn)
    def getTilley2011(self, tn=1):
        self.prepareData("COV174")
        self.getTilley(tn)
    def getShaykhiev2011(self, tn=1):
        self.prepareData("COV175")
        self.getWang2010(tn)
    def getStruloviciBarel2010(self, tn=1):
        self.prepareData("COV176")
        self.getWang2010(tn)
    def getTuretz2009(self, tn=1):
        self.prepareData("COV177")
        self.getTilley(tn)
    def getCarolan2008(self, tn=1):
        self.prepareData("COV178")
        self.getWang2010(tn)
    def getTilley2009(self, tn=1):
        self.prepareData("COV179")
        self.getWang2010(tn)
    def getCarolan2006I(self, tn=1):
        self.prepareData("COV180")
        self.getWang2010(tn)
    def getCarolan2006II(self, tn=1):
        self.prepareData("COV181")
        self.getWang2010(tn)
    def getCarolan2006III(self, tn=1):
        self.prepareData("COV182")
        self.getWang2010(tn)

    def getAlmansa2012(self, tn=1):
        self.prepareData("MACV151")
        atype = self.h.getSurvName('c Characteristics[DiseaseState]')
        atypes = ['H', 'COPD']
        ahash = {'critical chronic obstructive pulmonary disease':1,
                'normal':0}
        self.initData(atype, atypes, ahash)

    def getBigler2017(self, tn=1):
        self.prepareData("COV162")
        atype = self.h.getSurvName('c cohort')
        atypes = ['H', 'Asthma']
        ahash = {'Healthy, non-smoking':0, 'Severe asthma, non-smoking':1,
                'Severe asthma, smoking':1, 'Moderate asthma, non-smoking':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['NS', 'S']
            ahash = {'Healthy, non-smoking':0, 'Severe asthma, non-smoking':0,
                    'Severe asthma, smoking':1, 'Moderate asthma, non-smoking':0}
        if (tn == 3):
            atype = self.h.getSurvName('c gender')
            atypes = ['F', 'M']
            ahash = {'male':1, 'female':0}
        if (tn == 4):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'AA']
            ahash = {'white_caucasian':0, 'black_african':1}
        if (tn == 5):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'AA', 'A', 'O']
            ahash = {'white_caucasian':0, 'south_asian':2, 'other':3,
                    'black_african':1, 'arabic_north_heritage':3,
                    'south_east_asian':2, 'multiple_races':3, 'east_asian':2,
                    'central_asian':2}
            #atype = [atype[i] if aval[i] == 0
            #        else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKlenerman2020(self, tn=1):
        self.prepareData("COV183")
        atype = self.h.getSurvName('c cirrhosis present')
        atypes = ['C', 'Cir']
        ahash = {'Yes':1, 'No':0}
        self.initData(atype, atypes, ahash)

    def getWyler2020(self, tn=1, ta = 0, tb = 4):
        self.prepareData("COV185")
        atype = self.h.getSurvName('c cell line')
        ahash = {'Caco2':0, 'Calu3':1, 'H1299':2}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c molecule subtype")
        ahash = {'polyA RNA extracted from whole cells':0,
                'total RNA extracted from whole cells':1}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c time point')
        ahash = {'4h':4, '12h':12, '24h':24, '8h':8, '36h':36}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c infection')
        atypes = ['U', 'M', 'CoV1', 'CoV2']
        ahash = {'SARS-CoV-1':2, 'SARS-CoV-2':3, 'mock':1, 'untreated':0}
        if (tn == 2):
            atypes = ['C', 'CoV2']
            ahash = {'SARS-CoV-2':1, 'mock':0, 'untreated':0}
            atype = [atype[i] if gval[i] == ta and tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'CoV2']
            ahash = {'SARS-CoV-2':1, 'mock':0, 'untreated':0}
            atype = [atype[i] if gval[i] == ta
                    else None for i in range(len(atype))]
        if (tn == 4):
            atypes = ['C', 'CoV2']
            ahash = {'SARS-CoV-2':1, 'mock':0, 'untreated':0}
            atype = [atype[i] if gval[i] == 1 and mval[i] == 0
                    and tval[i] >= 12
                    else None for i in range(len(atype))]
        if (tn == 5):
            atypes = ['C', 'CoV2']
            ahash = {'SARS-CoV-2':1, 'mock':0, 'untreated':0}
            atype = [atype[i] if gval[i] == 1 and mval[i] == 0
                    and tval[i] <= 12
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLamers2020(self, tn=1):
        self.prepareData("COV186")
        atype = self.h.getSurvName('c desc')
        atype = [re.sub("Bulk.*d, in ", "", str(k)) for k in atype]
        medium = [re.sub("ion.*", "ion", str(k)) for k in atype]
        ahash = {'differentiation':0, 'expansion':1}
        mval = [ahash[i] if i in ahash else None for i in medium]
        atype = [re.sub("Biological.*", "", str(k)) for k in atype]
        time = [k.split(" ")[2] if len(k.split()) > 2 else '' for k in atype]
        ahash = {'':0, '24':24, '60':60, '72':72}
        tval = [ahash[i] if i in ahash else None for i in time]
        time = [k.split(" ")[2] if len(k.split()) > 2 else '' for k in time]
        atype = [k.split(" ")[5] if len(k.split()) > 5 else '' for k in atype]
        atypes = ['U', 'CoV1', 'CoV2']
        ahash = {'':0, 'SARS-CoV2':2, 'SARS-CoV':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['U', 'CoV2']
            ahash = {'':0, 'SARS-CoV2':1}
        if (tn == 3):
            atypes = ['U', 'CoV1']
            ahash = {'':0, 'SARS-CoV':1}
        if (tn == 4):
            atypes = ['U', 'CoV2']
            ahash = {'':0, 'SARS-CoV2':1}
            atype = [None if tval[i] != 72 and aval[i] == 2
                    else atype[i] for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getVerhoeven2014(self, tn=1):
        self.prepareData("COV187")
        atype = self.h.getSurvName('c tissue')
        ahash = {'lung mucosa':0, 'colonic mucosa':1, 'jejunal mucosa':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c desc')
        atype = [re.sub("Gene.*from ", "", str(k)) for k in atype]
        atype = [re.sub(" mac.*", "", str(k)) for k in atype]
        atype = [k.split(" ")[-1] for k in atype]
        atypes = ['H', 'U', 'T']
        ahash = {'treated':2, 'untreated':1, 'healthy':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHosmillo2020(self, tn=1):
        self.prepareData("COV188")
        atype = self.h.getSurvName('c tissue')
        ahash = {'ileum organoid TI006':0, 'ileum organoid TI365':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['M', 'NoV', 'uNoV']
        ahash = {'HuNoV GII.4 infection 48h':1, 'Mock 48h':0,
                'UV-treated HuNoV GII.4 48h':2}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['M', 'NoV']
            ahash = {'HuNoV GII.4 infection 48h':1, 'Mock 48h':0}
        self.initData(atype, atypes, ahash)

    def getCuadras2002I(self, tn=1):
        self.prepareData("COV189")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'T1', 'T6', 'T12', 'T24']
        ahash = {'Tc 1h':1, 'Tc control':0, 'Tc 6h':2, 'Tc 24h':4, 'Tc 12h':3}
        if (tn == 2):
            atypes = ['C', 'I']
            ahash = {'Tc 1h':0, 'Tc control':0, 'Tc 6h':1, 'Tc 24h':1, 'Tc 12h':1}
        self.initData(atype, atypes, ahash)

    def getCuadras2002II(self, tn=1):
        self.prepareData("COV189.2")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(" .*[^0-9]([0-9]+)h", " \\1h", str(k)) for k in atype]
        atypes = ['C', 'I']
        ahash = {'Infection 1h':0, 'Control 16h':0, 'Control 1h':0, 'Infection 16h':1}
        self.initData(atype, atypes, ahash)

    def getMedigeshi2020(self, tn=1):
        self.prepareData("COV191")
        atype = self.h.getSurvName('c infection status')
        atypes = ['M', 'JEV']
        ahash = {'Japanese encephalitis virus':1, 'mock':0}
        self.initData(atype, atypes, ahash)

    def getHu2013(self, tn=1):
        self.prepareData("MACV146")
        atype = self.h.getSurvName('c group')
        atypes = ['Ctrl', 'RNA-virus','DNA-virus']
        ahash = {'Adenovirus':2, 'Virus-negative Control':0,'HHV6':2,
                'Enterovirus':1,'Rhinovirus':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'I']
            ahash = {'Adenovirus':1, 'Virus-negative Control':0,'HHV6':1,
                    'Enterovirus':1,'Rhinovirus':1}
        if (tn == 3):
            atype = self.h.getSurvName("c ethnicity")
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'White':0, 'Black':1, 'Other':3}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 4):
            atypes = ['HC', 'I']
            ahash = {'Virus-negative Control':0, 'Rhinovirus':1}
        if (tn == 5):
            atypes = ['HC', 'I']
            ahash = {'Virus-negative Control':0, 'Enterovirus':1}
        if (tn == 6):
            atypes = ['HC', 'I']
            ahash = {'Virus-negative Control':0, 'Adenovirus':1}
        if (tn == 7):
            atypes = ['HC', 'I']
            ahash = {'Virus-negative Control':0, 'HHV6':1}
        self.initData(atype, atypes, ahash)

    def getAltman2019I(self, tn=1):
        self.prepareData("COV192")
        atype = self.h.getSurvName('c analysis.visit')
        ahash = {'Visit 0':0, 'Visit 1b':1, 'Visit 1a':2, 'Visit 2a':3, 'Visit 2b':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c csteroid.start.relative.to.visit')
        ahash = {'Before':0, 'After':1, '':2}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c case.or.control.status.original')
        ahash = {'':2, 'Control':0, 'Case':1}
        bval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c viral.type.at.visit')
        ahash = {'Viral':1, 'Non-viral':0, '':2, 'NA':2} 
        mval = [ahash[i] if i in ahash else None for i in atype]
        atypes = [ 'Non-viral', 'Viral']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName('c GLI_RACE')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'Black':1, 'Other':3, 'White':0, 'SE Asian':2}
        self.initData(atype, atypes, ahash)

    def getAltman2019II(self, tn=1):
        self.prepareData("COV192.2")
        atype = self.h.getSurvName('c analysis.visit')
        ahash = {'Visit 0':0, 'Visit 1b':1, 'Visit 1a':2, 'Visit 2a':3, 'Visit 2b':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c csteroid.start.relative.to.visit')
        ahash = {'Before':0, 'After':1, '':2}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c case.or.control.status.v0')
        ahash = {'':2, 'Control':0, 'Case':1}
        bval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c GLI_RACE')
        ahash = {'Black':1, 'Other':3, 'White':0, 'SE Asian':2}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c viral.type.at.visit')
        ahash = {'Viral':1, 'Non-viral':0, '':2, 'NA':2} 
        mval = [ahash[i] if i in ahash else None for i in atype]
        atypes = [ 'Non-viral', 'Viral']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if rval[i] == 0 and sval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName('c GLI_RACE')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'Black':1, 'Other':3, 'White':0, 'SE Asian':2}
            atype = [atype[i] if sval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHealy2020(self, tn=1):
        self.prepareData("COV193")
        atype = self.h.getSurvName('c Title')
        tissue = [re.sub("-.*", "", str(k)) for k in atype]
        ahash = {'Macrophage':0, 'Microglia':1}
        tval = [ahash[i] if i in ahash else None for i in tissue]
        atype = self.h.getSurvName('c treatment')
        atypes = ['Ctrl', 'VitD']
        ahash = {'':0, '100nM calcitriol':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getCasella2019(self, tn=1):
        self.prepareData("COV194")
        atype = self.h.getSurvName('c desc')
        cells = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'WI-38':0, 'IMR-90':1, 'HAECs':2, 'HUVEC':3}
        tval = [ahash[i] if i in ahash else None for i in cells]
        atype = [re.sub("[^ ]*\s+(.*)", "\\1", str(k)) for k in atype]
        atypes = ['Ctrl', 'Sen']
        ahash = {'doxorubicin-induced senescence':1,
                'proliferating control':0, 'control':0,
                'replicative senescence':1, 'IR-induced senescence':1,
                'empty vector control':0, 'oncogene-induced senescence':1,
                'IR treatment':1}
        if (tn == 2):
            atype = self.h.getSurvName('c desc')
            atypes = ['C', 'S']
            ahash = {'WI-38 doxorubicin-induced senescence':1,
                    'WI-38  proliferating control':0,
                    'WI-38 control':0,
                    'WI-38  replicative senescence':1,
                    'WI-38 IR-induced senescence':1,
                    'WI-38 empty vector control':0,
                    'WI-38 oncogene-induced senescence':1}
        if (tn == 3):
            atype = self.h.getSurvName('c desc')
            atypes = ['PC', 'IR', 'S']
            ahash = {'IMR-90 proliferating control':0,
                    'IMR-90 IR treatment':1,
                    'IMR-90  replicative senescence':2}
        if (tn == 4):
            atype = self.h.getSurvName('c desc')
            atypes = ['C', 'S']
            ahash = {'HAECs IR-induced senescence':1,
                    'HAECs control':0}
        if (tn == 5):
            atype = self.h.getSurvName('c desc')
            atypes = ['C', 'S']
            ahash = {'HUVEC control':0,
                    'HUVEC IR-induced senescence':1}
        self.initData(atype, atypes, ahash)

    def getNurminen2015(self, tn=1):
        self.prepareData("COV195")
        atype = self.h.getSurvName('c treatment')
        atypes = ['Ctrl', 'VitD']
        ahash = {'vehicle':0, '1,25D':1}
        self.initData(atype, atypes, ahash)

    def getSalamon2014(self, tn=1):
        self.prepareData("COV196")
        atype = self.h.getSurvName('c infection')
        ahash = {'M. tuberculosis H37Rv':1, 'uninfected':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['Ctrl', 'VitD']
        ahash = {'100 nM of 1α,25-dihydroxyvitamin D3':1, 'untreated':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getHeikkinen2011(self, tn=1):
        self.prepareData("COV197")
        atype = self.h.getSurvName('c treatment')
        atypes = ['Ctrl', 'VitD']
        ahash = {'vehicle':0, '1,25D':1}
        self.initData(atype, atypes, ahash)

    def getCosta2019(self, tn=1):
        self.prepareData("COV198")
        atype = self.h.getSurvName('c desc')
        atype = [re.sub("G.*with a ", "", str(k)) for k in atype]
        atype = [re.sub("VDR.*with ", "", str(k)) for k in atype]
        atypes = ['Ctrl', 'VitD', 'Mut', 'MutVitD']
        ahash = {'wild-type 1,25-vitamin D':1,
                'wild-type vehicle':0,
                'null p.Arg30* vehicle':2,
                'null p.Arg30* 1,25-vitamin D':3}
        if (tn == 2):
            atypes = ['Ctrl', 'VitD']
            ahash = {'wild-type 1,25-vitamin D':1, 'wild-type vehicle':0}
        if (tn == 3):
            atypes = ['Mut', 'MutVitD']
            ahash = {'null p.Arg30* vehicle':0, 'null p.Arg30* 1,25-vitamin D':1}
        self.initData(atype, atypes, ahash)

    def getWang2011(self, tn=1):
        self.prepareData("COV199")
        atype = self.h.getSurvName('c src1')
        atypes = ['C', 'VitD', 'tC', 'tVitD']
        ahash = {'No-Testosterone':0,
                '5nM-Testosterone-with100nM-VitD':3,
                'No-Testosterone-withVitD':1,
                '5nM-Testosterone':2}
        if (tn == 2):
            atypes = ['C', 'VitD']
            ahash = {'No-Testosterone':0,
                    'No-Testosterone-withVitD':1}
        if (tn == 3):
            atypes = ['tC', 'tVitD']
            ahash = {'5nM-Testosterone-with100nM-VitD':1, '5nM-Testosterone':1}
        self.initData(atype, atypes, ahash)

    def getCosta2009(self, tn=1):
        self.prepareData("COV200")
        atype = self.h.getSurvName('c Title')
        atypes = ['DR1', 'DR2']
        ahash = {'MCF7 VDr (2) X MCF7 (2)':0, 'MCF7 DR X MCF7':0,
                'MCF7 VDr X MCF7':0, 'MCF7 DRA (2) X MCF7 P38 (2)':1,
                'MCF7 DRA X MCF7 P38':1, 'MCF7 D3RE (2) X MCF7 P38 (2)':1,
                'MCF7 D3RE X MCF7 P38':1}
        self.initData(atype, atypes, ahash)

    def getMartinezSena2020(self, tn=1):
        self.prepareData("COV201")
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'C+VDR', 'VitD']
        ahash = {'control Ad-C + Vehicle':0, 'Ad-VDR + Vehicle':1, 'Ad-VDR + 10nMVitD':2}
        if (tn == 2):
            atypes = ['C', 'VitD']
            ahash = {'Ad-VDR + Vehicle':0, 'Ad-VDR + 10nMVitD':1}
        self.initData(atype, atypes, ahash)

    def getFerrerMayorga(self, tn=1):
        self.prepareData("COV202")
        atype = self.h.getSurvName('c tissue')
        ahash = {'Colon normal fibroblasts':0, 'Colon tumor fibroblasts':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'VitD']
        ahash = {'Vehicle':0, '1,25(OH)2D3':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLiu2015(self, tn=1):
        self.prepareData("COV203")
        atype = self.h.getSurvName('c cell line')
        ahash = {'LNCaP':0, 'MCF-7':1, '':2} #Prostasphere
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['C', 'VitD']
        ahash = {'control for calcitriol':0, '100nM calcitriol':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 2
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getOda2015Mm(self, tn=1):
        self.prepareData("COV204")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'VDR_KO']
        ahash = {'VDR Control (CON) wounded skin':0,
                'VDR knockout (KO) wounded skin':1,
                'VDR knockout (KO) keratinocytes':1,
                'Control (CON) keratinocytes':0}
        self.initData(atype, atypes, ahash)

    def getHangelbroek2019(self, tn=1):
        self.prepareData("COV205")
        atype = self.h.getSurvName('c time of sampling')
        ahash = {'after intervention':1, 'before intervention (baseline)':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c intervention group')
        atypes = ['C', 'VitD']
        ahash = {'Placebo':0, '25-hydroxycholecalciferol (25(OH)D3)':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atypes = ['C', 'VitD']
            ahash = {0:0, 1:1}
            atype = [tval[i] if aval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getmeugnier2018rat(self, tn=1):
        self.prepareData("COV206")
        atype = self.h.getSurvName('c diet')
        atypes = ['C', 'VitDd', 'VitD']
        ahash = {'control diet (1000UI/Kg vit D)':0,
                'Vitamin D depletion diet':1,
                'depletion and repletion of vitamin D in the diet':2}
        if (tn == 2):
            atypes = ['C', 'VitDd']
            ahash = {'control diet (1000UI/Kg vit D)':0,
                    'Vitamin D depletion diet':1}
        if (tn == 3):
            atypes = ['C', 'VitD']
            ahash = {'control diet (1000UI/Kg vit D)':0,
                    'depletion and repletion of vitamin D in the diet':1}
        if (tn == 4):
            atypes = ['C', 'VitD']
            ahash = {'Vitamin D depletion diet':0,
                    'depletion and repletion of vitamin D in the diet':1}
        self.initData(atype, atypes, ahash)

    def getCummings2017(self, tn=1):
        self.prepareData("COV207")
        atype = self.h.getSurvName('c tissue')
        ahash = {"Barrett's esophagus segment":1, 'Normal esophageal squamous mucosa':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c arm')
        ahash = {'Arm A':0, 'Arm B':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c timepoint')
        atypes = ['C', 'VitD']
        ahash = {'T1':1, 'T0':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 and gval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 0 and gval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if tval[i] == 1 and gval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = [atype[i] if tval[i] == 1 and gval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBae2011rat(self, tn=1):
        self.prepareData("COV208")
        atype = self.h.getSurvName('c treatment')
        atypes = ['U', 'EV', 'VitD', 'V', 'EP']
        ahash = {'untreated':0, 'enalapril and vehicle':1, 'paricalcitol':2,
                'vehicle':3, 'enalapril and paricalcitol':4}
        if (tn == 2):
            atypes = ['V', 'VitD']
            ahash = {'paricalcitol':1, 'vehicle':0}
        if (tn == 3):
            atypes = ['V', 'VitD']
            ahash = {'enalapril and vehicle':0, 'enalapril and paricalcitol':1}
        self.initData(atype, atypes, ahash)

    def getDankers2019(self, tn=1):
        self.prepareData("COV209")
        atype = self.h.getSurvName('c treatment')
        atypes = ['V', 'VitD']
        ahash = {'VitD':1, 'vehicle':0}
        self.initData(atype, atypes, ahash)

    def getColdren2006II(self, tn=1):
        self.prepareData("COV211")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['A', 'B']
        ahash = {'H358':0, 'Calu3':0, 'A427':0, 'A549':0, 'H1703':1, 'H1299':1, 'HCC827':1}
        self.initData(atype, atypes, ahash)

    def getLoboda2011II(self, tn=1):
        self.prepareData("COV213")
        atype = self.h.getSurvName('c cell line')
        atypes = ['A', 'B']
        ahash = {'H358':0, 'CALU3':0, 'A427':0, 'A549':0, 'H1703':1, 'H1299':1, 'HCC827':1}
        self.initData(atype, atypes, ahash)

    def getByers2013(self, tn=1):
        self.prepareData("COV215")
        atype = self.h.getSurvName('c cell line')
        atypes = ['A', 'B']
        ahash = {'H358':0, 'Calu-3':0, 'A427':0, 'A549':0, 'H1703':1, 'H1299':1, 'HCC827':1}
        self.initData(atype, atypes, ahash)

    def getAstraZeneca2014(self, tn=1):
        self.prepareData("COV218")
        atype = self.h.getSurvName('c cell line')
        atypes = ['A', 'B']
        ahash = {'H358':0, 'CALU3':0, 'A427':0, 'A549':0, 'CACO2':1, 'H1299':1, 'HCC827':1}
        self.initData(atype, atypes, ahash)

    def getTan2019(self, tn=1):
        self.prepareData("COV219")
        atype = self.h.getSurvName('c cell type')
        ahash = {'CD3-CD56+ NK cells':0, 'Whole PBMC':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['U', 'I']
        ahash = {'12 hour exposure to influenza-infected A549 cells':1,
                '12 hour exposure to non-infected A549 cells':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBoeijen2019(self, tn=1):
        self.prepareData("COV220")
        atype = self.h.getSurvName('c hbv clinical phase')
        atypes = ['C', 'HCV', 'HIV', 'M0', 'M1', 'M2', 'Hep']
        ahash = {'HCV':1, 'HIV-1':2, 'Healthy':0, 'Immune Active':4,
                'HBeAg- Hepatitis':6, 'Immune Control':3, 'Immune Tolerant':5}
        if (tn == 2):
            atypes = ['C', 'HCV']
            ahash = {'HCV':1, 'Healthy':0}
        if (tn == 3):
            atypes = ['C', 'HIV']
            ahash = {'HIV-1':1, 'Healthy':0}
        if (tn == 4):
            atypes = ['C', 'HBV']
            ahash = {'Healthy':0, 'Immune Active':1, 'HBeAg- Hepatitis':1,
                    'Immune Control':1, 'Immune Tolerant':1}
        if (tn == 5):
            atypes = ['C', 'Hep']
            ahash = {'Healthy':0, 'HBeAg- Hepatitis':1}
        self.initData(atype, atypes, ahash)

    def getWagstaffe2019(self, tn=1):
        self.prepareData("COV221")
        atype = self.h.getSurvName('c cell fraction')
        ahash = {'NK cell depleted':0, 'NK cell enriched':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c time point')
        atypes = ['B', 'pV']
        ahash = {'30 days post-vaccination':1, 'Baseline (pre-vaccination)':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSantosa2020(self, tn=1):
        self.prepareData("COV222")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Wt', 'Mut']
        ahash = {'LDHA-deficient':1, 'wild-type or LDHA-deficient':0}
        self.initData(atype, atypes, ahash)

    def getTing2020(self, tn=1, tb = 0):
        self.prepareData("COV223")
        atype = self.h.getSurvName('c src1')
        ahash = {'lung':0, 'heart':1, 'jejunum':2, 'liver':3, 'kidney':4, 'bowel':5,
                'fat':6, 'skin':7, 'marrow':8}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c sample case')
        atypes = ['C', 'I']
        ahash = {'1':1, '2':1, '3':1, '4':1, '5':1, 'Control':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [tval[i] if  tval[i] is not None
                    else None for i in range(len(atype))]
            atype = [-1 if  aval[i] is not None and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getHoek2015(self, tn=1, tb = 0, tc=1):
        self.prepareData("COV224")
        atype = self.h.getSurvName('c src1')
        ahash = {'PBMC':0, 'myeloid DC':1, 'Monocytes':2, 'Neutrophils':3,
                'B cells':4, 'T cells':5, 'NK cells':6}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c time')
        ahash = {'0 d':0, '1 d':1, '3 d':3, '7 d':7}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['0 d', '1 d', '3 d', '7 d']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if sval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if sval[i] == tb
                    else None for i in range(len(atype))]
            atypes = ['C', 'pV']
            ahash = {'0 d':0, '1 d':1, '3 d':1, '7 d':1}
        if (tn == 4):
            atype = [atype[i] if sval[i] == tb and
                    (tval[i] == 0 or tval[i] == tc)
                    else None for i in range(len(atype))]
            atypes = ['C', 'pV']
            ahash = {'0 d':0, '1 d':1, '3 d':1, '7 d':1}
        self.initData(atype, atypes, ahash)

    def getXing2014mm(self, tn=1):
        self.prepareData("COV225")
        atype = self.h.getSurvName("c treatment")
        atypes = ['PBS', 'IL15']
        ahash = {'IL15':1, 'PBS- ip':0}
        self.initData(atype, atypes, ahash)

    def getCribbs2018(self, tn=1):
        self.prepareData("COV226")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_.*", "", str(k)) for k in atype]
        atypes = ['DMSO', 'GSK-J4']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getBayo2019(self, tn=1):
        self.prepareData("COV228")
        atype = self.h.getSurvName("c treatment")
        atypes = ['DMSO', 'GSK-J4', 'JIB-04', 'SD-70']
        ahash = {'24 h with DMSO':0,
                '24 h with JIB-04 (1mM)':2,
                '24 h with GSK-J4 (6.2 mM)':1,
                '24 h with SD-70 (2.2 mM)':3}
        if (tn == 2):
            atypes = ['DMSO', 'GSK-J4']
            ahash = {'24 h with DMSO':0, '24 h with GSK-J4 (6.2 mM)':1}
        self.initData(atype, atypes, ahash)

    def getNarang2018(self, tn=1):
        self.prepareData("COV229")
        atype = self.h.getSurvName("c gender")
        ahash = {'Female':0, 'Male':1}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c time point")
        ahash = {'Day 0':0, 'Day 2':2, 'Day 7':7, 'Day 28':28}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['Day 0', 'Day 2', 'Day 7', 'Day 28']
        ahash = {}
        if (tn == 2):
            atype = [sval[i] if tval[i] == 28
                    else None for i in range(len(atype))]
            atypes = ['F', 'M']
            ahash = {0:0, 1:1}
        self.initData(atype, atypes, ahash)

    def getHoang2014(self, tn=1, tb=2):
        self.prepareData("COV238")
        atype = self.h.getSurvName("c timepoint")
        ahash = {'Acute':0, 'Follow Up':1, 'day_28':1, 'day_0':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c severity")
        atypes = ['OFI', 'Mild', 'Severe']
        ahash = {'OFI':0, 'Mild':1, 'Moderate':1, 'Severe':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = tval
            atypes = ['CV', 'AV']
            ahash = {0:1, 1:0}
            atype = [atype[i] if aval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKim2015(self, tn=1, tb=2):
        self.prepareData("COV239")
        atype = self.h.getSurvName("c time point")
        ahash = {'48hrs':48, '2hrs':2, '12hrs':12, '8hrs':8, '36hrs':36,
                '4hrs':4, '72hrs':72, '60hrs':60, '24hrs':24, 'baseline':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c virus infection")
        atypes = ['U', 'IAV', 'RSV', 'IAV+RSV']
        ahash = {'Influenza & Rhino virus infected':3, 'Influenza virus infected':1,
                'Rhino virus infected':2, 'uninfected':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [0 if aval[i] == 0
                    else tval[i] for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getMuramoto2014(self, tn=1, tb=2):
        self.prepareData("COV240")
        atype = self.h.getSurvName("c time")
        ahash = {'Pre':0, '3':3, '1':1, '5':5, '7':7}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c influenza strain")
        atypes = ['0', '8', '9']
        ahash = {'VN30259':2, 'VN3040':0, 'VN3028II':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [tval[i] if aval[i] == tb
                    else None for i in range(len(atype))]
            atypes = sorted(hu.uniq([i for i in atype if i is not None]))
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getDavenport2015(self, tn=1, tb=2):
        self.prepareData("COV241")
        atype = self.h.getSurvName("c timepoint")
        ahash = {'Pre-challenge':0, '48 hours post-challenge':48,
                '12 hours post-challenge':12, '24 hours post-challenge':24}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c vaccination status")
        ahash = {'Control':0, 'Vaccinee':1}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c symptom severity")
        atypes = ['H', 'M', 'S']
        ahash = {'Moderate/severe':2, 'None':0, 'Mild':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 24
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getChang2020(self, tn=1, tb=2):
        self.prepareData("COV242")
        atype = self.h.getSurvName("c clinical _w")
        ahash = {'Yes':1, 'No':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c virus_treatment_status")
        atypes = ['Control', 'RVA', 'RVC']
        ahash = {'Control':0, 'RVA':1, 'RVC':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [tval[i] if aval[i] == 2
                    else None for i in range(len(atype))]
            atypes = ['No', 'Yes']
            ahash = {0:0, 1:1}
        self.initData(atype, atypes, ahash)

    def getHuang2018(self, tn=1):
        self.prepareData("COV243")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'CP', 'AP']
        ahash = {'acute-phase KD':2, 'convalescent-phase KD':1, 'control':0}
        if (tn == 2):
            atypes = ['C', 'AP']
            ahash = {'acute-phase KD':1, 'control':0}
        if (tn == 3):
            atypes = ['CV', 'AV']
            ahash = {'acute-phase KD':1, 'convalescent-phase KD':0}
        self.initData(atype, atypes, ahash)

    def getPopper2007I(self, tn=1):
        self.prepareData("COV244")
        atype = self.h.getSurvName("c Disease State")
        atypes = ['SA', 'TX', 'A', 'C', 'O']
        ahash = {'':4}
        if (tn == 2):
            atypes = ['Conv', 'SubAcute', 'Acute']
            ahash = {'C':0, 'SA':1, 'A':2}
        if (tn == 3):
            atypes = ['Conv', 'Acute']
            ahash = {'C':0, 'A':1}
        if (tn == 4):
            atypes = ['SubAcute', 'Acute']
            ahash = {'SA':0, 'A':1}
        self.initData(atype, atypes, ahash)

    def getPopper2007II(self, tn=1):
        self.prepareData("COV245")
        atype = self.h.getSurvName("c Phenotype")
        atypes = ['A', 'R', 'D', 'NR', 'N']
        ahash = {}
        if (tn == 2):
            atypes = ['R', 'NR']
        self.initData(atype, atypes, ahash)

    def getWright2018I(self, tn=1):
        self.prepareData("COV246")
        atype = self.h.getSurvName("c category")
        atypes = ['C', 'V', 'B', 'KD', 'U']
        ahash = {'Definite Viral':1, 'Control':0, 'Uncertain':4,
                'Definite Bacterial':2, 'Kawasaki Disease':3}
        if (tn == 2):
            atypes = ['C', 'KD']
            ahash = {'Control':0, 'Kawasaki Disease':1}
        self.initData(atype, atypes, ahash)

    def getWright2018II(self, tn=1):
        self.prepareData("COV247")
        atype = self.h.getSurvName("c category")
        atypes = ['C', 'V', 'B', 'KD', 'U', 'JIA', 'HSP']
        ahash = {'Definite Bacterial':2, 'Control':0, 'Uncertain':4,
                'Kawasaki Disease':3, 'Definite Viral':1}
        if (tn == 2):
            atypes = ['C', 'KD']
            ahash = {'Control':0, 'Kawasaki Disease':1}
        if (tn == 3):
            atypes = ['B', 'KD']
            ahash = {'Definite Bacterial':0, 'Kawasaki Disease':1}
        if (tn == 4):
            atypes = ['V', 'KD']
            ahash = {'Definite Viral':0, 'Kawasaki Disease':1}
        if (tn == 5):
            atypes = ['C', 'V']
            ahash = {'Control':0, 'Definite Viral':1}
        if (tn == 6):
            atypes = ['C', 'B']
            ahash = {'Control':0, 'Definite Bacterial':1}
        if (tn == 7):
            atypes = ['B', 'V']
            ahash = {'Definite Bacterial':0, 'Definite Viral':1}
        if (tn == 8):
            atypes = ['NS', 'S']
            atype = self.h.getSurvName('c Shock: 1=yes, 2=no')
            ahash = {'2.0':0, '1.0':1}
        self.initData(atype, atypes, ahash)

    def getPopper2009(self, tn=1):
        self.prepareData("COV248")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C-dr', 'C-ai', 'C-sf', 'KD']
        ahash = {}
        if (tn == 2):
            atypes = ['C', 'KD']
            ahash = {'C-dr':0}
        self.initData(atype, atypes, ahash)

    def getOkuzaki2017(self, tn=1):
        self.prepareData("COV249")
        atype = self.h.getSurvName("c ivig treatment")
        atypes = ['Pre', 'Post']
        ahash = {'before IVIg treatment':0, 'after IVIg treatment':1}
        if (tn == 2):
            atype = self.h.getSurvName("c gender")
            atypes = ['female', 'male']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getJaggi2018(self, tn=1):
        self.prepareData("COV250")
        atype = self.h.getSurvName("c final condition")
        atypes = ['healthy', 'cKD', 'HAdV', 'inKD', 'GAS', 'GAS/SF']
        ahash = {'healthy':0, 'cKD':1, 'HAdV':2, 'inKD':3, 'GAS':4, 'GAS/SF':5}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['healthy', 'KD']
            ahash = {'inKD':1}
        if (tn == 3):
            atypes = ['healthy', 'cKD']
            ahash = {}
        if (tn == 4):
            atypes = ['healthy', 'inKD']
            ahash = {}
        if (tn == 5):
            atype = self.h.getSurvName("c race")
            atypes = ['W', 'B', 'A', 'H', 'O']
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getHoang2014(self, tn=1):
        self.prepareData("COV251")
        atype = self.h.getSurvName("c ivig")
        ahash = {'Responsive':0, 'Resistant':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c phase")
        ahash = {'Convalescent':0, 'Acute':1}
        hval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c phenotype")
        atypes = ['OFI', 'Mild', 'Moderate', 'Severe']
        ahash = {'OFI':0, 'Mild':1, 'Moderate':2, 'Severe':3}
        pval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = hval
            atypes = ['CV', 'AV']
            ahash = {0:0, 1:1}
        if (tn == 3):
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = tval
            atypes = ['R', 'NR']
            ahash = {0:0, 1:1}
            atype = [atype[i] if pval[i] is not None and pval[i] > 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getOgata2009(self, tn=1):
        self.prepareData("COV252")
        atype = self.h.getSurvName("c time")
        ahash = {'post-treatment':1, 'pre-treatment':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c patient")
        ahash = {'IVIG-resistant patient':1, 'IVIG-responsive patient':0}
        pval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        ahash = {'IVIG':0, 'methylprednisolone (IVMP) + IVIG':1}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['IVIG', 'IVMP+IVIG']
        if (tn == 2):
            atype = pval
            atypes = ['R', 'NR']
            ahash = {0:0, 1:1}
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = pval
            atypes = ['R', 'NR']
            ahash = {0:0, 1:1}
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = tval
            atypes = ['Pre', 'Post']
            ahash = {0:0, 1:1}
        self.initData(atype, atypes, ahash)

    def getFury2009(self, tn=1):
        self.prepareData("COV253")
        atype = self.h.getSurvName("c treatment_category")
        atypes = ['HC', 'R_A', 'R_IVIG', 'NR_A', 'NR_IVIG']
        ahash = {'IVIG responder_Acute':1, 'healthy controls':0,
                'IVIG non responder_Acute':3, 'IVIG non responder_after IVIG':4,
                'IVIG responder_after IVIG':2}
        if (tn == 2):
            atypes = ['HC', 'KD']
            ahash = {'healthy controls':0, 'IVIG responder_Acute':1,
                    'IVIG non responder_Acute':1}
        if (tn == 3):
            atypes = ['Pre', 'Post']
            ahash = {'IVIG responder_Acute':0, 'IVIG responder_after IVIG':1,
                    'IVIG non responder_Acute':0, 'IVIG non responder_after IVIG':1}
        self.initData(atype, atypes, ahash)

    def getRowley2015(self, tn=1):
        self.prepareData("COV254")
        atype = self.h.getSurvName("c disease")
        atypes = ['C', 'tKD', 'uKD']
        ahash = {'untreated Kawasaki Disease':2, 'treated Kawasaki Disease':1, 'Control':0}
        if (tn == 2):
            atypes = ['HC', 'KD']
            ahash = {'Control':0, 'untreated Kawasaki Disease':1}
        if (tn == 3):
            atypes = ['Pre', 'Post']
            ahash = {'untreated Kawasaki Disease':0, 'treated Kawasaki Disease':1}
        self.initData(atype, atypes, ahash)

    def getOCarroll2014(self, tn=1):
        self.prepareData("MACV182")
        atype = self.h.getSurvName("c cell state")
        atypes = ['C', 'A', 'T', 'R']
        ahash = {'Recovery':3, 'control':0, 'LPS Tolerance':2,
                'Acute response to LPS':1}
        if (tn == 2):
            atypes = ['C', 'A', 'T']
            ahash = {'control':0, 'LPS Tolerance':2, 'Acute response to LPS':1}
        self.initData(atype, atypes, ahash)

    def getMages2007(self, tn=1):
        self.prepareData("MACV183")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*with (.*) stim.*", "\\1", str(k)) for k in atype]
        atypes = ['1_0', '0_0', '1_1', '0_1']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getFoster2007(self, tn=1):
        self.prepareData("MACV184")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*ages, ", "", str(k)) for k in atype]
        atypes = ['U', 'T', 'TR']
        ahash = {'treated with LPS for 24hours, then restimulated':2,
                'treated with LPS':1,
                 'untreated':0}
        self.initData(atype, atypes, ahash)

    def getUtz2020MmII(self, tn=1):
        self.prepareData("MACV185.2")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(".*- ", "", str(k)) for k in atype]
        atypes = ['Microglia', 'BAM']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getHagemeyer2016Mm(self, tn=1):
        self.prepareData("MACV186")
        atype = self.h.getSurvName("c genotype")
        atypes = ['Wt', 'Irf8']
        ahash = {'Irf8 knockout':1, 'wildtype':0}
        if (tn == 2):
            atype = self.h.getSurvName("c tissue")
            atypes = ['liver', 'brain', 'skin', 'kidney', 'yolk sac']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getPoidinger2018Mm(self, tn=1):
        self.prepareData("MACV187")
        atype = self.h.getSurvName("c src1")
        atypes = ['M6h', 'M6l', 'S']
        ahash = {'microglia Ly6Chigh':0, 'microglia Ly6CLow':1, 'spleen monocytes':2}
        if (tn == 2):
            atypes = ['Ly6C+', 'Ly6C-']
            ahash = {'microglia Ly6Chigh':0, 'microglia Ly6CLow':1}
        self.initData(atype, atypes, ahash)

    def getAvraham2015Mm(self, tn=1, tb=0):
        self.prepareData("MACV188")
        atype = self.h.getSurvName("c time after salmonella exposure")
        ahash = {'0':0, '4':4, '2.5':2.5, '8':8}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c phrodo positive")
        ahash = {'Yes':1, 'No':0}
        pval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c gfp positive")
        atypes = ['GFP-', 'GFP+']
        ahash = {'Yes':1, 'No':0}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = [2 if gval[i] == 1
                else pval[i] for i in range(len(atype))]
        atypes = ['P-', 'P+', 'P+GFP+']
        ahash = {0:0, 1:1, 2:2}
        mval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = tval
            atypes = [0, 2.5, 4, 8]
            ahash = {}
        if (tn == 3):
            atype = tval
            atypes = [0, 2.5, 4, 8]
            ahash = {}
            atype = [atype[i] if mval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDelorey2018Mm(self, tn=1):
        self.prepareData("MACV189")
        atype = self.h.getSurvName("c src1")
        atypes = ['M', 'M-', 'M+']
        ahash = {'macrophage':1, 'macrophage and C. albicans':2, '':0}
        if (tn == 2):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub("_.*", "", str(k)) for k in atype]
            atypes = ['Ct', 'U', 'M', 'CA', 'E']
            ahash = {'Ct':0, 'Uninfected':1, '':2, 'CandidaOnly':3, 'Exposed':4}
        if (tn == 3):
            atypes = ['M-', 'M+']
            ahash = {'macrophage':0, 'macrophage and C. albicans':1}
        if (tn == 4):
            atype = self.h.getSurvName("c Title")
            atype = [re.sub("_.*", "", str(k)) for k in atype]
            atypes = ['Ct', 'U']
            ahash = {'Ct':0, 'Uninfected':1}
        self.initData(atype, atypes, ahash)

    def getWestermann2016Hs(self, tn=1):
        self.prepareData("MACV190")
        atype = self.h.getSurvName("c infection agent")
        atypes = ['M', 'M+1', 'M+2']
        ahash = {'Salmonella typhimurium SL1344':2, 'Salmonella Typhimurium SL1344':2,
                'Salmonella Typhimurium SL1344 delta-pinT':1, '':0}
        self.initData(atype, atypes, ahash)

    def getWestermann2016Pig(self, tn=1):
        self.prepareData("MACV190.2")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(" h repl.*", "", str(k)) for k in atype]
        atype = [re.sub(".*pinT", "pinT", str(k)) for k in atype]
        atype = [re.sub(".*31 ", "", str(k)) for k in atype]
        atypes = ['pinT 00', 'pinT 01', 'pinT 02', 'pinT 06', 'pinT 16',
                'WT 00', 'WT 01', 'WT 02', 'WT 06', 'WT 16', 'mock 01']
        ahash = {}
        if (tn == 2):
            atypes = ['pinT 00', 'pinT 01', 'pinT 02', 'pinT 06', 'pinT 16']
        if (tn == 3):
            atypes = ['WT 00', 'WT 01', 'WT 02', 'WT 06', 'WT 16', 'mock 01']
        self.initData(atype, atypes, ahash)

    def getSanchezCabo2018Mm(self, tn=1):
        self.prepareData("MACV193")
        atype = self.h.getSurvName("c condition")
        atypes = ['Uh', '3', '7', '30']
        ahash = {'uninfarcted heart':0, '3 days post cryoinjury':1,
                '7 days post cryoinjury':2, '30 days post cryoinjury':3}
        self.initData(atype, atypes, ahash)

    def getSu2016tamMm(self, tn=1):
        self.prepareData("MACV194")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_\s*23.*", "", str(k)) for k in atype]
        atypes = ['mWT', 'mK', 'tWt', 'tK']
        ahash = {'KOM_CD206hi':1, 'WT_tumor':2, 'KOM_tumor':3, 'WT_CD206hi':0}
        self.initData(atype, atypes, ahash)

    def getPisu2020Mm(self, tn=1):
        self.prepareData("MACV195")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("M.*", "M", str(k)) for k in atype]
        atypes = ['AM', 'IM']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getChevalier2015Mm(self, tn=1):
        self.prepareData("MACV159")
        atype = self.h.getSurvName("c src1")
        atypes = ['rt', '6C', 'rt+anti', '6C+anti']
        ahash = {'room temp 30 days':0, '6 C 30 days':1,
                'room temp + antibiotics 30 days':2, '6 C + antibiotics 30 days':3}
        self.initData(atype, atypes, ahash)

    def getBlish2020PseudoBulk(self, tn=1):
        self.prepareData("COV255")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBlish2020Mac(self, tn=1):
        self.prepareData("COV255.2")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBlish2020B(self, tn=1):
        self.prepareData("COV255.3")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBlish2020T(self, tn=1):
        self.prepareData("COV255.4")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBlish2020NK(self, tn=1):
        self.prepareData("COV255.5")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getBlish2020Epi(self, tn=1):
        self.prepareData("COV255.6")
        atype = self.h.getSurvName("c sample origin")
        atypes = ['C', 'CoV']
        ahash = {'Patient with confirmed COVID-19':1, 'Healthy control':0}
        self.initData(atype, atypes, ahash)

    def getPG2020Gut(self, tn=1):
        self.prepareData("COV256")
        atype = self.h.getSurvName("c Group")
        atypes = ['C', 'CoV']
        ahash = {'colon-48h':1, 'ileum-72h':1, 'ileum-un':0, 'colon-un':0, 'ileum-48h':1}
        if (tn == 2):
            ahash = {'colon-48h':1, 'colon-un':0}
        if (tn == 3):
            atypes = ['C', '48', '72']
            ahash = {'ileum-72h':2, 'ileum-un':0, 'ileum-48h':1}
        if (tn == 4):
            atypes = ['C', 'CoV']
            ahash = {'ileum-72h':1, 'ileum-un':0, 'ileum-48h':1}
        self.initData(atype, atypes, ahash)

    def getShalova2015(self, tn=1):
        self.prepareData("MACV196")
        atype = self.h.getSurvName("c treatment")
        ahash = {'None':0, 'LPS':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c src1")
        atypes = ['H', 'R', 'S']
        ahash = {'healthy donor':0,
                'patient recovering from sepsis':1,
                'patient with sepsis':2}
        self.initData(atype, atypes, ahash)

    def getAGonzalez2017mm(self, tn=1, tb=0):
        self.prepareData("MACV197")
        atype = self.h.getSurvName("c tissue")
        ahash = {'Spleen':0, 'Bone marrow':1, 'Intestine':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c phenotype")
        atypes = ['NE', 'E']
        ahash = {'Non engulfing':0, 'Engulfing':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getButcher2018mm(self, tn=1, tb=0):
        self.prepareData("MACV200")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(",.*", "", str(k)) for k in atype]
        atypes = ['U', 'PA', 'PT', 'CA', 'CT', 'LA', 'LT', 'IA', 'IT']
        ahash = {'Untreated':0,
                'Pam3csk4 tolerance':2, 'Pam3csk4 acute':1, 'CpG acute':3, 'CpG tolerance':4,
                'LPS acute':5, 'LPS tolerance':6, 'Poly IC acute':7, 'Poly IC tolerance':8}
        if (tn == 2):
            atypes = ['U', 'LA', 'LT']
            ahash = {'Untreated':0, 'LPS acute':1, 'LPS tolerance':2}
        self.initData(atype, atypes, ahash)

    def getBurns2020KD(self, tn=1, tb=0):
        self.prepareData("COV257")
        atype = self.h.getSurvName("c ethnicity")
        ahash = {'0': 'Unknown', '1': 'Asian', '2':'Black/ African American',
                '3': 'Caucasian', '4': 'Hispanic', '6': 'More than one race',
                 '7': 'American Indian/Alaska Native',
                '8': 'Native Hawaiian or Other Pacific Islander', 9: 'Other'}
        rval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease_phase")
        atypes = ['C', 'A']
        ahash = {'Convalescent':0, 'Acute':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        idhash = {'UCSD-3815_C','UCSD-3450','UCSD-3929','UCSD-3291_A','UCSD-3318',
                'UCSD-3573','UCSD-3838','UCSD-3877_C','UCSD-3868_C','UCSD-3826_A',
                'UCSD-3836'}
        if (tn == 2):
            atype = [atype[i] if self.h.headers[i] not in idhash
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if self.h.headers[i] in idhash
                    else None for i in range(len(atype))]
        if (tn == 4):
            btype = self.h.getSurvName("c with_matching_pair")
            atype = [atype[i] if btype[i] == 'yes'
                    else None for i in range(len(atype))]
        if (tn == 5):
            atypes = ['C', 'A1', 'A2', 'A4']
            btype = self.h.getSurvName("c CA status")
            ahash = {'4':3, '2':2, '1':1}
            atype = [ahash[btype[i]] if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {'Convalescent':0, 'Acute':1, 0:0, 1:1, 2:2, 3:3}
        if (tn == 6):
            atypes = ['W', 'B', 'A', 'H']
            atype = rval
            ahash = {'Caucasian':0, 'Black/ African American':1, 'Asian':2, 
                    'Hispanic':3}
        if (tn == 7):
            atype = self.h.getSurvName("c illday")
            ahash = {'10':1}
            dval = [ahash[i] if i in ahash else None for i in atype]
            atypes = ['A2', 'A4']
            btype = self.h.getSurvName("c CA status")
            ahash = {'4':3, '2':2, '1':1}
            atype = [ahash[btype[i]] if aval[i] == 1 and dval[i] is None
                    else atype[i] for i in range(len(atype))]
            ahash = {2:0, 3:1}
        self.initData(atype, atypes, ahash)

    def getVallveJuanico2019(self, tn=1, tb=0):
        self.prepareData("MACV202")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_[0-9]*$", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M1_Ctrl':1, 'M1_Endo':1, 'M2_Ctrl':2, 'M2_Endo':2}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'M1_Ctrl':1, 'M2_Ctrl':2}
        if (tn == 3):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'M1_Endo':1, 'M2_Endo':2}
        self.initData(atype, atypes, ahash)

    def getCader2020Mm(self, tn=1, tb=0):
        self.prepareData("MACV203")
        atype = self.h.getSurvName("c phenotype")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M0':0, 'M1 polarised':1, 'M2 polarised':2}
        self.initData(atype, atypes, ahash)

    def getWang2019(self, tn=1, tb=0):
        self.prepareData("MACV204")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub(", do.*", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2']
        ahash = {'IFNy/LPS':1, 'IL4':2, 'control media':0}
        if (tn == 2):
            ahash = {'UTD, IL13':2, 'UTD, IL4':2, 'UTD, media':0}
        if (tn == 3):
            ahash = {'CAR, IL13':2, 'CAR, IL4':2, 'CAR, media':0}
        self.initData(atype, atypes, ahash)

    def getOConnell2019Mm(self, tn=1, tb=0):
        self.prepareData("MACV205")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2']
        ahash = {'RAW media':0, 'RAW M1':1, 'RAW M2':2,
                'BMM media':0, 'BMM M1':1, 'BMM M2':2}
        if (tn == 2):
            ahash = {'RAW media':0, 'RAW M1':1, 'RAW M2':2}
        if (tn == 3):
            ahash = {'BMM media':0, 'BMM M1':1, 'BMM M2':2}
        self.initData(atype, atypes, ahash)

    def getDiazJimenez2020I(self, tn=1, tb=0):
        self.prepareData("MACV206")
        atype = self.h.getSurvName("c cell type")
        atypes = ['Mono', 'Mac']
        ahash = {'Monocytes':0, 'macrophages-derived monocyte':1}
        if (tn == 2):
            atype = self.h.getSurvName("c agent")
            atypes = ['V', 'D']
            ahash = {'Vehicle':0, 'Dex':1}
        self.initData(atype, atypes, ahash)

    def getDiazJimenez2020II(self, tn=1, tb=0):
        self.prepareData("MACV206.2")
        atype = self.h.getSurvName("c cell subtype")
        atypes = ['Mono', 'Mac']
        ahash = {'Cell line':0, 'Cell line diffferentited':1}
        if (tn == 2):
            atype = self.h.getSurvName("c treatment")
            atypes = ['V', 'D']
            ahash = {'Vehicle':0, 'Dex treatment':1}
        self.initData(atype, atypes, ahash)

    def getGrossVered2020Mm(self, tn=1, tb=0):
        self.prepareData("MACV207")
        atype = self.h.getSurvName("c src1")
        ahash = {'Blood':0, 'Colon':1, 'Ileum':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c genotype")
        ahash = {'B6.FVB-Tg[Itgax-DTR/GFP]57Lan/J':0, 'Cx3cr1-DTR':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c selection marker")
        ahash = {'CD45+CD11b+CD115+Ly6C+/-':0,
                'DAPI-CD45+CD11b+CD64+Ly6C-MHCII+':1}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        ahash = {'N/A':0, '9-18 ng DT/gr bodyweight':1}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atypes = ['U', 'T']
        if (tn == 2):
            atype = tval
            atypes = ['B', 'C', 'I']
            ahash = {0:0, 1:1, 2:2}
        self.initData(atype, atypes, ahash)

    def getGopinathan2017(self, tn=1, tb=0):
        self.prepareData("MACV208")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_3h.*$", "", str(k)) for k in atype]
        atypes = ['U', 'Nm', 'IL10', 'Nm+Il10', 'm-Il10', 'p']
        ahash = {'Monocytes_unstimulated':0, 'Monocytes_IL-10':2, 'Monocytes_Nm':1,
                'Monocytes__mild_meningococcemia_plasma_immunodepleted for IL-10':4,
                'Monocytes_Nm+IL-10':3,
                'Monocytes__meningococcal_sepsis_plasma_immunodepleted_for_IL-10':4,
                'Monocytes__mild_meningococcemia_plasma':5,
                'Monocytes__meningitis_plasma_immunodepleted for IL-10':4,
                'Monocytes__meningococcal_sepsis_plasma':5,
                'Monocytes__meningitis_plasma':5}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'Monocytes_unstimulated':0, 'Monocytes_IL-10':2, 'Monocytes_Nm':1}
        self.initData(atype, atypes, ahash)

    def getSerazin2020(self, tn=1, tb=0):
        self.prepareData("MACV209")
        atype = self.h.getSurvName("c cell type")
        atypes = ['Mono', 'Mac']
        ahash = {'monocytes':0, 'macrophages':1}
        if (tn == 2):
            atype = self.h.getSurvName("c src1")
            atypes = ['M0', 'M1', 'M2', 'M-IL34', 'M-MCSF']
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getBlack2018(self, tn=1, tb=0):
        self.prepareData("MACV210")
        atype = self.h.getSurvName("c cell type")
        atypes = ['M', 'Mc', 'Mnc']
        ahash = {'Monocyte':0,
                'MonocyteNonclassical':2,
                'MonocyteClassical':1}
        self.initData(atype, atypes, ahash)

    def getMuhitch2019(self, tn=1, tb=0):
        self.prepareData("MACV211")
        atype = self.h.getSurvName("c src1")
        atypes = ['Mc', 'Mnc', 'DCc', 'DCnc']
        ahash = {'non-classical monocyte derived dendritic cell':3,
                'classical monocyte derived dendritic cell':2,
                'non-classical monocytes':1,
                'classical monocytes':0}
        self.initData(atype, atypes, ahash)

    def getMuhitch2019MonoI(self, tn=1, tb=0):
        self.prepareData("MACV212")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'RCC']
        ahash = {'control':0, 'renal cell carcinoma (RCC)':1}
        self.initData(atype, atypes, ahash)

    def getMuhitch2019MonoII(self, tn=1, tb=0):
        self.prepareData("MACV213")
        atype = self.h.getSurvName("c monocyte subset")
        ahash = {'intermediate':1, 'classical':0, 'non-classical':2}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'RCC']
        ahash = {'renal cell carcinoma':1, 'healthy':0}
        if (tn == 2):
            atype = [atype[i] if mval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getXu2019(self, tn=1, tb=0):
        self.prepareData("MACV215")
        atype = self.h.getSurvName("c cell type")
        atypes = ['Mc', 'Mi', 'Mnc']
        ahash = {'Classical monocytes':0,
                'Intermediate monocytes':1,
                'Non classical monocytes':2}
        self.initData(atype, atypes, ahash)

    def getBouchlaka2017(self, tn=1, tb=0):
        self.prepareData("MACV216")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(" Mac.*", "", str(k)) for k in atype]
        atypes = ['BM', 'BL', 'MSC']
        ahash = {'Blood Derived':1, 'MSC Educated':2, 'BM Derived':0}
        self.initData(atype, atypes, ahash)

    def getTausendschon2015(self, tn=1, tb=0):
        self.prepareData("MACV217")
        atype = self.h.getSurvName("c src1")
        atype = [re.sub("hu.* \(", "", str(k)) for k in atype]
        atype = [re.sub("\) .*IL-10", " IL10", str(k)) for k in atype]
        atype = [re.sub("\)\s*16.*, ", " -IL10 ", str(k)) for k in atype]
        atypes = ['C1', 'C21', 'IL10-1', 'IL10-21']
        ahash = {'si control IL10, 4h 1% oxygen':2,
                'si control IL10, 4h 21% oxygen':3,
                'si control -IL10 4h 1% oxygen':0,
                'si control -IL10 4h 21% oxygen':1}
        self.initData(atype, atypes, ahash)

    def getWhite2012(self, tn=1, tb=0):
        self.prepareData("MACV218")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'LPS', '4F', '4F+LPS']
        ahash = {'control':0, '4F':2, '4F + lipopolysaccharides (LPS)':3,
                'control + lipopolysaccharides (LPS)':1}
        self.initData(atype, atypes, ahash)

    def getJardine2019(self, tn=1, tb=0):
        self.prepareData("MACV219")
        atype = self.h.getSurvName("c cell type")
        tissue = [re.sub(",.*", "", str(k)) for k in atype]
        ahash = {'Blood':0, 'BAL':1}
        tval = [ahash[i] if i in ahash else None for i in tissue]
        ctype = [re.sub(".*, ", "", str(k)) for k in atype]
        ahash = {'Classical Monocyte':0, 'int. Monocyte':1,
                'Non-classical Monocyte':2, 'monocyte-derived DC':3,
                'AM':4, 'DC1':5, 'DC2':6, 'DC3':7, 'DC2/3':8, 'PDC':9}
        sval = [ahash[i] if i in ahash else None for i in ctype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'LPS']
        ahash = {'Healthy':0, 'Saline':0, 'LPS':1}
        if (tn == 2):
            atype = [atype[i] if sval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [sval[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atypes = ['C', 'I', 'NC']
            ahash = {0:0, 1:1, 2:2}
        self.initData(atype, atypes, ahash)

    def getSansom2020(self, tn=1, tb=0):
        self.prepareData("MACV221")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_hh.*$", "", str(k)) for k in atype]
        atype = [re.sub("cLP_", "", str(k)) for k in atype]
        atypes = ['mac_ko', 'mac_wt', 'p1mono_ko', 'p1mono_wt', 'p2mono_ko', 'p2mono_wt']
        ahash = {}
        if (tn == 2):
            atypes = ['mac_wt', 'p1mono_wt', 'p2mono_wt']
        if (tn == 3):
            atypes = ['mac_wt', 'mac_ko']
        self.initData(atype, atypes, ahash)

    def getLi2019Mm(self, tn=1, tb=0):
        self.prepareData("MACV222")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_.*$", "", str(k)) for k in atype]
        atypes = ['M0', 'M1', 'M2', 'lnATM', 'obATM']
        ahash = {}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
        if (tn == 3):
            atypes = ['lnATM', 'obATM']
        self.initData(atype, atypes, ahash)

    def getLi2019MmBlk(self, tn=1, tb=0):
        self.prepareData("MACV222.2")
        atype = self.h.getSurvName("c Title")
        atypes = ['M0', 'M1', 'M2', 'P']
        ahash = {'M012':3,
                'A0.0':1, 'A0.1':1, 'A0.2':1, 'A0.3':1, 'A0.4':1,
                'A0.5':2, 'A0.6':2, 'A0.7':2, 'A0.8':2, 'A0.9':2, 'A1.0':2}
        self.initData(atype, atypes, ahash)

    def getRealegeno2016(self, tn=1, tb=0):
        self.prepareData("MACV223")
        atype = self.h.getSurvName("c stimulation")
        atypes = ['C', 'IFNG', 'TLR2/1']
        ahash = {'media alone':0, 'TLR2/1 ligand':2, 'interferon gamma':1}
        self.initData(atype, atypes, ahash)

    def getOkuzaki2020(self, tn=1, tb=0):
        self.prepareData("COV258")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub("_.*", "", str(k)) for k in atype]
        atypes = ['A549', 'lungNHBE', 'lungORG', 'V', 'VC', 'Ctl']
        ahash = {}
        if (tn == 2):
            atypes = ['C', 'CoV']
            ahash = {'V':1, 'Ctl':0}
        self.initData(atype, atypes, ahash)

    def getBrykczynska2020(self, tn=1, tb=0):
        self.prepareData("MACV224")
        atype = self.h.getSurvName("c tissue")
        ahash = {'adipose tissue macrophages':0, 'monocytes':1, 'microglia':2,
                'colonic macrophages':3, 'islet macrophages':4, 'Kupffer cells':5,
                'peritoneal macrophages':6}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['N', 'HFD', 'STZ', 'F', 'RE']
        ahash = {'fasted':3, 'refed':4, 'normal_diet':0, 'high_fat_diet_with_STZ':2,
                'high_fat_diet':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLavin2014(self, tn=1, tb=0):
        self.prepareData("MACV225")
        atype = self.h.getSurvName("c cell type")
        atypes = ['Nu', 'Mono', 'B', 'PC', 'Sp', 'Li', 'Co', 'Il', 'Lu']
        ahash = {'Neutrophils':0, 'Monocytes':1, 'Brain microglia':2,
                'Peritoneal cavity macrophages':3, 'Spleen red pulp macrophages':4,
                'Kupffer cells':5, 'Large intestine macrophages':6,
                'Small intestine macrophages':7, 'Lung macrophages':8}
        self.initData(atype, atypes, ahash)

    def getHaney2018(self, tn=1, tb=0):
        self.prepareData("MACV226")
        atype = self.h.getSurvName("c protocol")
        atypes = ['U', 'D', 'Uc', 'Dc']
        ahash = {'Undifferentiated GFP':0, 'Undifferentiated Ctrl sgRNA':2,
                'Differentiated GFP':1, 'Differentiated Ctrl sgRNA':3}
        self.initData(atype, atypes, ahash)

    def getDutertre2019(self, tn=1, tb=0):
        self.prepareData("MACV227.2")
        atype = self.h.getSurvName("c disease state")
        atypes = ['H', 'SLE', 'Ss', 'NA']
        ahash = {'Healthy':0, 'NA':3, 'SLE':1, 'Systemic sclerosis':2}
        if (tn == 2):
            atype = self.h.getSurvName("c cell type")
            atypes = ['5+', '5-163-', '163+14-', '163+14+']
            ahash = {'cDC2_CD5+':0, 'cDC2_CD5-CD163-':1,
                    'cDC2_CD5-CD163+CD14-':2, 'cDC2_CD5-CD163+CD14+':3}
        self.initData(atype, atypes, ahash)

    def getBeins2016I(self, tn=1, tb=0):
        self.prepareData("MACV228")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2', 'TGFb']
        ahash = {'TGFb':3, 'none':0, 'LPS/IFNg':1, 'IL-4':2}
        self.initData(atype, atypes, ahash)

    def getBeins2016II(self, tn=1, tb=0):
        self.prepareData("MACV228.2")
        atype = self.h.getSurvName("c treatment")
        atypes = ['M0', 'M1', 'M2', 'TGFb']
        ahash = {'TGFb':3, 'none':0, 'LPS/IFNg':1, 'IL-4':2}
        self.initData(atype, atypes, ahash)

    def getBeins2016III(self, tn=1, tb=0):
        self.prepareData("MACV228.3")
        atype = self.h.getSurvName("c stimulation")
        atypes = ['M0', 'M1', 'M2', 'TGFb']
        ahash = {'unstimulated':0, 'LPS+IFNg':1, 'IL4':2, 'TGFb':3}
        self.initData(atype, atypes, ahash)

    def getRossi2018Mm(self, tn=1, tb=0):
        self.prepareData("MACV229")
        atype = self.h.getSurvName("c Title")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['GFP', 'IL4']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getWel2020(self, tn=1, tb=0):
        self.prepareData("MACV230")
        atype = self.h.getSurvName("c genotype/variation")
        atypes = ['WT', 'FES']
        ahash = {'Wild-type':0, 'FES_S700C mutant':1}
        self.initData(atype, atypes, ahash)

    def getSingh2016(self, tn=1, tb=0):
        self.prepareData("MACV231")
        atype = self.h.getSurvName("c src1")
        atypes = ['CD15-', 'CD15+']
        ahash = {'CD15-':0, 'CD15+ cancer stem like cells':1}
        self.initData(atype, atypes, ahash)

    def getJoshi2014(self, tn=1, tb=0):
        self.prepareData("MACV232")
        atype = self.h.getSurvName("c genotype/variation")
        atypes = ['WT', 'Rac2']
        ahash = {'Rac2 -/-':1, 'WT mice':0}
        self.initData(atype, atypes, ahash)

    def getJoshi2020MmI(self, tn=1, tb=0):
        self.prepareData("MACV233")
        atype = self.h.getSurvName("c src1")
        atypes = ['Un', 'LPS', 'IL4', 'TAM', 'TAMSyk']
        ahash = {'BMDMs, IL4-stimulated, Syk flox':2,
                'BMDMs, IL4-stimulated, Syk cre':2,
                'BMDMs, not stimulated, Syk cre':0,
                'BMDMs, LPS-stimulated, Syk flox':1,
                'BMDMs, LPS-stimulated, Syk cre':1,
                'BMDMs, not stimulated, Syk flox':0,
                'Tumor-associated macrophages, not stimulated, Syk flox':4,
                'Tumor-associated macrophages, not stimulated, Syk cre':3}
        if (tn == 2):
            atypes = ['C', 'T']
            ahash = {'Tumor cells, vehicle treated, Syk WT':0,
                    'Tumor cells, SRX3207 treated, Syk WT':1,
                    'Tumor cells, not stimulated, Syk cre':0}
        self.initData(atype, atypes, ahash)

    def getJoshi2020MmII(self, tn=1, tb=0):
        self.prepareData("MACV233.2")
        atype = self.h.getSurvName("c Group")
        atypes = ['WT', 'Syk']
        ahash = {'KO':1, 'WT':0}
        self.initData(atype, atypes, ahash)

    def getJoshi2020MmIII(self, tn=1, tb=0):
        self.prepareData("MACV233.3")
        atype = self.h.getSurvName("c Title")
        atypes = ['WT', 'Syk']
        ahash = {'CD11b_Syk_WT':0, 'CD11b_Syk_KO':1, 'CD45_Syk_KO':1,
                'CD45_Syk_WT':0, 'Tumor_Syk_KO':1, 'Tumor_Syk_WT':0,
                'CD11b_Syk_WT_mac':0, 'CD11b_Syk_KO_mac':1, 'CD45_Syk_KO_mac':1,
                'CD45_Syk_WT_mac':0, 'Tumor_Syk_KO_mac':1, 'Tumor_Syk_WT_mac':0}
        self.initData(atype, atypes, ahash)

    def getRedmond2020(self, tn=1, tb=0):
        self.prepareData("COV259")
        atype = self.h.getSurvName("c tissue")
        ahash = {'hESC Pancreas':0, 'Lung':1, 'Liver Organoid':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infected")
        atypes = ['C', 'CoV']
        ahash = {'Mock':0, 'sars-Cov2':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName('c Title')
            atype = [re.sub(".*SARS", "SARS", str(k)) for k in atype]
            atype = [re.sub(".*Mock", "Mock", str(k)) for k in atype]
            atypes = ['C', 'CoV']
            ahash = {'Mock 1':0, 'Mock 2':0, 'SARS-CoV-2 1':1, 'SARS-CoV-2 2':1}
        if (tn == 4):
            atype = self.h.getSurvName('c Title')
            atype = [re.sub(".*SARS", "SARS", str(k)) for k in atype]
            atype = [re.sub(".*Mock", "Mock", str(k)) for k in atype]
            atypes = ['C', 'CoV']
            ahash = {'Mock 1 R2':0, 'Mock 2 R2':0, 'SARS-CoV-2 1 R2':1, 'SARS-CoV-2 2 R2':1}
        self.initData(atype, atypes, ahash)

    def getHe2020Mm(self, tn=1, tb=0):
        self.prepareData("COV260")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'hACE2']
        ahash = {'control':0, 'Ad5-hACE2-transduced':1}
        self.initData(atype, atypes, ahash)

    def getWang2020Cov(self, tn=1, tb=0):
        self.prepareData("COV262")
        atype = self.h.getSurvName("c infection")
        atypes = ['M', 'CoV']
        ahash = {'SARS-CoV-2':1, 'mock':0}
        self.initData(atype, atypes, ahash)

    def getPG2020HSAE(self, tn=1, tb=0):
        self.prepareData("COV263")
        atype = self.h.getSurvName("c Group")
        atypes = ['M', 'CoV']
        ahash = {'CoV-72':1, 'U':0, 'CoV-48':0}
        if (tn == 2):
            atype = self.h.getSurvName("c Protocol")
            atypes = ['M', 'CoV', '96well']
            ahash = {'72':1, '0':0, '96 well 48':2, '48':1, '96 well 72':2}
        if (tn == 3):
            atype = self.h.getSurvName("c Protocol")
            atypes = ['M', 'CoV']
            ahash = {'72':1, '0':0, '96 well 48':1, '48':1, '96 well 72':1}
        self.initData(atype, atypes, ahash)

    def getJulia2020(self, tn=1, tb=0):
        self.prepareData("COV264")
        atype = self.h.getSurvName("c treatment")
        atypes = ['control', 'abatacept']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getRuppin2020(self, tn=1, tb=0):
        self.prepareData("COV265")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'I']
        ahash = {'Control (mock infection)':0, 'SARS-CoV-2 infection':1}
        if (tn == 2):
            atype = self.h.getSurvName("c src1")
            ahash = {'Vero E6 cells':0, 'SARS-CoV-2-infected Vero E6 cells':1}
        self.initData(atype, atypes, ahash)

    def getChua2020Blk(self, tn=1, tb=0):
        self.prepareData("COV267")
        atype = self.h.getSurvName("c Type")
        ahash = {'NS':0, 'BL':1, 'PB':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c COVID-19 severity')
        atypes = ['C', 'M', 'S']
        ahash = {'':0, 'critical':2, 'moderate':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'CoV']
            ahash = {'':0, 'critical':1, 'moderate':1}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getPG2020LungHs(self, tn=1, tb=0):
        #self.prepareData("covirus1",
        #        cfile="/Users/sataheri/public_html/Hegemon/explore.conf")
        self.prepareData("COV372")
        atype = self.h.getSurvName('c title')
        atype = [str(k).split("_")[2] if len(str(k).split("_")) > 2
                         else None for k in atype]
        ahash = {'48h':48, 'UN':0, '72h':72, 'Un':0}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c type")
        ahash = {'monolayer':0, 'ALI':1, 'organoid_P1':2, 'organoid_P4':2,
                'Tissue_a':3, 'Tissue_b':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        ahash = {'infected':1, 'Uninfected':0}
        atypes = ['Un', 'I']
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = self.h.getSurvName('c title')
            atypes = ['48h', 'UN']
            ahash = {'48h':1, 'UN':0}
#            atype = [atype[i] if tval[i] == 0
#                    else None for i in range(len(atype))]
            atype =['U', '48']
        self.initData(atype, atypes, ahash)
        
        


    def getPG2020LungHam(self, tn=1, tb=0):
        #self.prepareData("covirus1.2",
        #        cfile="/Users/sataheri/public_html/Hegemon/explore.conf")
        self.prepareData("COV373")
        atype = self.h.getSurvName("c title")
        atype = [re.sub("[-_].*", "", str(k)) for k in atype]
        ahash = {}
        atypes = ['UN', '3', '4']
        if (tn == 2):
            atypes = ['UN', '3']
        if (tn == 3):
            atypes = ['3', '4']
        self.initData(atype, atypes, ahash)

    def getHan2020(self, tn=1, tb=0):
        self.prepareData("COV269")
        atype = self.h.getSurvName("c tissue/cell type")
        ahash = {'hPSC_Lung organoid':0, 'adult lung autopsy':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c subject status')
        atypes = ['H', 'CoV']
        ahash = {'healthy':0, 'COVID-19':1}
        if (tn == 2):
            atype = self.h.getSurvName("c treatment")
            atypes = ['M', 'CoV', 'DMSO', 'Im']
            ahash = {'Mock':0, 'SARS-CoV-2':1, 'SARS-CoV-2+imatinib':3,
                    'SARS-CoV-2+DMSO':2}
        if (tn == 3):
            atype = self.h.getSurvName("c treatment")
            atypes = ['M', 'CoV']
            ahash = {'Mock':0, 'SARS-CoV-2':1}
        self.initData(atype, atypes, ahash)

    def getMaayan2020(self, tn=1, tb=0):
        self.prepareData("COV270")
        atype = self.h.getSurvName("c src1")
        ahash = {'A549-ACE2':0, 'Pancreatic organoids':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c drug")
        ahash = {'mock':0, 'DMSO':1, 'Amlodipine':2,
                'Terfenadine':2, 'Loperamide':2, 'Berbamine':2,
                'Trifluoperazine':2, 'RS504395':2, 'RS504393':2, 'RS504394':2}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c sars-cov-2 infected')
        atypes = ['H', 'CoV']
        ahash = {'Yes':1, 'No':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atypes = ['H', 'CoV', 'T']
            atype = [dval[i] if tval[i] == 1
                    else None for i in range(len(atype))]
            ahash = {0:0, 1:1, 2:2}
        if (tn == 5):
            atype = [atype[i] if tval[i] == 0 and (dval[i] == 0 or dval[i] == 1)
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getChen2020(self, tn=1, tb=0):
        self.prepareData("COV271")
        atype = self.h.getSurvName('c Title')
        atypes = ['H', 'CoV']
        ahash = {'GSM4451223':0, 'GSM4451224':1}
        self.initData(atype, atypes, ahash)

    def getLeem2020Mm(self, tn=1, tb=0):
        self.prepareData("COV272")
        atype = self.h.getSurvName('c src1')
        atypes = ['N', 'Il15', 'NI', 'NIL15', 'TC', 'IFN']
        ahash = {'Naive mouse without IL-15':0,
                'Naive mouse with IL-15':1,
                'MCMV-infected mouse without IL-15':2,
                'MCMV-infected mouse with IL-15':3,
                'TC_1_Ct':4,
                'TC_1_IFNr':5}
        self.initData(atype, atypes, ahash)

    def getLangelier2020(self, tn=1, tb=0):
        self.prepareData("COV273")
        atype = self.h.getSurvName('c disease state')
        atypes = ['C', 'CoV', 'V']
        ahash = {'no virus':0, 'SC2':1, 'other virus':2}
        if (tn == 2):
            atypes = ['C', 'CoV']
            ahash = {'no virus':0, 'SC2':1}
        self.initData(atype, atypes, ahash)

    def getJaitovich2020(self, tn=1, tb=0):
        self.prepareData("COV274")
        atype = self.h.getSurvName('c disease state')
        atypes = ['C', 'CoV']
        ahash = {'COVID-19':1, 'non-COVID-19':0}
        self.initData(atype, atypes, ahash)

    def getGeorge2019Mm(self, tn=1, tb=0):
        self.prepareData("MACV234")
        atype = self.h.getSurvName('c injected with')
        atypes = ['PBS', 'LPS', 'IL4']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getNugent2020Mm(self, tn=1, tb=0):
        self.prepareData("MACV235")
        atype = self.h.getSurvName('c cell_type')
        ahash = {'microglia':0, 'other':2, 'astrocyte':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c genotype')
        atypes = ['WT', 'Het', 'M']
        ahash = {'Trem2 +/-':1, 'Trem2 -/-':2, 'Trem2 +/+':0,
                'TREM2 +/+':0, 'TREM2 -/-':2}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 4):
            atype = [atype[i] if tval[i] == 2
                    else None for i in range(len(atype))]
        if (tn == 5):
            atype = tval
            atypes = ['M', 'A', 'O']
            ahash = {0:0, 1:1, 2:2}
        self.initData(atype, atypes, ahash)

    def getGutbier2020(self, tn=1, tb=0):
        self.prepareData("MACV236")
        atype = self.h.getSurvName('c cell type')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'iPSC-derived M0 macrophage':0,
                'iPSC-derived M1 macrophage':1,
                'iPSC-derived M2 macrophage':2,
                'iPSC-derived macrophage progenitors':0,
                'PBMC monocyte':0,
                'iPSC-derived microglia cell':0,
                'PBMC-derived M0 macrophage':0,
                'PBMC-derived M1 macrophage':1,
                'PBMC-derived M2 macrophage':2}
        self.initData(atype, atypes, ahash)

    def getDuan2020(self, tn=1, tb=0):
        self.prepareData("MACV237")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'M1', 'M2', 'V']
        ahash = {'GSM4557108':0, 'GSM4557109':1, 'GSM4557110':3,
                'GSM4557111':2, 'GSM4557112':3}
        self.initData(atype, atypes, ahash)

    def getDuan2020Mac(self, tn=1, tb=0):
        self.prepareData("MACV237.2")
        atype = self.h.getSurvName('c Title')
        atypes = ['V', 'M1', 'M2']
        ahash = {'GSM4557108':0, 'GSM4557109':1, 'GSM4557110':0,
                'GSM4557111':2, 'GSM4557112':0}
        self.initData(atype, atypes, ahash)

    def getDuan2020MacII(self, tn=1, tb=0):
        self.prepareData("MACV237.3")
        atype = self.h.getSurvName('c Title')
        atypes = ['V', 'M1', 'M2']
        ahash = {'M0':0, 'A0.0':1, 'A0.1':1, 'A0.2':1, 'A0.3':1, 'A0.4':1,
                'A0.5':2, 'A0.6':2, 'A0.7':2, 'A0.8':2, 'A0.9':2, 'A1.0':2}
        self.initData(atype, atypes, ahash)

    def getDuan2020MacIII(self, tn=1, tb=0):
        self.prepareData("MACV237.4")
        atype = self.h.getSurvName('c Title')
        atypes = ['M0', 'M1', 'M2']
        ahash = {'M012':0,
                'A0.0':1, 'A0.1':1, 'A0.2':1, 'A0.3':1, 'A0.4':1,
                'A0.5':2, 'A0.6':2, 'A0.7':2, 'A0.8':2, 'A0.9':2, 'A1.0':2,
                'B0.0':1, 'B0.1':1, 'B0.2':1, 'B0.3':1, 'B0.4':1,
                'B0.5':2, 'B0.6':2, 'B0.7':2, 'B0.8':2, 'B0.9':2, 'B1.0':2}
        self.initData(atype, atypes, ahash)

    def getDuan2020Lung(self, tn=1, tb=0):
        self.prepareData("MACV237.5")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'M1', 'M2', 'V']
        ahash = {'GSM4557108':0, 'GSM4557109':1, 'GSM4557110':3,
                'GSM4557111':2, 'GSM4557112':3}
        self.initData(atype, atypes, ahash)

    def getDuan2020LungII(self, tn=1, tb=0):
        self.prepareData("MACV237.6")
        atype = self.h.getSurvName('c Title')
        atypes = ['C', 'CoV']
        ahash = {'L0':0,
                'A0.0':0, 'A0.1':0, 'A0.2':0, 'A0.3':0, 'A0.4':0,
                'A0.5':1, 'A0.6':1, 'A0.7':1, 'A0.8':1, 'A0.9':1, 'A1.0':1,
                'B0.0':0, 'B0.1':0, 'B0.2':0, 'B0.3':0, 'B0.4':0,
                'B0.5':1, 'B0.6':1, 'B0.7':1, 'B0.8':1, 'B0.9':1, 'B1.0':1}
        self.initData(atype, atypes, ahash)

    def getGurvich2020(self, tn=1, tb=0):
        self.prepareData("MACV238")
        atype = self.h.getSurvName('c cell type')
        atypes = ['M0', 'M1', 'M2', 'O']
        ahash = {'CD14':3, 'M0':0, 'M1':1, 'M2a':2, 'Mreg':3, 'Mreg_UKR':3, 'PCMO':3}
        if (tn == 2):
            atypes = ['M0', 'M1', 'M2']
            ahash = {'M0':0, 'M1':1, 'M2a':2}
        self.initData(atype, atypes, ahash)

    def getNienhold2020(self, tn=1, tb=0):
        self.prepareData("COV276")
        atype = self.h.getSurvName('c diagnosis')
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['C', 'CoV', 'O']
        ahash = {'COVID-19':1, 'Control':0, 'Other':2}
        if (tn == 2):
            atypes = ['C', 'CoV']
            ahash = {'COVID-19':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getSuarez2015(self, tn=1):
        self.prepareData("MACV133")
        atype = self.h.getSurvName("c condition")
        atypes = ['C', 'S']
        ahash = {'COINFECTION':1, 'Healthy Control':0, 'VIRUS':1, 'BACTERIA':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'B', 'V', 'CO']
            ahash = {'COINFECTION':3, 'Healthy Control':0, 'VIRUS':2, 'BACTERIA':1}
        if (tn == 3):
            atype = self.h.getSurvName("c race")
            atypes = ['W', 'B', 'A']
            ahash = {'White':0, 'Black':1, 'Asian':2}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getAhn2015(self, tn=1):
        self.prepareData("MACV142")
        atype = self.h.getSurvName("c pathogen")
        atypes = ['C', 'S']
        ahash = {'-':0,
                'Staphylococcus aureus':1,
                'Escherichia coli':1,
                'Staphylococcus aureus and Streptococcus pneumoniae':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'Ec', 'Sa', 'S']
            ahash = {'-':0,
                    'Staphylococcus aureus':2,
                    'Escherichia coli':1,
                    'Staphylococcus aureus and Streptococcus pneumoniae':3}
        if (tn == 3):
            atype = self.h.getSurvName("c ethnicity")
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'White':0, 'Unknown':3, 'Black':1, 'unknown':3, 'Asian':2,
                    'black':1, 'white':0}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBloom2013(self, tn=1):
        self.prepareData("MACV155")
        atype = self.h.getSurvName("c disease state")
        atypes = ['C', 'TB', 'B', 'P', 'S', 'C']
        ahash = {'TB':1, 'Sarcoid':4, 'Control':0, 'pneumonia':3,
                'Active sarcoidosis':4, 'Non-active sarcoidosis':4,
                'lung cancer':5, 'Active Sarcoid':4, 'Pneumonia':3,
                'Baseline':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'TB']
            ahash = {'TB':1, 'Control':0}
        if (tn == 3):
            atype = self.h.getSurvName("c ethnicity")
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'Afro-Carribbean':3, 'Indian subcontinent':2,
                    '':3, 'Black':1, 'White':0, 'Caucasian':0,
                    'Middle Eastern':3, 'SE Asian':2, 'Central Asia':2,
                    'None':3}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getYang2017(self, tn=1):
        self.prepareData("MACV252")
        atype = self.h.getSurvName("c asthma")
        atypes = ['C', 'A']
        ahash = {'TRUE':1, 'FALSE':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 3):
            atype1 = self.h.getSurvName('c race_white')
            atype2 = self.h.getSurvName('c race_aa')
            atype3 = self.h.getSurvName('c race_hispanic')
            atype = [" ".join([str(atype1[i]), str(atype2[i]), str(atype3[i])]) \
                    for i in range(len(atype1))]
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'No Yes No':1, 'Yes Yes Yes':3, 'No No Yes':3,
               'Yes No Yes':0, 'Yes Yes No':3, 'No No No':3, 'No Yes Yes':1}
            #atype = [atype[i] if aval[i] == 0
            #        else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLewis2019(self, tn=1):
        self.prepareData("COV159")
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'V']
        ahash = {'VARILRIX':1, 'ENGERIXB3':1, 'PLACEBOB3':0, 'ENGERIXB1':1,
                 'AGRIPPAL':1, 'PLACEBOAB1C':0, 'STAMARIL':1, 'FLUADC':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['C', 'VAR', 'ENG', 'AGR', 'STA', 'FLU']
            ahash = {'VARILRIX':1, 'ENGERIXB3':2, 'PLACEBOB3':0, 'ENGERIXB1':2,
                     'AGRIPPAL':3, 'PLACEBOAB1C':0, 'STAMARIL':4, 'FLUADC':5}
        if (tn == 3):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'white':0, 'asian':2, 'black or african american':1, 'other':3}
            #atype = [atype[i] if aval[i] == 0
            #        else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getObermoser2013(self, tn=1):
        self.prepareData("COV160")
        atype = self.h.getSurvName("c vaccine")
        atypes = ['C', 'I', 'V', 'NS']
        ahash = {'saline':0, 'PNEUM':1, 'Pneumovax':2, 'FLUZONE':2, 'Saline':0,
                'Pneumovax vaccine group':2, 'Influenza vaccine group':2,
                'Flu':1, 'NS':3, 'Influenza':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['C', 'I', 'V']
            ahash = {'saline':0, 'PNEUM':1, 'Pneumovax':2, 'FLUZONE':2, 'Saline':0,
                    'Pneumovax vaccine group':2, 'Influenza vaccine group':2,
                    'Flu':1, 'NS':0, 'Influenza':1}
        if (tn == 3):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'B', 'A']
            ahash = {'Caucasian':0, 'Asian':2, 'African American':1}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMahajan2016(self, tn=1):
        self.prepareData("COV261")
        atype = self.h.getSurvName("c condition")
        atypes = ['C', 'SBI', 'nSBI']
        ahash = {'nonSBI':2, 'SBI':1, 'Healthy Control':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['C', 'I']
            ahash = {'nonSBI':1, 'SBI':1, 'Healthy Control':0}
        if (tn == 3):
            atype = self.h.getSurvName('c race')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'Black or African American':1, 'White':0, 'Stated as Unknown':3,
                    'Asian':2, 'Other':3, 'American Indian or Alaska Native':3,
                    'Native Hawiian or Other Pacific Islander':3}
        self.initData(atype, atypes, ahash)

    def getChaussabel2008(self, tn=1):
        self.prepareData("COV163")
        atype = self.h.getSurvName("c Illness")
        atypes = ['HC', 'I']
        ahash = {'Healthy':0, 'SLE':1, 'UTI':1, 'Bacteremia':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['HC', 'I']
            ahash = {'Healthy':0, 'SLE':1}
        if (tn == 3):
            atype = self.h.getSurvName('c Race')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'(Race Not Reported)':3, 'Black or African American':1,
                    'White':0, 'Other':3, 'Asian':2, 'Selected More than One Race':3}
        self.initData(atype, atypes, ahash)

    def getMiller2014(self, tn=1):
        self.prepareData("MACV262")
        atype = self.h.getSurvName("c sample group")
        atypes = ['HC', 'S']
        ahash = {'caregiver group':1, 'non-stressed control group':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c education")
            atypes = ['L', 'H', 'U', 'O']
            ahash = {'4':2, '1':0, '2':1, '6':3, '3':1, '5':2, 'NA':3, '0':0}
        if (tn == 3):
            atype = self.h.getSurvName('c ethnicity')
            atypes = ['W', 'O']
            ahash = {'Caucasian':0, 'Non-Caucasian':1}
        self.initData(atype, atypes, ahash)

    def getYang2020(self, tn=1):
        self.prepareData("COV279")
        atype = self.h.getSurvName("c diagnosis")
        atypes = ['HC', 'IPF', 'CHP']
        ahash = {'chp':2, 'ipf':1, 'control':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = self.h.getSurvName("c Sex")
            atypes = ['F', 'M']
            ahash = {'male':1, 'female':0}
        if (tn == 3):
            atype = self.h.getSurvName('c race (hispanic;1, black;3,asian;4, white;5, other;6)')
            atypes = ['W', 'B', 'A', 'O']
            ahash = {'5':0, '1':3, '3':1, '4':2, 'UNKNOWN':3, '6':3}
        self.initData(atype, atypes, ahash)

    def getHuang2020(self, tn=1):
        self.prepareData("COV281")
        atype = self.h.getSurvName("c infection status")
        atypes = ['C', 'CoV']
        ahash = {'uninfected':0, 'infected with SARS-CoV-2 MOI 140':1}
        if (tn == 2):
            atype = self.h.getSurvName("c days post infection")
            ahash = {'NA':0, '1':1}
        if (tn == 3):
            atype = self.h.getSurvName("c days post infection")
            ahash = {'NA':0, '4':1}
        self.initData(atype, atypes, ahash)

    def getYouk2020hAO(self, tn=1):
        self.prepareData("COV282")
        atype = self.h.getSurvName("c Timepoint")
        atypes = ['C', 'CoV']
        ahash = {'D0':0, 'D1':1, 'D3':1}
        if (tn == 2):
            ahash = {'D0':0, 'D1':1}
        if (tn == 3):
            ahash = {'D0':0, 'D3':1}
        self.initData(atype, atypes, ahash)

    def getYouk2020hBO(self, tn=1):
        self.prepareData("COV283")
        atype = self.h.getSurvName("c Timepoint")
        atypes = ['C', 'CoV']
        ahash = {'D0':0, 'D1':1, 'D3':1}
        if (tn == 2):
            ahash = {'D0':0, 'D1':1}
        if (tn == 3):
            ahash = {'D0':0, 'D3':1}
        self.initData(atype, atypes, ahash)

    def getVlasovaStLouis2018(self, tn=1):
        self.prepareData("COV290")
        atype = self.h.getSurvName("c group description")
        atypes = ['C', 'E', 'L']
        ahash = {'earlyIRIS':1, 'na':0, 'LateIRIS':2}
        if (tn == 2):
            atype = self.h.getSurvName("c group")
            ahash = {'IRIS':1, 'control':0}
            atypes = ['C', 'I']
        if (tn == 3):
            ahash = {'earlyIRIS':1, 'na':0}
            atypes = ['C', 'I']
        self.initData(atype, atypes, ahash)

    def getZaas2010Mm(self, tn=1):
        self.prepareData("COV294")
        atype = self.h.getSurvName("c infection duration")
        ahash = {'1':1, '2':2, '3':3, '4':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection status")
        atypes = ['H', 'B', 'F']
        ahash = {'candida':2, 'healthy':0, 'staph':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['H', 'F']
            ahash = {'candida':1, 'healthy':0}
            atype = [atype[i] if tval[i] == 2 or aval[i] == 0
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['H', 'B']
            ahash = {'staph':1, 'healthy':0}
            atype = [atype[i] if tval[i] == 2 or aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getLi2019I(self, tn=1):
        self.prepareData("COV295")
        atype = self.h.getSurvName("c disease state")
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['Healthy', 'Fungal']
        ahash = {'healthy':0, 'patient':1}
        self.initData(atype, atypes, ahash)

    def getDix2015(self, tn=1):
        self.prepareData("COV296")
        atype = self.h.getSurvName("c infection")
        atypes = ['M', 'B', 'F']
        ahash = {'Escherichia coli':1, 'mock':0, 'Staphylococcus aureus':1,
                'Candida albicans':2, 'Aspergillus fumigatus':2}
        if (tn == 2):
            atypes = ['Mock', 'Fungal']
            ahash = {'mock':0, 'Candida albicans':1, 'Aspergillus fumigatus':1}
            ahash = {'mock':0, 'Aspergillus fumigatus':1}
            ahash = {'mock':0, 'Candida albicans':1}
        if (tn == 3):
            atypes = ['Mock', 'B']
            ahash = {'mock':0, 'Escherichia coli':1, 'Staphylococcus aureus':1}
            ahash = {'mock':0, 'Escherichia coli':1}
            ahash = {'mock':0, 'Staphylococcus aureus':1}
        self.initData(atype, atypes, ahash)

    def getSubramani2020Mm(self, tn=1):
        self.prepareData("MACV266")
        atype = self.h.getSurvName("c infection state")
        atypes = ['Mock', 'Fungal']
        ahash = {'mock-infected':0, 'infected':1, 'not infected':0}
        self.initData(atype, atypes, ahash)

    def getBruno2020(self, tn=1):
        self.prepareData("COV297")
        atype = self.h.getSurvName("c stimulus")
        atypes = ['Ctl', 'Fungal']
        ahash = {'calb_man':1, 'calb_bg':1, 'caur_KTClive':1,
                'calb_live':1, 'RPMI':0, 'caur_KTCbg':1, 'caur_KTCman':1}
        if (tn == 2):
            ahash = {'calb_live':1, 'RPMI':0}
        self.initData(atype, atypes, ahash)

    def getDAiuto2015(self, tn=1):
        self.prepareData("COV298")
        atype = self.h.getSurvName("c source_name (ch1)")
        atype = [re.sub("[ -].*", "", str(k)) for k in atype]
        atypes = ['C', 'I']
        ahash = {'HSV':1, 'uninfected':0}
        self.initData(atype, atypes, ahash)

    def getTommasi2020(self, tn=1):
        self.prepareData("COV299")
        atype = self.h.getSurvName("c vzv infection (ch1)")
        atype = [re.sub("[ -].*", "", str(k)) for k in atype]
        atypes = ['C', 'I']
        ahash = {'VZV':1, 'Uninfected':0}
        self.initData(atype, atypes, ahash)

    def getHU2019(self, tn=1):
        self.prepareData("COV300")
        atype = self.h.getSurvName("c source_name (ch1)")
        atypes = ['C', 'I']
        ahash = {'CHB WTC':1, 'CHB YTC':1, 'Healthy':0}
        self.initData(atype, atypes, ahash)

    def getRiou2017(self, tn=1):
        self.prepareData("COV301")
        atype = self.h.getSurvName("c hcmv exposition (ch1)")
        atypes = ['C', 'I']
        ahash = {'No HCMV-specific IgG detected':0,
                'HCMV-specific IgG detected':0,
                'During acute primary HCMV infection':1}
        self.initData(atype, atypes, ahash)

    def getZilliox2007(self, tn=1):
        self.prepareData("COV302")
        atype = self.h.getSurvName("c title2")
        atypes = ['C', 'I']
        ahash = {'Control':0, 'Patient':1}
        self.initData(atype, atypes, ahash)

    def getOberstein2017(self, tn=1):
        self.prepareData("COV303")
        atype = self.h.getSurvName("c treatment (ch1)")
        atypes = ['C', 'I']
        ahash = {'mock':0, 'HCMV-infected':1}
        self.initData(atype, atypes, ahash)

    def getOliver2017(self, tn=1):
        self.prepareData("COV304")
        atype = self.h.getSurvName("c status (ch1)")
        atypes = ['C', 'I']
        ahash = {'Uninfected':0, 'infected':1}
        self.initData(atype, atypes, ahash)

    def getMarkus2014(self, tn=1):
        self.prepareData("COV305")
        atype = self.h.getSurvName("c treatment (ch1)")
        atype = [re.sub("[ -].*", "", str(k)) for k in atype]
        atypes = ['C', 'I']
        ahash = {'infection':1, 'none':0}
        self.initData(atype, atypes, ahash)

    def getYoon2019(self, tn=1):
        self.prepareData("COV306")
        atype = self.h.getSurvName("c ebv status (ch1)")
        atypes = ['C', 'I']
        ahash = {'EBV-negative':0, 'EBV-positive':1}
        self.initData(atype, atypes, ahash)

    def getAkinci2020(self, tn=1, ta=0):
        self.prepareData("COV307")
        atype = self.h.getSurvName("c tissue")
        ahash = {'Colon':0, 'Liver':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atype = [re.sub(",.*", "", str(k)) for k in atype]
        atypes = ['C', 'HCQ', 'Rem']
        ahash = {'DMSO':0, 'HCQ':1, 'Remdesivir':2}
        if (tn == 2):
            atype = [atype[i] if tval[i] == ta
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'Rem']
            ahash = {'DMSO':0, 'Remdesivir':1}
            atype = [atype[i] if tval[i] == ta
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getThompson2017(self, tn=1):
        self.prepareData("COV308")
        atype = self.h.getSurvName('c disease state')
        atypes = ['H', 'TB', 'DxC', 'MTP']
        ahash = {'TB Subjects':1,
                'Healthy Controls':0,
                'Lung Dx Controls':2,
                'MTP Controls':3}
        self.initData(atype, atypes, ahash)

    def getLindeboom2020(self, tn=1):
        self.prepareData("COV317")
        atype = self.h.getSurvName('c icu admission')
        ahash = {'no ICU admission':1, 'ICU admission':2, 'NA':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['U', '0', '5', 'CA', 'CA+HCQ']
        ahash = {'0 days HCQ':1, '5 days HCQ':2, 'untreated':0, 'HKCA':3, 'HKCA+HCQ':4}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['C', 'HCQ']
            ahash = {'0 days HCQ':0, '5 days HCQ':1}
        if (tn == 3):
            atypes = ['U', 'CA', 'CA+HCQ']
            ahash = {'untreated':0, 'HKCA':1, 'HKCA+HCQ':2}
        if (tn == 4):
            atypes = ['U', 'CoV2']
            ahash = {'untreated':0, '0 days HCQ':1}
        if (tn == 5):
            atypes = ['U', 'no ICU', 'ICU']
            atype = [tval[i] if aval[i] == 0 or aval[i] == 1
                    else None for i in range(len(atype))]
            ahash = {0:0, 1:1, 2:2}
        self.initData(atype, atypes, ahash)

    def getBossel2019(self, tn=1):
        self.prepareData("COV318")
        atype = self.h.getSurvName('c time (post-infection)')
        atypes = ['0', '4', '8']
        ahash = {'8 hr':2, '0 hr':0, '4 hr':1}
        if (tn == 2):
            atypes = ['C', 'I']
            ahash = {'8 hr':1, '0 hr':0}
        self.initData(atype, atypes, ahash)

    def getRobinson2020(self, tn=1):
        self.prepareData("COV319")
        atype = self.h.getSurvName('c infection')
        atypes = ['C', 'I']
        ahash = {'none':0, 'Salmonella typhimurium':1}
        self.initData(atype, atypes, ahash)

    def getMcNeill2018(self, tn=1):
        self.prepareData("COV309")
        atype = self.h.getSurvName('c day')
        ahash = {'D28':28, 'D0':0}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c cmv status')
        ahash = {'CMVn':0, 'CMVp':1}
        mval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c risk for tb')
        atypes = ['C', 'I']
        ahash = {'Control':0, 'Case':1}
        atype = [atype[i] if dval[i] == 0 and mval[i] == 0
                else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKaforou2013(self, tn=1):
        self.prepareData("COV310")
        atype = self.h.getSurvName('c hiv status')
        ahash = {'HIV negative':0, 'HIV positive':1}
        hval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease state')
        atypes = ['C', 'Tb', 'lTb']
        ahash = {'latent TB infection':2,
                'other disease':0,
                'active tuberculosis':1}
        atype = [atype[i] if hval[i] == 0
                else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getAnderson2014(self, tn=1):
        self.prepareData("COV312")
        atype = self.h.getSurvName('c hiv status')
        ahash = {'HIV negative':0, 'HIV positive':1}
        hval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease status')
        atypes = ['C', 'Tb', 'lTb']
        ahash = {'latent TB infection':2,
                'other disease':0,
                'active tuberculosis':1}
        atype = [atype[i] if hval[i] == 0
                else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBartholomeus2019(self, tn=1):
        self.prepareData("COV313")
        atype = self.h.getSurvName('c infected with/healthy control')
        atypes = ['C', 'I']
        ahash = {'Control':0, 'Streptococcus pneumoniae':1, 
                'Haemophilus influenzae':1}
        self.initData(atype, atypes, ahash)

    def getHerberg2016I(self, tn=1):
        self.prepareData("COV168")
        atype = self.h.getSurvName('c src1')
        atype = [re.sub("W.*from ", "", str(k)) for k in atype]
        atype = [re.sub("pa.*with ", "", str(k)) for k in atype]
        atypes = ['C', 'pB', 'pV', 'B', 'V', 'U']
        ahash = {'healthy control':0,
                'Probable Bacterial infection':1,
                'Probable Viral infection':2,
                'Definite Bacterial infection':3,
                'Definite Viral infection':4,
                'infection of uncertain bacterial or viral aetiology':5}
        if (tn == 2):
            atypes = ['C', 'B']
            ahash = {'healthy control':0, 'Definite Bacterial infection':1}
        self.initData(atype, atypes, ahash)

    def getLieberman2020(self, tn=1):
        self.prepareData("COV275")
        atype = self.h.getSurvName('c src1')
        ahash = {'Nasopharyngeal Swab':0, 'Human Airway Epithelial Cells':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c src1')
        ahash = {'Nasopharyngeal Swab':0, 'Human Airway Epithelial Cells':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c sars-cov-2 infection')
        ahash = {'infected':1, 'uninfected':0}
        atype = self.h.getSurvName('c sars-cov-2 positivity')
        atypes = ['C', 'I']
        ahash = {'pos':1, 'neg':0}
        self.initData(atype, atypes, ahash)

    def getPrice2020Cov2(self, tn=1):
        self.prepareData("COV278")
        atype = self.h.getSurvName('c day post-infection')
        ahash = {'Day 0':0, 'Day 1':1, 'Day 3':3, 'Day 5':5, 'Day 7':7,
                'Day 10':10, 'Day 12':12, 'Day 15':15, 'Day 17':17}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c agent')
        atypes = ['C', 'CoV2']
        ahash = {'Control':0, 'SARS-CoV-2':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 or tval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = [0, 1, 3, 5, 7, 10, 12, 15, 17]
            ahash = {}
        self.initData(atype, atypes, ahash)

    def getFiege2020(self, tn=1):
        self.prepareData("COV320")
        atype = self.h.getSurvName('c treatment')
        ahash = {'untreated':0, 'Remdesivir pretreated':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c infection status')
        atypes = ['C', 'CoV2']
        ahash = {'uninfected':0,
                'SARS-CoV-2 infected, MOI 5, 24 hours post infection':1,
                'SARS-CoV-2 infected, MOI 5, 48 hours post infection':1}
        if (tn == 2):
            atypes = ['C', 'CoV2', 'R']
            atype = ['R' if tval[i] == 1
                    else atype[i] for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getGajMSD(self, tn=1):
        self.prepareData("COV257.2")
        atype = self.h.getSurvName('c Disease_stage')
        ahash = {'':0, 'Acute':1}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Diagnosis')
        ahash = {'':0, 'KD':1, 'MIS-C':2}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c src1')
        atypes = ['C', 'I', 'S']
        ahash = {'1y Convalescent (CAA+)':0, '1y Convalescent (CAA-)':0,
                'Acute (CAA+)':1, 'Acute(CAA-)':1, 'MIS-C':1,
                'Healthy Adult':0, 'Acute COVID19':1, 'Standards':2}
        if (tn == 2):
            atype = dval
            ahash = {1:0, 2:1}
            atypes = ['KD', 'MISC']
            atype = [atype[i] if sval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['CV', 'AV']
            ahash = {'1y Convalescent (CAA+)':0, '1y Convalescent (CAA-)':0,
                    'Acute (CAA+)':1, 'Acute(CAA-)':1}
        if (tn == 4):
            atypes = ['CV', 'AV', 'M']
            ahash = {'1y Convalescent (CAA+)':0, '1y Convalescent (CAA-)':0,
                    'Acute (CAA+)':1, 'Acute(CAA-)':1, 'MIS-C':2}
        if (tn == 5):
            atypes = ['AV', 'M']
            ahash = {'Acute (CAA+)':0, 'Acute(CAA-)':0, 'MIS-C':1}
        if (tn == 6):
            atype = self.h.getSurvName('c Group')
            atypes = ['C', 'CoV']
            ahash = {'Healthy Adult':0, 'Acute COVID19':1}
        if (tn == 7):
            atype = self.h.getSurvName('c Gender')
            ahash = {'F':0, 'M':1}
            sval = [ahash[i] if i in ahash else None for i in atype]
            atype = self.h.getSurvName('c Severity')
            atypes = ['NC', 'Critical']
            ahash = {'severe':1, 'critical':1, 'moderate':0,
                    'moderate to severe':0, 'fatal':1, 'Asymptomatic':0,
                    'Fatal':1, 'Severe':1, 'Critical':1, 'Mod-Severe':0,
                    'Moderate':0}
            atype = [atype[i] if sval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBurns2020KDMISC(self, tn=1):
        self.prepareData("COV257.3")
        atype = self.h.getSurvName('c Disease_stage')
        ahash = {'Acute':1, 'Convalescent':0, '':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Diagnosis')
        atypes = ['K', 'M', 'F', 'U']
        ahash = {'KD':0, '':3, 'FC':2, 'MIS-C':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['CV', 'AV', 'M']
            atype = ['CV' if tval[i] == 0 and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if tval[i] == 1 and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 3):
            atypes = ['CAA-', 'CAA+']
            atype = self.h.getSurvName('c CAA_pos_neg')
            ahash = {'pos':1, 'neg':0}
        if (tn == 4):
            atypes = ['C', 'A1', 'A2', 'A3', 'A4']
            btype = self.h.getSurvName("c CA status")
            atype = self.h.getSurvName('c Disease_stage')
            ahash = {'Acute':1, 'Convalescent':0, '':2}
            aval = [ahash[i] if i in ahash else None for i in atype]
            ahash = {'4':4, '3':3, '2':2, '1':1, '':-1, 'na': -1}
            atype = [ahash[btype[i]] if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {'Convalescent':0, 'Acute':1, 0:0, 1:1, 2:2, 3:3, 4:4}
        self.initData(atype, atypes, ahash)

    def getBurns2020KDMISCII(self, tn=1):
        self.prepareData("COV257.4")
        atype = self.h.getSurvName('c Treatment with statin')
        ahash = {'No':0, 'Yes':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease_status')
        atypes = ['K', 'M']
        ahash = {'KD':0, 'MIS-C':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['K', 'M', 'D']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atype = ['D' if str(self.h.headers[i]) == 'S204026'
                    else atype[i] for i in range(len(atype))]
        if (tn == 3):
            atypes = ['K', 'M']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atype = [None if str(self.h.headers[i]) == 'S204026'
                    else atype[i] for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBurns2020KDMISCIII(self, tn=1, ta=0):
        self.prepareData("COV257.5")
        atype = self.h.getSurvName('c zworstever')
        zval = ['Z', 'c Z']
        for i in atype[2:]:
            if i == '' or i == 'na':
                zval.append(0)
            elif float(i) < 2:
                zval.append(1)
            elif float(i) >= 2.5 and float(i) < 10:
                zval.append(2)
            elif float(i) >= 10:
                zval.append(4)
            else:
                zval.append(0)
        iday = self.h.getSurvName("c illday")
        idval = ['i', 'c i']
        for i in iday[2:]:
            if i == '' or i == 'na':
                idval.append(0)
            elif float(i) <= 10:
                idval.append(1)
            elif float(i) > 10:
                idval.append(2)
            else:
                idval.append(0)
        atype1 = self.h.getSurvName('c sex')
        atype2 = self.h.getSurvName('c gender')
        atype = [atype1[i] if atype1[i] != '' else atype2[i]
                for i in range(len(atype1))]
        ahash = {'2':0, '1':1, '':2} #1=male, 2=female
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c batch')
        ahash = {'statin KD WB':2, 'MIS-C_UCSD':3, 'iLess10':0, 'UCL':1, '':4}
        bval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Treatment with statin')
        ahash = {'No':0, 'Yes':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Disease_stage')
        ahash = {'Acute ':2, 'Subacute':1, 'Acute':2, 'Convalescent':0,
                'Acute (Label changed)':2, '':3}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Diagnosis')
        atypes = ['K', 'M', 'FC']
        ahash = {'KD':0, 'MIS-C':1, '':0, 'FC':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atypes = ['K', 'M', 'D']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atype = ['D' if str(self.h.headers[i]) == 'S204026'
                    else atype[i] for i in range(len(atype))]
        if (tn == 3):
            atypes = ['K', 'M']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
            atype = [None if str(self.h.headers[i]) == 'S204026'
                    else atype[i] for i in range(len(atype))]
        if (tn == 4):
            atypes = ['CV', 'SA', 'AV', 'M', 'FC']
            atype = ['CV' if dval[i] == 0 and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['SA' if dval[i] == 1 and aval[i] == 0 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and aval[i] == 0 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            atype = ['FC' if aval[i] == 2
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 5):
            atypes = ['CAA-', 'CAA+']
            atype = self.h.getSurvName('c CAA_pos_neg')
            ahash = {'pos':1, 'neg':0}
        if (tn == 6):
            atypes = ['CV', 'CAA-', 'CAA+S', 'CAA+G']
            btype = self.h.getSurvName("c CA status")
            atype = self.h.getSurvName('c Disease_stage')
            ahash = {'Acute ':2, 'Subacute':1, 'Acute':2, 'Convalescent':0,
                    'Acute (Label changed)':2, '':3}
            aval = [ahash[i] if i in ahash else None for i in atype]
            ahash = {'4':4, '3':3, '2':2, '1':1, '':-1, 'na': -1}
            atype = [ahash[btype[i]] if aval[i] == 2
                    else atype[i] for i in range(len(atype))]
            ahash = {'Convalescent':0, 'Subacute':1, 'Acute (Label changed)':1,
                    'Acute':1, 'Acute ':1, 0:0, 1:1, 2:2, 4:3}
        if (tn == 7):
            atypes = ['CV', 'AV']
            atype = ['CV' if dval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 8):
            atypes = ['SA', 'AV']
            atype = ['SA' if dval[i] == 1 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and tval[i] is not None and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 9):
            atypes = ['SA', 'AV', 'ST', 'M']
            atype = ['SA' if dval[i] == 1 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and tval[i] is not None and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['ST' if dval[i] == 1 and tval[i] == 1 and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 10):
            atypes = ['CV', 'CAA-', 'CAA+S', 'CAA+G']
            atype = self.h.getSurvName('c Disease_stage')
            ahash = {'Acute ':2, 'Subacute':1, 'Acute':2, 'Convalescent':0,
                    'Acute (Label changed)':2, '':3}
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = [zval[i] if aval[i] == 2 and idval[i] == 1 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            ahash = {'Convalescent':0, 1:1, 2:2, 4:3}
        if (tn == 11):
            atypes = ['SA', 'AV', 'M']
            atype = ['SA' if dval[i] == 1 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and tval[i] is not None
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 12):
            atypes = ['SA', 'AV', 'M']
            atype = ['SA' if dval[i] == 1 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and tval[i] is not None
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            atype = [None if str(self.h.headers[i]) == 'S204026'
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 13):
            atypes = ['K', 'M']
            atype = ['K' if dval[i] == 2 and tval[i] is not None
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 14):
            atypes = ['CAA+S', 'CAA+G']
            atype = self.h.getSurvName('c Disease_stage')
            ahash = {'Acute ':2, 'Subacute':1, 'Acute':2, 'Convalescent':0,
                    'Acute (Label changed)':2, '':3}
            aval = [ahash[i] if i in ahash else None for i in atype]
            atype = [zval[i] if aval[i] == 2 and idval[i] == 1 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            ahash = {2:0, 4:1}
        if (tn == 15):
            atypes = ['SA', 'AV', 'M', 'FC']
            atype = ['SA' if dval[i] == 1 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and aval[i] == 0 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and tval[i] is not None
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            atype = ['FC' if aval[i] == 2
                    else atype[i] for i in range(len(atype))]
            ahash = {}
        if (tn == 16):
            atypes = ['CV', 'SA', 'AV', 'M', 'FC']
            atype = ['CV' if dval[i] == 0 and aval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['SA' if dval[i] == 1 and aval[i] == 0 and tval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['AV' if dval[i] == 2 and aval[i] == 0 and bval[i] == 0
                    else atype[i] for i in range(len(atype))]
            atype = ['M' if aval[i] == 1
                    else atype[i] for i in range(len(atype))]
            atype = ['FC' if aval[i] == 2
                    else atype[i] for i in range(len(atype))]
            ahash = {}
            atype = [atype[i] if gval[i] == ta
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getManne2020(self, tn=1):
        self.prepareData("COV321")
        atype = self.h.getSurvName('c disease_stage')
        atypes = ['C', 'I', 'S']
        ahash = {'ICU':2, 'Non-ICU':1, 'Healthy':0}
        self.initData(atype, atypes, ahash)

    def getLacerdaMariano2020Mm(self, tn=1):
        self.prepareData("MACV267")
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("-R.:.*", "", str(k)) for k in atype]
        atypes = ['N-L', 'N-H', 'I-L', 'I-H']
        ahash = {'naive-L':0, 'naive-H':1, 'inf-L':2, 'inf-H':3}
        self.initData(atype, atypes, ahash)

    def getMatkovich2017(self, tn=1):
        self.prepareData("MACV268")
        atype = self.h.getSurvName('c condition')
        atypes = ['N', 'S', 'ICM', 'NICM']
        ahash = {'septic cardiomyopathy':1,
                'ischemic heart disease':2,
                'nonischemic dilated cardiomyopathy':3,
                'nonfailing heart':0}
        if (tn == 2):
            atypes = ['N', 'S']
            ahash = {'septic cardiomyopathy':1,
                    'nonfailing heart':0}
        self.initData(atype, atypes, ahash)

    def getCoulibaly2019Gr(self, tn=1):
        self.prepareData("MACV269")
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['PS', 'SS', 'SIRS']
        ahash = {'SIRS':2, 'septic shock':1, 'presurgical':0}
        self.initData(atype, atypes, ahash)

    def getCoulibaly2019NK(self, tn=1):
        self.prepareData("MACV270")
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['PS', 'SS', 'SIRS']
        ahash = {'SIRS':2, 'septic shock':1, 'presurgical':0}
        self.initData(atype, atypes, ahash)
        

    def getPG2020LungHamNamir(self, tn=1):
        self.prepareData("COV323")
        atype = self.h.getSurvName('c info')
        atypes = ['U', 'V', 'N', '3', '4']
        ahash = {'Merck Drug':2, 'Merck Drug Vehicle Control':1, 'UN':0}
        if (tn == 2):
            atypes = ['U', 'V', 'N']
            ahash = {'Merck Drug':2, 'Merck Drug Vehicle Control':1, 'UN':0}
            ah = {'3', '4'}
            atype = [None if k in ah else k for k in atype]
        if (tn == 3):
            atypes = ['U', 'V']
            ahash = {'Merck Drug Vehicle Control':1, 'UN':0}
            ah = {'3', '4'}
            atype = [None if k in ah else k for k in atype]
        if (tn == 4):
            atypes = ['V', 'N']
            ahash = {'Merck Drug':1, 'Merck Drug Vehicle Control':0}
            ah = {'3', '4'}
            atype = [None if k in ah else k for k in atype]
        self.initData(atype, atypes, ahash)

    def getThair2020(self, tn=1, tb=0):
        self.prepareData("COV327")
        atype = self.h.getSurvName('c disease')
        atypes = ['H', 'CoV']
        ahash = {'Healthy control':0, 'COVID19':1}
        self.initData(atype, atypes, ahash)

    def getBernardes2020(self, tn=1, tb=0):
        self.prepareData("COV328")
        atype = self.h.getSurvName('c remission')
        atypes = ['H', 'R', 'NR']
        ahash = {'Remission':1, 'Healthy':0, 'No Remission':2}
        self.initData(atype, atypes, ahash)

    def getMaia2020(self, tn=1, tb=0):
        self.prepareData("COV336")
        atype = self.h.getSurvName('c src1')
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'Classical':0, 'Non-classical':1, 'Plasmacytoid':2,
                'Neutrophils':3, 'Basophils':4, 'Myeloid':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c haematological tumor')
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        ahash = {'No':0, 'Monoclonal':1, 'Acute':2, 'Diffuse':3}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c covid status')
        atypes = ['R', 'I']
        ahash = {'Infected':1, 'Recovered':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getParnell2013(self, tn=1, tb=0):
        self.prepareData("MACV272")
        atype = self.h.getSurvName('c disease status')
        atypes = ['H', 'S', 'NS']
        ahash = {'healthy':0, 'sepsis survivor':1, 'sepsis nonsurvivor':2}
        if (tn == 2):
            atypes = ['H', 'S']
            ahash = {'healthy':0, 'sepsis survivor':1, 'sepsis nonsurvivor':1}
        if (tn == 3):
            atypes = ['S', 'NS']
            ahash = {'sepsis survivor':0, 'sepsis nonsurvivor':1}
        self.initData(atype, atypes, ahash)

    def getParedes2020CRC(self, tn=1, tb=0):
        self.prepareData("CRC148")
        atype = self.h.getSurvName('c individual')
        atype = [re.sub("_.*", "", str(k)) for k in atype]
        ahash = {'Caucasian':0, 'AfricanAmerican':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c tissue')
        atype = [re.sub(" .*", "", str(k)) for k in atype]
        atypes = ['N', 'T']
        ahash = {'Tumor':1, 'Adjacent':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'AA']
            atype = tval
            ahash = {0:0, 1:1}
            atype = [atype[i] if aval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKim2020VV(self, tn=1, tb=0):
        self.prepareData("MACV273")
        atype = self.h.getSurvName('c Title')
        time = [str(k).split("_")[2] if len(str(k).split("_")) > 2
                         else None for k in atype]
        ahash = {'0h':0, '3h':3, '6h':6}
        tval = [ahash[i] if i in ahash else None for i in time]
        host = [str(k).split("_")[0] if len(str(k).split("_")) > 2
                         else None for k in atype]
        ahash = {'dTHP-1':0, 'HT-29':1}
        hval = [ahash[i] if i in ahash else None for i in host]
        atype = self.h.getSurvName('c src1')
        atype = [re.sub(" ce.*", "", str(k)) for k in atype]
        atypes = ['M', 'I']
        ahash = {'Vibrio-infected dTHP-1':1, 'Mock-treated dTHP-1':0,
                'Vibrio-infected HT-29':1, 'Mock-treated HT-29':0}
        ival = [ahash[i] if i in ahash else None for i in atype]
        atypes = [0, 3, 6]
        atype = tval
        atype = [atype[i] if hval[i] == 0
                else None for i in range(len(atype))]
        atype = [0 if ival[i] == 0
                else atype[i] for i in range(len(atype))]
        ahash= {}
        if (tn == 2 or tn == 4):
            atypes = [0, 3, 6]
            atype = tval
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]
            atype = [0 if ival[i] == 0
                    else atype[i] for i in range(len(atype))]
        if (tn == 3 or tn == 4):
            atypes = [0, 6]
        self.initData(atype, atypes, ahash)

    def getStathopoulos2009(self, tn=1, tb=0):
        self.prepareData("MACV274")
        atype = self.h.getSurvName('c src1')
        atypes = ['H', '1', '2']
        ahash = {'blood, two primary malignancies':2,
                'blood, healthy':0, 'blood, one primary malignancy':1}
        if (tn == 2):
            atypes = ['H', 'T']
            ahash = {'blood, two primary malignancies':1,
                    'blood, healthy':0, 'blood, one primary malignancy':1}
        self.initData(atype, atypes, ahash)

    def getGarridoMartin2020NSCLC(self, tn=1, tb=0):
        self.prepareData("MACV275")
        atype = self.h.getSurvName('c cells')
        atypes = ['M', 'TAM']
        ahash = {'Macrophages':0, 'Tumour Associated Macrophages':1}
        self.initData(atype, atypes, ahash)

    def getAltman2020Blood(self, tn=1, tb=0):
        self.prepareData("MACV256")
        atype = self.h.getSurvName('c src1')
        disease = [str(k).split("-")[1] if len(str(k).split("-")) > 1
                         else None for k in atype]
        stype = [str(k).split("-")[2] if len(str(k).split("-")) > 2
                         else None for k in atype]
        atypes = ['C', 'COPD']
        ahash = {'whole blood-COPD-Control':0,
                'whole blood-COPD-COPD':1}
        if (tn == 2):
            atypes = ['C', 'Staph']
            ahash = {'whole blood-Staph-Control':0,
                    'whole blood-Staph-Staph':1}
        if (tn == 3):
            atypes = ['C', 'Sepsis']
            ahash = {'whole blood-Sepsis-Control':0,
                    'whole blood-Sepsis-melioidosis':1}
        if (tn == 4):
            atypes = ['C', 'TB']
            ahash = {'whole blood-TB-Control':0,
                    'whole blood-TB-PTB':1}
        if (tn == 5):
            atypes = ['C', 'Melanoma']
            ahash = {'whole blood-Melanoma-Control':0,
                    'whole blood-Melanoma-Melanoma':1}
        if (tn == 6):
            atypes = ['C', 'Bcell def']
            ahash = {'whole blood-B-cell deficiency-Bcell':1,
                    'whole blood-B-cell deficiency-Control':0}
        if (tn == 7):
            atypes = ['C', 'Flu']
            ahash = {'whole blood-Flu-Control':0,
                    'whole blood-Flu-FLU':1}
        if (tn == 8):
            atypes = ['C', 'HIV']
            ahash = {'whole blood-HIV-Control':0,
                    'whole blood-HIV-HIV':1}
        if (tn == 9):
            atypes = ['C', 'JDM']
            ahash = {'whole blood-Juvenile Dermatomyositis-Control':0,
                    'whole blood-Juvenile Dermatomyositis-JDM':1}
        if (tn == 10):
            atypes = ['C', 'KD']
            ahash = {'whole blood-Kawasaki-Control':0,
                    'whole blood-Kawasaki-Kawasaki':1}
        if (tn == 11):
            atypes = ['C', 'Liver Transplant']
            ahash = {'whole blood-Liver Transplant-Control':0,
                    'whole blood-Liver Transplant-Transplant':1}
        if (tn == 12):
            atypes = ['C', 'MS']
            ahash = {'whole blood-MS-Control':0,
                    'whole blood-MS-MS Patient':1}
        if (tn == 13):
            atypes = ['C', 'Pregnancy']
            ahash = {'whole blood-Pregnancy-Control':0,
                    'whole blood-Pregnancy-Pregnancy':1}
        if (tn == 14):
            atypes = ['C', 'RSV']
            ahash = {'whole blood-RSV-Control':0,
                    'whole blood-RSV-RSV':1}
        if (tn == 15):
            atypes = ['C', 'SLE']
            ahash = {'whole blood-SLE-Control':0,
                    'whole blood-SLE-SLE':1}
        if (tn == 16):
            atypes = ['C', 'SoJIA']
            ahash = {'whole blood-SoJIA-Control':0,
                    'whole blood-SoJIA-SoJIA':1}
        self.initData(atype, atypes, ahash)

    def getBadea2008Panc(self, tn=1, tb=0):
        self.prepareData("PANC7")
        atype = self.h.getSurvName("c type")
        atypes = ['N', 'T']
        ahash = {'normal':0, 'tumor':1}
        self.initData(atype, atypes, ahash)

    def getYang2016PDAC(self, tn=1, tb=0):
        self.prepareData("PANC15")
        atype = self.h.getSurvName("c grading")
        ahash = {'G3':3, 'G2':2, 'G4':4, 'Gx':None, 'G1':1}
        gval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c tissue")
        atypes = ['N', 'T']
        ahash = {'Pancreatic tumor':1, 'adjacent pancreatic non-tumor':0}
        if (tn == 2):
            atype = [atype[i] if gval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKirby2016(self, tn=1, tb=0):
        self.prepareData("PANC14")
        atype = self.h.getSurvName("c src1")
        atypes = ['CL', 'T']
        ahash = {'pancreatic adenocarcinoma cancer tissue':1,
                'Pancreatic cancer cell line':0}
        self.initData(atype, atypes, ahash)

    def getTCGAPAAD(self, tn=1, tb=0):
        self.prepareData("PANC13")
        atype = self.h.getSurvName("c Histology")
        atypes = ['N', 'T']
        ahash = {'Primary Tumor':1, 'Solid Tissue Normal':0}
        if tn == 2:
            atypes = ['N', 'T', 'M']
            ahash = {'Primary Tumor':1, 'Solid Tissue Normal':0, 'Metastatic':2}
        self.initData(atype, atypes, ahash)

    def getZhang2012PDAC(self, tn=1, tb=0):
        self.prepareData("PANC11")
        atype = self.h.getSurvName("c sample type")
        atypes = ['N', 'T']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getPei2009PDAC(self, tn=1, tb=0):
        self.prepareData("PANC19")
        atype = self.h.getSurvName("c tissue")
        atypes = ['N', 'T']
        ahash = {'Tumor Tissue in Pancreatic Cancer Sample':1,
                'Normal Tissue in Pancreatic Cancer Sample':0}
        self.initData(atype, atypes, ahash)

    def getBalagurunathan2008PDAC(self, tn=1, tb=0):
        self.prepareData("PANC4")
        atype = self.h.getSurvName("c sample type")
        atypes = ['N', 'T']
        ahash = {'normal tissue':0, 'primary pancreatic tumor':1}
        self.initData(atype, atypes, ahash)

    def getJimeno2008PDAC(self, tn=1, tb=0):
        self.prepareData("PANC6")
        atype = self.h.getSurvName("c Classification")
        atypes = ['S', 'R']
        ahash = {'Sensitive':0, 'Resistant':1}
        self.initData(atype, atypes, ahash)

    def getIshikawa2005PDAC(self, tn=1, tb=0):
        self.prepareData("PANC8.1")
        atype = self.h.getSurvName("c atypicalCellProportion")
        atypes = ['L', 'H']
        ahash = {}
        if (tn == 2):
            atypes = ['L', 'M', 'H']
        self.initData(atype, atypes, ahash)

    def getPeral2021PDAC(self, tn=1, tb=0):
        self.prepareData("PANC17")
        atype = self.h.getSurvName("c src1")
        atypes = ['WT', 'G', 'Het', 'HetG']
        ahash = {'Wildtype control_PDAC':0,
                'CDH11 Heterozygous control_PDAC':2,
                'Wildtype Gemcitabine_PDAC':1,
                'CDH11 Heterozygous Gemcitabine_PDAC':3}
        self.initData(atype, atypes, ahash)

    def getYu2020PDACblood(self, tn=1, tb=0):
        self.prepareData("PANC21")
        atype = self.h.getSurvName("c disease state")
        atypes = ['N', 'T']
        ahash = {'healthy':0, 'PDAC':1}
        self.initData(atype, atypes, ahash)

    def getMoffitt2015PDAC(self, tn=1, tb=0):
        self.prepareData("PANC22")
        atype = self.h.getSurvName('c cell line/tissue')
        ahash = {'Pancreas':0, 'Lung':1, 'Spleen':2, 'Liver':3,
                'Peritoneal':4, 'Colon':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c tissue type")
        atypes = ['N', 'T']
        ahash = {'Primary':1, 'Normal':0}
        atype = [atype[i] if tval[i] == 0
                else None for i in range(len(atype))]
        if (tn == 2):
            atypes = ['N', 'T', 'M']
            ahash = {'Primary':1, 'Metastasis':2, 'Normal':0}
        self.initData(atype, atypes, ahash)

    def getMaurer2019PDAC(self, tn=1, tb=0):
        self.prepareData("PANC23")
        atype = self.h.getSurvName("c compartment")
        atypes = ['S', 'E']
        ahash = {'Epithelium':1, 'Stroma':0}
        if (tn == 2):
            atypes = ['S', 'E', 'B']
            ahash = {'Epithelium':1, 'Stroma':0, 'Bulk':2}
        self.initData(atype, atypes, ahash)

    def getPommier2018PDACmm(self, tn=1, tb=0):
        self.prepareData("PANC24")
        atype = self.h.getSurvName("c cell subpopulation")
        atypes = ['E-', 'E+']
        ahash = {'Ecad+':1, 'Ecad-':0}
        self.initData(atype, atypes, ahash)

    def getJanky2016PDAC(self, tn=1, tb=0):
        self.prepareData("PANC25")
        atype = self.h.getSurvName("c tissue")
        atypes = ['N', 'T']
        ahash = {'pancreatic tumor':1, 'non-tumoral pancreatic tissue':0}
        self.initData(atype, atypes, ahash)

    def getRashid2020PDAC(self, tn=1, tb=0):
        self.prepareData("PANC28")
        atype = self.h.getSurvName("c sample type")
        atypes = ['FNA', 'FFPE', 'FF', 'TB']
        ahash = {'FNA':0, 'FFPE':1, 'FF':2, 'tumor biopsies':3}
        self.initData(atype, atypes, ahash)

    def getRamaswamy2021MISC(self, tn=1, tb=0):
        self.prepareData("COV337")
        atype = self.h.getSurvName("c disease subtype")
        ahash = {'Severe':4, 'Severe; Recovered':3, '--':0, 'Moderate':2,
                 'Moderate; Recovered':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease state")
        atypes = ['H', 'M']
        ahash = {'Multisystem inflammatory syndrome in children (MIS-C)':1, 'healthy':0}
        if (tn == 2):
            atype = tval
            atypes = ['H', 'M', 'S']
            ahash = {0:0, 2:1, 4:2}
        if (tn == 3):
            atype = tval
            atypes = ['H', 'M', 'S']
            ahash = {0:0, 1:0, 3:0, 2:1, 4:2}
        self.initData(atype, atypes, ahash)

    def getBrunetta2021CoV2(self, tn=1, tb=0):
        self.prepareData("COV338")
        atype = self.h.getSurvName("c condition")
        atypes = ['H', 'CoV']
        ahash = {'Healthy Control individual':0, 'COVID-19 hospitalized patient':1}
        self.initData(atype, atypes, ahash)

    def getXu2020CoV2(self, tn=1, tb=0):
        self.prepareData("COV339")
        atype = self.h.getSurvName("c disease condition")
        atypes = ['H', 'CoV', 'Hp', 'IPF', 'Ma', 'Ssa']
        ahash = {'Hypersensitivity pneumonitis':2,
                'Donor':0,
                'Idiopathic pulmonary fibrosis':3,
                'Myositis-associated interstitial lng disease':4,
                'Systemic slcerosis-associated interstitial lung disease':5, '':1}
        if (tn == 2):
            atypes = ['H', 'CoV', 'IPF']
            ahash = {'Donor':0, 'Idiopathic pulmonary fibrosis':2, '':1}
        if (tn == 3):
            atypes = ['H', 'CoV']
            ahash = {'Donor':0, '':1}
        if (tn == 4):
            atypes = ['H', 'IPF']
            ahash = {'Donor':0, 'Idiopathic pulmonary fibrosis':1}
        if (tn == 5):
            atypes = ['IPF', 'CoV']
            ahash = {'Idiopathic pulmonary fibrosis':0, '':1}
        self.initData(atype, atypes, ahash)

    def getGeng2019IPF(self, tn=1, tb=0):
        self.prepareData("COV341")
        atype = self.h.getSurvName("c cell type")
        atypes = ['NI', 'I']
        ahash = {'Invasive cells':1, 'non-Invasive cells':0}
        self.initData(atype, atypes, ahash)

    def getYao2021IPF(self, tn=1, tb=0):
        self.prepareData("COV343","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c group")
        atypes = ['H', 'IPF']
        ahash = {'Donor':0, 'IPF':1}
        self.initData(atype, atypes, ahash) 

    def getBharat2020CoV2(self, tn=1, tb=1):
        dbid = 'COV342.1'
        if tn == 2:
            dbid = 'COV342.2'
        elif tn == 3:
            dbid = 'COV342.3'
        elif tn == 4:
            dbid = 'COV342.4'
        self.prepareData(dbid)
        atype = self.h.getSurvName("c Tissue Type")
        ahash = {'Parenchyma':0, 'Biopsy':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Diagnosis")
        atypes = ['H', 'CoV', 'IPF']
        ahash = {'COVID-19':1, 'Control (B.)':0, 'Control (H.)':0,
                'IPF':2, 'Other PF':2}
        if (tb == 2):
            atypes = ['H', 'CoV']
            atype = [atype[i] if tval[i] == 0
                    else None for i in range(len(atype))]
        if (tb == 3):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBharat2020CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV342.5')
        atype = self.h.getSurvName("c cell type")
        ahash = {'Sorted Stromal population':2,
                'Sorted Myeloid population':1,
                'Sorted CD31 population':3,
                'Sorted Epithelial population':0,
                'Sorted Stromal and Myeloid populations':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c covid-19")
        atypes = ['H', 'CoV']
        ahash = {'Negative':0, 'Positive':1}
        if (tn == 2):
            atypes = ['H', 'CoV']
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBoesch2020ipf(self, tn=1, tb=1):
        self.prepareData('COV344')
        atype = self.h.getSurvName("c condition")
        atypes = ['C', 'IPF']
        ahash = {'idiopathic pulmonary fibrosis (IPF)':1, 'control':0}
        self.initData(atype, atypes, ahash)

    def getLuo2021ildRat(self, tn=1, tb=1):
        self.prepareData('COV345')
        atype = self.h.getSurvName("c bleomycin")
        atypes = ['C', 'Bleomycin']
        ahash = {'with':1, 'without':0}
        self.initData(atype, atypes, ahash)

    def getBauer2015ipfRat(self, tn=1, tb=1):
        self.prepareData('COV346')
        atype = self.h.getSurvName("c time")
        ahash = {'8 WEEKS':8, '2 WEEKS':2, '6 WEEKS':6, '3 DAYS':0,
                '1 WEEKS':1, '4 WEEKS':4, '3 WEEKS':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'Bleomycin']
        ahash = {'VEHICLE':0, 'BLEOMYCIN':1, 'NAIVE (UNTREATED)':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getGuillotin2021ipf(self, tn=1, tb=1):
        self.prepareData('COV347')
        atype = self.h.getSurvName("c source")
        atypes = ['Biopsy', 'transplant']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMeltzer2011ipf(self, tn=1, tb=1):
        self.prepareData('COV348')
        atype = self.h.getSurvName("c gender")
        ahash = {'female':0, 'male':2, 'unknown':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c phenotype")
        atypes = ['H', 'E', 'A']
        ahash = {'advanced idiopathic pulmonary fibrosis (IPF)':2,
                'early idiopathic pulmonary fibrosis (IPF)':1, 'healthy':0}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 2 or aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getYao2020IPF(self, tn=1, tb=1):
        self.prepareData('COV268')
        atype = self.h.getSurvName("c group")
        atypes = ['control', 'IPF']
        ahash = {}
        self.initData(atype, atypes, ahash)
        
    def getYao2020bulk(self, tn=1, tb=1):
        self.prepareData('COV268')
        atype = self.h.getSurvName("c Cell Type")
        #ahash = {'IgG CTC chip': 0}
        ahash = {'bulk':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName("c group")
        atypes = ['control', 'IPF']
        ahash = {}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)          
        

    def getYao2020IPFmm(self, tn=1, tb=1):
        self.prepareData('COV268.2')
        atype = self.h.getSurvName("c treatment")
        atypes = ['control', 'tamoxifen']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getYao2020Lung(self, tn=1, tb=1):
        self.prepareData('COV349')
        atype = self.h.getSurvName("c cd66 status")
        atypes = ['CD66-', 'CD66+']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getYao2020scblkIPFII(self, tn=1, tb=1):
        self.prepareData('COV349.2')
        atype = self.h.getSurvName("c group")
        atypes = ['Donor', 'IPF']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getSullivan2021a549(self, tn=1, tb=1):
        self.prepareData('COV350')
        atype = self.h.getSurvName('c treatment')
        ahash = {'none':0, 'Doxicycline':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c genotype")
        atypes = ['control', 'senescent']
        ahash = {'WT':0, 'TRF2-DN':1}
        atype = ['WT' if tval[i] == 0
                else atype[i] for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getYao2020AT2CoV2(self, tn=1, tb=1):
        self.prepareData('COV367')
        atype = self.h.getSurvName("c group (ch1)")
        atypes = ['C', 'CoV2']
        ahash = {'Infected':1, 'mock':0}
        self.initData(atype, atypes, ahash)

    def getYao2020AT2CoV2II(self, tn=1, tb=1):
        self.prepareData('COV369')
        atype = self.h.getSurvName("c group")
        atypes = ['C', 'CoV2']
        ahash = {'infected with SARS-CoV2':1, 'Mock':0}
        self.initData(atype, atypes, ahash)

    def getLamers2020CoV2(self, tn=1, tb=1):
        self.prepareData('COV326')
        atype = self.h.getSurvName("c cell type")
        ahash = {'Bronchioalveolar':0, 'Small airway':1, 'Lung bud tip':2,
                'Differentiating lung bud tip':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c treatment")
        atypes = ['C', 'CoV2']
        ahash = {'Mock':0, 'SARS-CoV-2':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getKatsura2020at2CoV2(self, tn=1, tb=1):
        self.prepareData('COV370')
        atype = self.h.getSurvName("c infection status")
        atypes = ['C', 'CoV2']
        ahash = {'without infection':0, 'infected SARS-CoV-2':1}
        self.initData(atype, atypes, ahash)

    def getPG2020iAT2(self, tn=1, tb=1):
        self.prepareData('COV371')
        atype = self.h.getSurvName("c type")
        atypes = ['C', '48', '72']
        ahash = {'un':0, '48h':1, '72h':2}
        if (tn == 2):
            atypes = ['C', 'CoV2']
            ahash = {'un':0, '48h':1, '72h':1}
        self.initData(atype, atypes, ahash)

    def getWinkler2021CoV2hACE2mm(self, tn=1, tb=1):
        self.prepareData('COV374')
        atype = self.h.getSurvName("c treatment")
        ahash = {'Naïve':0, 'LALA-PG D+2':2, 'LALA-PG D+1':3,
                '2050 D+2':4, '2050 D+1':5, 'Isotype':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'CoV2']
        ahash = {'Naïve':0, 'SARS2':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == 0 or tval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getZarrinpar2021aging(self, tn=1, tb=1):
        self.prepareData('AGE1')
        atype = self.h.getSurvName("c Sample code")
        ahash = {'D':0, 'BAT':1, 'TI':2, 'LIVER':3, 'WAT':4, 'SKM':5, 'CB':6, '':7}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c Sample code.1")
        atypes = ['NCD', 'HFD']
        ahash = {}
        if (tn == 2):
            atypes = ['ZT6-8', 'ZT9-12', 'ZT18-20', 'ZT21-24']
            ahash = {'ZT6':0, 'ZT7':0, 'ZT8':0, 'ZT9':1, 'ZT10':1, 'ZT11':1, 'ZT12':1,
                    'ZT18':2, 'ZT19':2, 'ZT20':2, 'ZT21':3, 'ZT22':3, 'ZT23':3, 
                    'ZT24':3}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getMelms2021CoV2snblk(self, tn=1, tb=1):
        self.prepareData('COV376')
        atype = self.h.getSurvName("c disease")
        atypes = ['C', 'CoV2']
        ahash = {'COVID-19':1, 'Control':0}
        self.initData(atype, atypes, ahash)

    def getDelorey2021CoV2I(self, tn=1, tb=1):
        self.prepareData('COV377')
        atype = self.h.getSurvName("c src1")
        ahash = {'Parenchyma':0, 'LUL':1, 'blank sample':2, 'Trachea':3,
                'HeartLV':4, 'RUL':5, 'LLL':6}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c segment")
        atypes = ['C', 'CoV2']
        ahash = {'COVID-':0, 'COVID+':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDelorey2021CoV2II(self, tn=1, tb=1):
        self.prepareData('COV377.2')
        atype = self.h.getSurvName("c morphology")
        atypes = ['NA', 'IA', 'BE', 'A']
        ahash = {'Inflamed Alveoli':1, 'Artery':3,
                'Bronchial Epithelium':2, 'Normal Alveoli':0}
        if (tn == 2):
            atypes = ['NA', 'IA']
            ahash = {'Inflamed Alveoli':1, 'Normal Alveoli':0}
        self.initData(atype, atypes, ahash)

    def getDelorey2021CoV2IV(self, tn=1, tb=1):
        self.prepareData('COV378')
        atype = self.h.getSurvName("c tissue")
        atypes = ['Lung', 'Brain', 'Heart']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getDelorey2021CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV378.2')
        atype = self.h.getSurvName("c tissue")
        atypes = ['lung', 'kidney', 'heart', 'liver',
                'LN', 'spleen', 'airway', 'trachea', 'brain']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getMelillo2021CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV379')
        atype = self.h.getSurvName("c disease severity")
        atypes = ['Stable', 'Progressive']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLangelier2021CoV2(self, tn=1, tb=1):
        self.prepareData('COV380')
        atype = self.h.getSurvName("c study group")
        atypes = ['E', 'L', 'T']
        ahash = {'No VAP - longitudinal':1, 'No VAP - late':2, 'No VAP - early':0,
                'VAP - early':0, 'VAP - longitudinal':1, 'VAP - late':2}
        if (tn == 2):
            ahash = {'No VAP - longitudinal':1, 'No VAP - late':2, 'No VAP - early':0}
        if (tn == 3):
            ahash = {'VAP - early':0, 'VAP - longitudinal':1, 'VAP - late':2}
        self.initData(atype, atypes, ahash)

    def getZhao2021CoV2scblk (self, tn=1, tb=1):
        self.prepareData('COV381')
        atype = self.h.getSurvName("c cell type")
        ahash = {'CD3 positive':0, 'CD3 negative':1, 'CD45 negative':2,
                'EpCAM_positive':3, 'CD45 positive':4}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c disease")
        atypes = ['BN', 'CoV2']
        ahash = {'COVID19':1, 'Bacterial pneumonia':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBost2021CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV382')
        atype = self.h.getSurvName("c tissue")
        ahash = {'Blood':0, 'BAL':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c clinical outcome")
        atypes = ['H', 'M', 'S']
        ahash = {'NA':0, 'Dead':2, 'Alive':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['C', 'CoV2']
            ahash = {'NA':0, 'Dead':1, 'Alive':1}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getDesai2020CoV2(self, tn=1, tb=1):
        self.prepareData('COV383')
        atype = self.h.getSurvName("c tissue substructure")
        ahash = {'Alveoli':0, 'Bronchial':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c segment type")
        ahash = {'Geometric':0, 'PanCK_Pos':1, 'PanCK_Neg':2}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c sars-cov-2 rna ish")
        atypes = ['C', 'CoV2']
        ahash = {'Positive':1, 'Negative':0, 'NEARBY_Positive':1, 'Control':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = [atype[i] if sval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getWang2020CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV384')
        atype = self.h.getSurvName("c type of death")
        atypes = ['DBD', 'DCD']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getRen2021CoV2scblk(self, tn=1, tb=1):
        self.prepareData('COV385')
        atype = self.h.getSurvName("c sample type")
        ahash = {'frozen PBMC':1, 'fresh PBMC':0,
                'CD19+ B cell sorted from fresh PBMC (FACS)':2,
                'CD3+ T cell sorted from fresh PBMC (FACS)':3,
                'fresh BALF':4, 'fresh PFMC':5, 'fresh Sputum':6,
                'CD3+ T cell and CD19+ B cell sorted from fresh PBMC (FACS)':7,
                'B cells sorted from frozen PBMC (MACS, STEMCELL 19054)':8,
                'CD19+ B cell sorted from fresh PBMC (MACS)':9}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c sample time")
        ahash = {'progression':2, 'convalescence':1, 'control':0}
        sval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c covid-19 severity")
        atypes = ['H', 'M', 'S']
        ahash = {'severe/critical':2, 'mild/moderate':1, 'control':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName("c sample time")
            atypes = ['C', 'CV', 'AV']
            ahash = {'progression':2, 'convalescence':1, 'control':0}
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getSaichi2021cov2(self, tn=1, tb=1):
        self.prepareData('COV386')
        atype = self.h.getSurvName("c disease severity")
        atypes = ['H', 'M', 'S']
        ahash = {'Severe':2, 'Moderate':1, 'Healthy':0}
        self.initData(atype, atypes, ahash)

    def getFernandez2021iAT2 (self, tn=1, tb=1):
        self.prepareData('COV387')
        atype = self.h.getSurvName("c timepoint")
        ahash = {'Day28':28, 'Day47':47, 'Day50':50, 'Day70':70}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c genotype")
        atypes = ['WT', 'Mut']
        ahash = {'DKC1 A386':0, 'DKC1 A386T':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = [atype[i] if tval[i] == 70
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = tval
            atypes = ['Early', 'Late']
            ahash = {28:0, 70:1}
            atype = [atype[i] if aval[i] == 0
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getRamirez2021drugsMm(self, tn=1, tb=1):
        self.prepareData('COV362')
        atype = self.h.getSurvName('c Title')
        atype = [re.sub("([74]) .*", "\\1", str(k)) for k in atype]
        ahash = {'Control None 7':1, 'Control None 14':1, 'Bleomycin None 7':2,
                'Bleomycin None 14':3, 'Bleomycin Nintedanib 7':4,
                'Bleomycin Nintedanib 14':5}
        dval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c days after treatment")
        ahash = {'(-)':0, '(day 0-6)':1, '(day 7-13)':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName("c days after treatment")
        atypes = ['7', '14']
        ahash = {'7':0, '14':1}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn == 2):
            atype = dval
            atypes = ['UnRx', '7dRx']
            ahash = {2:0, 4:1}
        if (tn == 3):
            atype = dval
            atypes = ['UnRx', '14dRx']
            ahash = {3:0, 5:1}
        self.initData(atype, atypes, ahash)

    def getBorok2020at2mm(self, tn=1, tb=1):
        self.prepareData('COV389')
        atype = self.h.getSurvName('c genotype')
        atypes = ['WT', 'KO']
        ahash = {'Sftpc+/creERT2; Grp78f/f without Tmx':0,
                'Sftpc+/creERT2; Grp78f/f with Tmx':1}
        self.initData(atype, atypes, ahash)

    def getSimonis2021CoV2(self, tn=1, tb=1):
        self.prepareData('COV390')
        atype = self.h.getSurvName('c disease state')
        ahash = {'SARS-CoV-2 convalescent patient':1,
                'SARS-CoV-2 naïve individual':2}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c stimulation')
        atypes = ['NC', 'LPS', 'SP']
        ahash = {'stimulated with LPS':1, 'unstimulated':0,
                'stimulated with S-protein':2}
        aval = [ahash[i] if i in ahash else None for i in atype]
        if (tn >= 2):
            atypes = ['N', 'C']
            atype = tval
            ahash = {2:0, 1:1}
            atype = [atype[i] if aval[i] == (tn - 2)
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getAng2021CoV2(self, tn=1, tb=1):
        self.prepareData('COV391')
        atype = self.h.getSurvName('c pcr status')
        ahash = {'positive':2, 'negative':1, 'healthy_control':0}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['H', 'CoV2']
        ahash = {'COVID-19':1, 'Healthy':0}
        if (tn == 2):
            atype = self.h.getSurvName('c pcr status')
            atypes = ['H', 'PCR-', 'PCR+']
            ahash = {'positive':2, 'negative':1, 'healthy_control':0}
        if (tn == 3):
            atype = self.h.getSurvName('c gene deletion status')
            atypes = ['H', 'ndel', 'del', 'NA']
            ahash = {'no_deletion_hash':1, 'deletion_star':2,
                    'Not Applicable':3, 'healthy_control':0}
            atype = [atype[i] if tval[i] == 0 or tval[i] == 2
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getRagan2021CoV2scblkHam(self, tn=1, tb=1):
        self.prepareData('COV392')
        atype = self.h.getSurvName('c treatment type')
        atypes = ['NV', 'V', 'V+C', 'V+O']
        ahash = {'vaccinated with SolaVAX and CpG1018 adjuvant':2,
                'no vaccination':0,
                'vaccinated with SolaVAX':1,
                'vaccinated with SolaVAX and ODN1668 adjuvant':3}
        self.initData(atype, atypes, ahash)

    def getShannon2020HBV(self, tn=1, tb=1):
        self.prepareData('COV394')
        atype = self.h.getSurvName('c visit')
        atypes = ['V3', 'V4', 'V5', 'V6', 'V7']
        ahash = {}
        self.initData(atype, atypes, ahash)

    def getLee2011Autoimmune(self, tn=1, tb=1):
        self.prepareData('COV400')
        atype = self.h.getSurvName('c disease')
        atypes = ['H', 'HC', 'RA', 'SLE', 'SOJIA', 'PTJIA']
        ahash = {'rheumatoid arthritis':2, 'healthy individual':0,
                'systemic lupus erythematosus':3, 'healthy child':1,
                'systemic-onset juvenile idiopathic arthritis':4,
                'polyarticular type juvenile idiopathic arthritis':5}
        self.initData(atype, atypes, ahash)

    def getTurnier2021SLE(self, tn=1, tb=1):
        self.prepareData('COV401')
        atype = self.h.getSurvName('c disease')
        atypes = ['N', 'SLE', 'JMnL', 'JML']
        ahash = {'cSLE skin lesion':1, 'JM Non-lesional skin':2,
                'Normal skin':0, 'JM Lesional skin':3}
        if (tn == 2):
            atypes = ['N', 'SLE']
            ahash = {'cSLE skin lesion':1, 'Normal skin':0}
        if (tn == 3):
            atypes = ['NL', 'L']
            ahash = {'JM Non-lesional skin':0, 'JM Lesional skin':1}
        self.initData(atype, atypes, ahash)

    def getPanwar2021SLE(self, tn=1, tb=1):
        self.prepareData('COV402')
        atype = self.h.getSurvName('c cell type')
        ahash = {'T cells':0, 'B cells':1, 'PMN':2, 'cDC':3, 'pDC':4, 'cMo':5}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease state')
        atypes = ['H', 'SLE']
        ahash = {'healthy control':0, 'systemic lupus erythematosus (SLE)':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getBrooks2021SLE(self, tn=1, tb=1):
        self.prepareData('COV404')
        atype = self.h.getSurvName('c timepoint')
        ahash = {'1':1, '56':56, '84':84}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['P', 'T']
        ahash = {'Placebo':0, 'Tofacitinib':1}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getZhang2021SLE(self, tn=1, tb=1):
        self.prepareData('COV405')
        atype = self.h.getSurvName('c disease state')
        atypes = ['H', 'SLE']
        ahash = {'Healthy Control':0,
                'Patients of Systematic Lupus Erythematosus':1}
        self.initData(atype, atypes, ahash)

    def getJiang2021SLE(self, tn=1, tb=1):
        self.prepareData('COV406')
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['H', 'SLE']
        ahash = {'SLE patient':1, 'normal control':0}
        self.initData(atype, atypes, ahash)

    def getBarnes2004jra(self, tn=1, tb=1):
        self.prepareData('COV407.1')
        atype = self.h.getSurvName('c Course Type')
        atypes = ['C', 'Pa', 'Po', 'JS']
        ahash = {'Ctrl':0, 'Pauci':1, 'Poly':2, 'JSpA':3}
        if (tn == 2):
            atypes = ['C', 'JRA']
            ahash = {'Ctrl':0, 'Pauci':1, 'Poly':1, 'JSpA':1}
        self.initData(atype, atypes, ahash)

    def getCharney2021MISC(self, tn=1, tb=1):
        self.prepareData('COV408')
        atype = self.h.getSurvName('c age range (years)')
        ahash = {'6-11':0, '18-23':1, '0-5':0, '12-17':0, '24-29':1, '35-40':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['H', 'CoV', 'M']
        ahash = {'covid':1, 'MISC':2, 'healthy':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getdeCevins2021MISC(self, tn=1, tb=1):
        self.prepareData('COV409')
        atype = self.h.getSurvName('c age group')
        ahash = {'pediatric':0, 'adult':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease group')
        atypes = ['H', 'K', 'M', 'MY']
        ahash = {'MISC (CoV2+)':2, 'MISC_MYO (CoV2+)':3, 'KD (CoV2+)':1, 'CTL':0}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atypes = ['M', 'MYO+']
            ahash = {'MISC (CoV2+)':0, 'MISC_MYO (CoV2+)':1}
        self.initData(atype, atypes, ahash)

    def getdeCevins2021MISCscblk(self, tn=1, tb=1):
        self.prepareData('COV409.3')
        atype = self.h.getSurvName('c MajorCellTypes')
        ahash = {'Tcells':0, 'Bcells':1, 'Myeloid cells':2, 'HSC':3}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Group')
        atypes = ['H', 'Inf', 'CoV', 'K', 'M', 'MY']
        ahash = {'CTL':0, 'Acute-Inf(CoV2-)':1, 'Acute-Inf(CoV2+)':2,
                'MIS-C(CoV2+)':4, 'MIS-C_MYO(CoV2+)':5, 'KD(CoV2-)':3}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

    def getdeCevins2021MISCscblkII(self, tn=1, tb=1):
        self.prepareData('COV409.4')
        atype = self.h.getSurvName('c Group')
        atypes = ['H', 'Inf', 'CoV', 'K', 'M', 'MY']
        ahash = {'CTL':0, 'Acute-Inf(CoV2-)':1, 'Acute-Inf(CoV2+)':2,
                'MIS-C(CoV2+)':4, 'MIS-C_MYO(CoV2+)':5, 'KD(CoV2-)':3}
        self.initData(atype, atypes, ahash)

    def getSchulert2020SJIA(self, tn=1, tb=1):
        self.prepareData('COV410')
        atype = self.h.getSurvName('c sample type')
        atypes = ['C', 'NOS', 'A', 'I', 'MAS']
        ahash = {'Control':0, 'NOS':1, 'Active':2, 'Inactive':3, 'NOS/MAS':4}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'Control':0, 'Active':1, 'NOS/MAS':1}
        self.initData(atype, atypes, ahash)

    def getBrown2018SJIA(self, tn=1, tb=1):
        self.prepareData('COV411')
        atype = self.h.getSurvName('c disease state')
        atypes = ['C', 'A', 'I']
        ahash = {'Active SJIA':1, 'Inactive SJIA':2, 'Control patient':0}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'Active SJIA':1, 'Control patient':0}
        self.initData(atype, atypes, ahash)

    def getGorelik2013SJIA(self, tn=1, tb=1):
        self.prepareData('COV412')
        atype = self.h.getSurvName('c age_at_onset')
        atypes = ['<6', '>=6']
        ahash = {'LT6':0, 'GTE6':1}
        self.initData(atype, atypes, ahash)

    def getFall2007SJIA(self, tn=1, tb=1):
        self.prepareData('COV413')
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(".*:", "", str(k)) for k in atype]
        atypes = ['N', 'sJIA']
        ahash = {' normal control':0, ' new onset sJIA':1}
        self.initData(atype, atypes, ahash)

    def getGharib2016sarcoidosis(self, tn=1, tb=1):
        self.prepareData('COV414')
        atype = self.h.getSurvName('c individual')
        atypes = ['N', 'S']
        ahash = {'Sarcoidosis Patient':1, 'Normal Control Subject':0}
        self.initData(atype, atypes, ahash)

    def getAubert2012nomid(self, tn=1, tb=1):
        self.prepareData('COV415')
        atype = self.h.getSurvName('c disease development')
        atypes = ['N', 'L', 'Pre', 'Post']
        ahash = {'post-treatment non-lesional':3, 'lesional':1,
                'pre-treatment non-lesional':2, 'normal':0}
        if (tn == 2):
            atypes = ['C', 'L']
            ahash = { 'lesional':1, 'pre-treatment non-lesional':0, 'normal':0}
        if (tn == 3):
            atypes = ['C', 'L']
            ahash = { 'lesional':1, 'normal':0}
        self.initData(atype, atypes, ahash)

    def getAlmeida2011nomid(self, tn=1, tb=1):
        self.prepareData('COV416')
        atype = self.h.getSurvName('c src1')
        atypes = ['N', 'T']
        ahash = {'bone tumor':1, 'cartilage':0}
        self.initData(atype, atypes, ahash)

    def getCanna2014mas(self, tn=1, tb=1):
        self.prepareData('COV417')
        atype = self.h.getSurvName('c src1')
        atypes = ['H', 'M', 'Pre', 'Post']
        ahash = {'patient with NLRC4-MAS':1,
                'NOMID patients with active disease prior to anakinra treatment':2,
                'NOMID patients with inactive disease after anakinra treatment':3,
                'healthy pediatric controls':0}
        if (tn == 2):
            atypes = ['H', 'D']
            ahash = {'patient with NLRC4-MAS':1,
                    'healthy pediatric controls':0}
        self.initData(atype, atypes, ahash)

    def getFrank2009jra(self, tn=1, tb=1):
        self.prepareData('COV418')
        atype = self.h.getSurvName('c Title')
        atype = [re.sub(" [0-9].*", "", str(k)) for k in atype]
        atypes = ['C', 'JIA', 'JDM']
        ahash = {'Neutrophil JIA':1, 'Neutrophil Control':0,
                'Neutrophil JDM':2, 'PBMC JIA':1,
                'PBMC JDM':2, 'Neutrophil control':0, 'PBMC Control':0}
        if (tn == 2):
            atypes = ['C', 'JIA', 'JDM']
            ahash = {'PBMC JIA':1, 'PBMC JDM':2, 'PBMC Control':0}
        if (tn == 3):
            atypes = ['C', 'JIA', 'JDM']
            ahash = {'Neutrophil JIA':1, 'Neutrophil Control':0,
                    'Neutrophil JDM':2, 'Neutrophil control':0}
        self.initData(atype, atypes, ahash)

    def getWong2016jia(self, tn=1, tb=1):
        self.prepareData('COV419')
        atype = self.h.getSurvName('c disease stage')
        atypes = ['C', 'U', 'T', 'R']
        ahash = {'healthy control':0,
                'JIA patient with clinical remission on medication':3,
                'JIA patient with active, untreated disease':1,
                'JIA patient with active disease on treatment':2}
        if (tn == 2):
            atypes = ['C', 'D']
            ahash = {'healthy control':0,
                    'JIA patient with active, untreated disease':1}
        self.initData(atype, atypes, ahash)

    def getZhang2021CoV2pbmc(self, tn=1, tb=1):
        self.prepareData('COV420')
        atype = self.h.getSurvName('c immune response')
        atypes = ['U', 'A', 'S', 'R', 'P']
        ahash = {'uninfected':0, 'asymptomatic':1, 'symptomatic':2,
                'recovering':3, 're-detectable positive patients':4}
        self.initData(atype, atypes, ahash)

    def getVanBooven2021GWI(self, tn=1, tb=1):
        self.prepareData('COV421')
        atype = self.h.getSurvName('c condition')
        ahash = {'Healthy Control':0, 'GWI':1}
        tval = [ahash[i] if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c time point')
        atypes = ['T0', 'T1', 'T2']
        ahash = {}
        if (tn == 2):
            atype = [atype[i] if tval[i] == tb
                    else None for i in range(len(atype))]
        if (tn == 3):
            atype = self.h.getSurvName('c src1')
            atypes = ['HT0', 'HT1', 'HT2', 'T0', 'T1', 'T2']
            ahash = {'HCGWI_T1_PBMC':1, 'HCGWI_T0_PBMC':0, 'HCGWI_T2_PBMC':2,
                    'GWI_T2_PBMC':5, 'GWI_T1_PBMC':4, 'GWI_T0_PBMC':3}
        if (tn == 4):
            atype = self.h.getSurvName('c src1')
            atypes = ['T2', 'T0']
            ahash = {'GWI_T2_PBMC':0, 'GWI_T0_PBMC':1}
        self.initData(atype, atypes, ahash)
        
    def getGow2009(self, tn=1, tb=0):
        self.prepareData("SS2", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['PBMC, control', 'PBMC, CFS']
        ahash = {'PBMC, control':0, 'PBMC, CFS':1}
        self.initData(atype, atypes, ahash)
        return
    
    
    def getSabath2021(self, tn=1, tb=0):
        self.prepareData("SS1", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c group (ch1)')
        atypes = ['NH', 'H']
        ahash = {'not hospitalized':0, 'hospitalized':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getByrnes2009CFS(self, tn=1, tb=0):
        self.prepareData("SS3", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c diagnonsis (ch1)')
        atypes = ['unaffected', 'CFS']
        ahash = {'unaffected':0, 'CFS':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getByrnes2009ICF(self, tn=1, tb=0):
        self.prepareData("SS3", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c diagnonsis (ch1)')
        atypes = ['unaffected', 'ICF']
        ahash = {'unaffected':0, 'ICF':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getOvanda2021(self, tn=1, tb=0):
        self.prepareData("SS4", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease status (ch1)')
        atypes = ['Severe asthma', 'healthy']
        ahash = {'Severe asthma':0, 'healthy':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getSeverino2014(self, tn=1, tb=0):
        self.prepareData("SS5", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['healthy control','septic patient_survivor', 'septic patient_non-survivor']
        ahash = {'healthy control':0,'septic patient_survivor':1, 'septic patient_non-survivor':2}
        self.initData(atype, atypes, ahash)
        return
    
    def getVan20211(self, tn=1, tb=0):
        self.prepareData("SS7", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c condition (ch1)')
        atypes = ['Healthy Control','GWI']
        ahash = {'Healthy Control':0,'GWI':1}
        self.initData(atype, atypes, ahash)
        return
       
    def getBhatta2001(self, tn=1, tb=0):
        self.prepareData("LU5", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample2')
        atypes = ['Normal Lung','Carcinoids']
        ahash = {'Normal Lung':0,'Carcinoids':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getrobles2015(self, tn=1, tb=0):
        self.prepareData("LU11", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c src1')
        atypes = ['lung, nontumor adjacent', 'lung, adenocarcinoma']
        ahash = {'lung, nontumor adjacent':0,'lung, adenocarcinoma':1}
        self.initData(atype, atypes, ahash)
        return
        
    def getreichmann2021(self, tn=1, tb=0):
        self.prepareData("SS9", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c condition (ch1)')
        atypes = ['Control', 'Mtb']
        ahash = {'Control':0,'Mtb':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getcasanova2020(self, tn=1, tb=0):
        self.prepareData("SS10", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type (ch1)')
        atypes = ['healthy tissue', 'sarcoidosis granuloma']
        ahash = {'healthy tissue':0,'sarcoidosis granuloma':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getmura2019(self, tn=1, tb=0):
        self.prepareData("COV355", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease state')
        atypes = ['normal control', 'idiopathic PAH patient']
        ahash = {'normal control':0,'idiopathic PAH patient':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getwest2019(self, tn=1, tb=0):
        self.prepareData("COV359", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type')
        atypes = ['Control', 'CTEPH']
        ahash = {'Control':0,'CTEPH':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getreyfman2019_IPF(self, tn=1, tb=0):
        self.prepareData("COV340", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease condition')
        atypes = ['Donor', 'Idiopathic pulmonary fibrosis']
        ahash = {'Donor':0,'Idiopathic pulmonary fibrosis':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getreyfman2019_My(self, tn=1, tb=0):
        self.prepareData("COV340", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease condition')
        atypes = ['Donor', 'Myositis-associated interstitial lng disease']
        ahash = {'Donor':0,'Myositis-associated interstitial lng disease':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getreyfman2019_Sys(self, tn=1, tb=0):
        self.prepareData("COV340", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease condition')
        atypes = ['Donor', 'Systemic slcerosis-associated interstitial lung disease']
        ahash = {'Donor':0,'Systemic slcerosis-associated interstitial lung disease':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getreyfman2019_Hy(self, tn=1, tb=0):
        self.prepareData("COV340", "/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease condition')
        atypes = ['Donor', 'Hypersensitivity pneumonitis']
        ahash = {'Donor':0,'Hypersensitivity pneumonitis':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getLi2021(self, tn=1, tb=0):
        self.prepareData("SS11", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['Healthy PBMC & BAL Donor  2 GEX', 'COVID-19 PBMC & BAL donor 2 GEX']
        ahash = {'Healthy PBMC & BAL Donor  2 GEX':0,'COVID-19 PBMC & BAL donor 2 GEX':1}
        self.initData(atype, atypes, ahash)
        return
    
    def getchen2021ACE2mm(self, tn=1, tb=1):
        self.prepareData('SS12', "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c genotype (ch1)")
        atypes = ['wild type', 'hACE2']
        ahash = {'wild type':0, 'hACE2':1}
        self.initData(atype, atypes, ahash)
        
    def getchen2021CD147mm(self, tn=1, tb=1):
        self.prepareData('SS12', "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c genotype (ch1)")
        atypes = ['wild type', 'hCD147']
        ahash = {'wild type':0, 'hCD147':1}
        self.initData(atype, atypes, ahash)
    
    def getWinkler2021CoV2hACE2mm(self, tn=1, tb=1):
        self.prepareData('COV374')
        atype = self.h.getSurvName("c infection")
        atypes = ['C', 'CoV2']
        ahash = {'Naïve':0, 'SARS2':1}
        self.initData(atype, atypes, ahash)
  
    def getPG2020EIDD(self, tn=1, tb=1):
        self.prepareData("COV323")
        atype = self.h.getSurvName('c info')
        atypes = ['3', '4']
        ahash = {'3':0, '4':1}
        self.initData(atype, atypes, ahash)
        
    def getPG2020UN(self, tn=1, tb=1):
        self.prepareData("COV323")
        atype = self.h.getSurvName('c info')
        atypes = ['UN', '3']
        ahash = {'UN':0, '3':1}
        self.initData(atype, atypes, ahash)
        
    def getChen2021CD4(self, tn=1, tb=1):
        self.prepareData("SS13", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0, 'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)
        
    def getChen2021CD8(self, tn=1, tb=1):
        self.prepareData("SS14", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 7 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1, 'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)

    def getChen2021Bcell(self, tn=1, tb=1):
        self.prepareData("SS15", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0, 'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)   
        
    def getChen2021epi(self, tn=1, tb=1):
        self.prepareData("SS16", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        #atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        #ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0, 'Healthy PBMC & BAL Donor  2 GEX':0}
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0}
        self.initData(atype, atypes, ahash) 
        
    def getChen2021mac(self, tn=1, tb=1):
        self.prepareData("SS17", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0, 'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)
        
    def getChen2021nk(self, tn=1, tb=1):
        self.prepareData("SS18", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1, 'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)
        
    def getChen2021tcell(self, tn=1, tb=1):
        self.prepareData("SS19", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        #atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy BAL Donor  1 GEX', 'Healthy PBMC & BAL Donor  2 GEX']
        #ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy BAL Donor  1 GEX':0, 'Healthy PBMC & BAL Donor  2 GEX':0}
        atypes = ['COVID-19 PBMC & BAL donor 2 GEX','COVID-19 PBMC & BAL donor 3 GEX','COVID-19 PBMC & BAL donor 4 GEX','COVID-19 PBMC & BAL donor 5 GEX','COVID-19 PBMC & BAL donor 6 GEX','COVID-19 PBMC & BAL donor 7 GEX','Healthy PBMC & BAL Donor  2 GEX']
        ahash = {'COVID-19 PBMC & BAL donor 2 GEX':1,'COVID-19 PBMC & BAL donor 3 GEX':1,'COVID-19 PBMC & BAL donor 4 GEX':1,'COVID-19 PBMC & BAL donor 5 GEX':1,'COVID-19 PBMC & BAL donor 6 GEX':1,'COVID-19 PBMC & BAL donor 7 GEX':1,'Healthy PBMC & BAL Donor  2 GEX':0}
        self.initData(atype, atypes, ahash)
    def getsorlie(self):
        self.getSurvival("BC1")        
    def getsorlie2(self):
        self.prepareData("BC1", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c \"Clinical ER (0=ER negative, 1=ER+ via immunohistochemical and/or DC-binding assay >9)\"")
        atypes = ['ER neg', 'ER Pos']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)     
    def getveer(self):
        self.getSurvival("BC2")
    def getPawitan(self):
        self.getSurvival("BC3")
    def getmiller(self):
        self.prepareData('BC4')
        atype = self.h.getSurvName('c ER status')
        atypes = ['ER-', 'ER+']
        ahash = {'ER+':1, 'ER-':0}
        self.initData(atype, atypes, ahash)   
        
    def getmiller2(self, tn=1, tb=0):
        self.prepareData("BC4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ER status')
        ahash = {'ER-':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('status')
        atypes = ['0','1']
        ahash = {'0':0, '1':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)   
        
    def getwang(self):
        self.getSurvival("BC5")
    def getwang2(self):
        self.prepareData("BC5", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c ER Status")
        atypes = ['ER-', 'ER+']
        ahash = {'ER-':0, 'ER+':1}
        self.initData(atype, atypes, ahash)   
        
    def getIvshina(self):
        self.getSurvival("BC6")
                
    def getBC(self):
        self.getSurvival("BC7")
        
    def getBC5(self):
        self.getSurvival("BC5")
    
    def getzhou2012(self, tn=1, tb=1):
        self.prepareData("SS19", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c source_name (ch1)")
        atypes = ['Healthy control', 'IPF patient']
        ahash = {'Healthy control':0, 'IPF patient':1}
        self.initData(atype, atypes, ahash)   
    def getipf(self, tn=1, tb=0):
        self.prepareDataDf("MACV75")
        
    def getdobosh2021(self, tn=1):
        self.prepareData("SS24", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("ArrayId")
        atypes = ['untreated', '24 hours__Infection_no_transmigration']
        ahash = {'GSM5652413':0, 'GSM5652414':0, 'GSM5652426':1, 'GSM5652427':1, 'GSM5652415':0, 'GSM5652416':0, 'GSM5652417':0}
        if (tn == 2):
            atypes = ['untreated', '24 hours__Infection_transmigration']
            ahash = {'GSM5652413':0, 'GSM5652414':0, 'GSM5652415':0, 'GSM5652416':0, 'GSM5652417':0, 'GSM5652439':1, 'GSM5652440':1}
        self.initData(atype, atypes, ahash)
        
    def getvono2021(self, tn=1):
        self.prepareData("SS26", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName("c time point (ch1)")
        atypes = ['Negative visit 1', 'POS interval 1']
        ahash = {'Negative visit 1':0, 'POS interval 1':1}
        if (tn == 2):
            atypes = ['Negative visit 2', 'POS interval 2']
            ahash = {'Negative visit 2':0, 'POS interval 2':1}
        if (tn == 3):
            atypes = ['Negative visit 3', 'POS interval 3']
            ahash = {'Negative visit 3':0, 'POS interval 3':1} 
        if (tn == 4):
            atypes = ['Negative visit 5', 'POS interval 5']
            ahash = {'Negative visit 5':0, 'POS interval 5':1}             
        self.initData(atype, atypes, ahash)

   # def getdesmedt(self):
    #    self.getSurvival("TNB1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        
    def getdesmedt(self, tn=1, tb=0):
        self.prepareData("TNB1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)

    def getdesmedt2(self, tn=1, tb=0):
        self.prepareData("BC16.1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)        
        
        
    def getsotiriou(self, tn=1, tb=0):
        self.prepareData("TNB2","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1', 'NA']
        ahash = {'0':0, '1':1, 'NA':2}
        self.initData(atype, atypes, ahash)
        
    def getsotiriou2(self, tn=1, tb=0):
        self.prepareData("TNB2","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c grade (ch1)')
        atypes = ['1', '2', '3']
        ahash = {'1':0, '2':1, '3':2}
        self.initData(atype, atypes, ahash)
        
    def getschmidt(self, tn=1, tb=0):
        self.prepareData("TNB3","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)
        
    def getLuker(self, tn=1, tb=0):
        self.prepareData("BC68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Condition')
        atypes = ['Mono-Culture', 'T47D+HS5 Co-Culture', 'T47D+HS27a Co-Culture', 'MCF7+HS27a Co-Culture', 'MCF7+HS5 Co-Culture']
        ahash = {'Mono-Culture':0, 'T47D+HS5 Co-Culture':1, 'T47D+HS27a Co-Culture':3, 'MCF7+HS27a Co-Culture':4, 'MCF7+HS5 Co-Culture':5}
        self.initData(atype, atypes, ahash)
        
    def getLuker2(self, tn=1, tb=0):
        self.prepareData("BC68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Condition')
        atypes = ['Mono-Culture', 'MCF7+HS27a Co-Culture']
        ahash = {'Mono-Culture':0, 'MCF7+HS27a Co-Culture':1}
        self.initData(atype, atypes, ahash)
        
    def getLuker3(self, tn=1, tb=0):
        self.prepareData("BC68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        #atype = self.h.getSurvName('c Cell type of interest')
        atype = self.h.getSurvName('ArrayID')
        #ahash = {'MCF7':0}
        ahash = {"285-JB-20":0, "285-JB-19":0, "139620":0, "285-JB-21":0, "285-JB-13":0, "285-JB-15":0, "285-JB-14":0, "139618":0, "139615":0, "285-JB-2":0, "285-JB-3":0, "285-JB-1":0}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Condition')
        atypes = ['Mono-Culture','MCF7+HS5 Co-Culture','MCF7+HS27a Co-Culture']
        ahash = {'Mono-Culture':0,'MCF7+HS5 Co-Culture':1,'MCF7+HS27a Co-Culture':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)

    def getLukerMCF7(self, tn=1, tb=0):
        self.prepareData("BC68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        #atype = self.h.getSurvName('c Cell type of interest')
        atype = self.h.getSurvName('c Cell type of interest')
        ahash = {'MCF7':0}
        #ahash = {"285-JB-20":0, "285-JB-19":0, "139620":0, "285-JB-21":0, "285-JB-13":0, "285-JB-15":0, "285-JB-14":0, "139618":0, "139615":0, "285-JB-2":0, "285-JB-3":0, "285-JB-1":0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Condition')
        atypes = ['Mono-Culture','MCF7+HS5 Co-Culture','MCF7+HS27a Co-Culture']
        ahash = {'Mono-Culture':0,'T47D+HS5 Co-Culture':1,'T47D+HS27a Co-Culture':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getLukerT47D(self, tn=1, tb=0):
        self.prepareData("BC68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        #atype = self.h.getSurvName('c Cell type of interest')
        atype = self.h.getSurvName('c Cell type of interest')
        ahash = {'T47D':0}
        #ahash = {"285-JB-20":0, "285-JB-19":0, "139620":0, "285-JB-21":0, "285-JB-13":0, "285-JB-15":0, "285-JB-14":0, "139618":0, "139615":0, "285-JB-2":0, "285-JB-3":0, "285-JB-1":0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Condition')
        atypes = ['Mono-Culture','T47D+HS5 Co-Culture','T47D+HS27a Co-Culture']
        ahash = {'Mono-Culture':0,'T47D+HS5 Co-Culture':1,'T47D+HS27a Co-Culture':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)

    def getryan2(self, tn=1):
        self.prepareData("SS27", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c timepoint (ch1)')
        ahash = {'Control':0,'12wpi':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease severity (ch1)')
        atypes = ['Healthy', 'Mild']
        ahash = {'Healthy':0, 'Mild':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]    
        if (tn == 2):
            atypes = ['Healthy', 'Moderate']
            ahash = {'Healthy':0, 'Moderate':1}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]             
        if (tn == 3):
            atypes = ['Healthy', 'Severe']
            ahash = {'Healthy':0, 'Severe':1}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]        
            
        if (tn == 4):
            atypes = ['Healthy', 'Critical']
            ahash = {'Healthy':0, 'Critical':1}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]
        if (tn == 5):
            atypes = ['Healthy', 'Mild', 'Critical']
            ahash = {'Healthy':0, 'Mild':1, 'Critical':2}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]
        self.initData(atype, atypes, ahash)

        
        
    def getryan3(self, tn=1):
        self.prepareData("SS27", "/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease severity (ch1)')
        #ahash = {'Healthy':0, 'Critical':1, 'Severe':2}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c timepoint (ch1)')
        atypes = ['Control', '12wpi']
        ahash = {'Control':0,'12wpi':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]    
        if (tn == 2):
            atypes = ['Control', '16wpi']
            ahash = {'Control':0, '16wpi':1}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]             
        if (tn == 3):
            atypes = ['Control','24wpi']
            ahash = {'Control':0, '24wpi':1}
            atype = [atype[i] if hval[i] == 1
                    else None for i in range(len(atype))]            
        self.initData(atype, atypes, ahash)
        
    def getutrero(self, tn=1, tb=0):
        self.prepareData("SS28","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c infection status (ch1)')
        atypes = ['Healthy Control', 'Acute COVID19', 'Post COVID19']
        ahash = {'Healthy Control': 0, 'Acute COVID19':1 , 'Post COVID19': 2}
        self.initData(atype, atypes, ahash)
        
    def getBos2(self, tn=1):
        self.prepareData("BC20")
        atype = self.h.getSurvName('c Bone relapses')
        atypes = ['0', '1']
        ahash = {'0': 0, '1':1}
        self.initData(atype, atypes, ahash)
        
        
    def getbusch(self, tn=1):
        self.prepareData("BC67")
        atype = self.h.getSurvName('c condition')
        atypes = ['resistant', 'naive', 'residual']
        ahash = {'resistant': 0, 'naive':1, 'residual':2}
        self.initData(atype, atypes, ahash)
        
    def getTiezzi(self, tn=1):
        self.prepareData("TNB4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['human breast tumor', 'human breast tumor_ALDH1+']
        ahash = {'human breast tumor_ALDH1+': 1, 'human breast tumor':0}
        self.initData(atype, atypes, ahash)
        
    def getwang2006(self, tn=1):
        self.prepareData("TNB5","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        atypes = ['MCF-7', 'MCF-7/ADR']
        ahash = {'MCF-7': 0, 'MCF-7/ADR':1}
        self.initData(atype, atypes, ahash)
        
        
    def getSavage2020(self, tn=1):
        self.prepareData("TNB6","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type (ch1)')
        atypes = ['Primary tumor', 'PDX', 'Ductal PDX', 'Chondroid PDX']
        ahash = {'Primary tumor': 0, 'PDX':1, 'Ductal PDX':2, 'Chondroid PDX':3}
        if (tn == 2):
                atype = self.h.getSurvName('c passage (ch1)')
                atypes = ['0', '1', '2', '3'] 
                ahash = {'0':0, '1':1, '2':2, '3':3}
        if (tn == 3):
                atype = self.h.getSurvName('c passage (ch1)')
                atypes = ['0', '3'] 
                ahash = {'0':0, '3':1}
        self.initData(atype, atypes, ahash)
        
    def getBC23(self, tn=1):
        self.prepareData("BC23")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0': 0, '1':1}
        self.initData(atype, atypes, ahash)

    def getmin2019(self, tn=1, tb=0):
        self.prepareData("TNB7","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c parental or metastasis? (ch1)')
        atypes = ['parental CTC', 'metastatic sample']
        ahash = {'parental CTC':0, 'metastatic sample':1}
        self.initData(atype, atypes, ahash)
        
    def getmin2(self, tn=1, tb=0):
        self.prepareData("TNB7","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c grown_in_culture (ch1)')
        ahash = {'no':0, 'yes':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['culture', 'Kidney','lung','ovary','brain','bone']
        ahash = {'culture':0, 'Kidney':1,'lung':2,'ovary':3,'brain':4,'bone':5}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)        
        
        
    def get2min2019(self, tn=1, tb=0):
        self.prepareData("TNB7","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c parental or metastasis? (ch1)')
        ahash = {'metastatic sample':0}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c metastatic site (ch1)')
        atypes = ['n/a','brain','bone']
        ahash = {'n/a':0, 'brain':1,'bone':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getebright2020(self, tn=1):
        self.prepareData("TNB8","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('celltype')
        atypes = ['CL', 'SC']
        ahash = {'CL': 0, 'SC':1}
        self.initData(atype, atypes, ahash)
        
    def getebrightall(self, tn=1):
        self.prepareData("TNB8.2","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('celltype')
        atypes = ['CL', 'SC']
        ahash = {'CL': 0, 'SC':1}
        self.initData(atype, atypes, ahash)        
        
    def getdelrio2(self, tn=1, tb=0):
        self.prepareData("CRC101","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c organism part')
        ahash = {'Primary Tumor':0}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c Status')
        atypes = ['R','NR']
        ahash = {'R':0, 'NR':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getaceto2013(self, tn=1):
        self.prepareData("TNB9","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c single cells or cluster (ch1)')
        atypes = ['CL', 'SC']
        ahash = {'CL': 0, 'SC':1}
        self.initData(atype, atypes, ahash)
        
    def getleon2021(self, tn=1, tb=0):
        self.prepareData("SS31","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        ahash = {'':0}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['control','Spike Protein S1']
        ahash = {'control':0, 'Spike Protein S1':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
        
    def getleon2(self, tn=1):
        self.prepareData("SS31","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['control','Spike Protein S1']
        ahash = {'control':0, 'Spike Protein S1':1}
        self.initData(atype, atypes, ahash)
        
    def getleon3(self, tn=1, tb=0):
        self.prepareData("SS31","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        ahash = {'Spike Protein S1':0}
        #ahash = {'Healthy':0, 'Mild':1}
        #ahash = {'Healthy':0, 'Moderate':1}
        #ahash = {'Healthy':0, 'Severe':1}
        #ahash = {'Healthy':0, 'Critical':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c cell line (ch1)')
        atypes = ['Alveolar_Spike_Protein','Bronchial_Spike_Protein']
        ahash = {'NCI-H441':0, '':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getonishi2021(self, tn=1):
        self.prepareData("TNB10","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c status (ch1)')
        atypes = ['alive', 'dead']
        ahash = {'alive': 0, 'dead':1}
        self.initData(atype, atypes, ahash)
        
        
    def getdharmasiri(self, tn=1):
        self.prepareData("SS32","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c patient diagnosis (ch1)')
        #atypes = ['Normal Control','IBD']
        atypes = ['Normal Control','Ulcerative Colitis','Crohn\'s Disease']
        #atypes = ['Normal Control','Ulcerative Colitis']
        ahash = {'Normal Control':0,'Ulcerative Colitis':1,'Crohn\'s Disease':2}
        self.initData(atype, atypes, ahash)
        
        
    def getPGhs0(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        atypes = ['MDA-MB-231_GIV_KO_0_FBS','MDA-MB-231_parental_0_FBS']
        ahash = {'MDA-MB-231_GIV_KO_0_FBS':0,'MDA-MB-231_parental_0_FBS': 1}
        #atypes = ['MDA-MB-231_parental_10_FBS', 'MDA-MB-231_GIV_KO_10_FBS']
        #ahash = {'MDA-MB-231_parental_10_FBS': 0, 'MDA-MB-231_GIV_KO_10_FBS':1}
        self.initData(atype, atypes, ahash)
        
    def getPGhs10(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        #atypes = ['MDA-MB-231_parental_0_FBS', 'MDA-MB-231_GIV_KO_0_FBS']
        #ahash = {'MDA-MB-231_parental_0_FBS': 0, 'MDA-MB-231_GIV_KO_0_FBS':1}
        atypes = ['MDA-MB-231_GIV_KO_10_FBS', 'MDA-MB-231_parental_10_FBS']
        ahash = {'MDA-MB-231_GIV_KO_10_FBS':0, 'MDA-MB-231_parental_10_FBS':1}
        self.initData(atype, atypes, ahash)
        

        
    def getPGhsWT(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        atypes = ['MDA-MB-231_parental_10_FBS', 'MDA-MB-231_parental_0_FBS']
        ahash = {'MDA-MB-231_parental_10_FBS':0,'MDA-MB-231_parental_0_FBS': 1}
        #atypes = ['MDA-MB-231_parental_10_FBS', 'MDA-MB-231_GIV_KO_10_FBS']
        #ahash = {'MDA-MB-231_parental_10_FBS': 0, 'MDA-MB-231_GIV_KO_10_FBS':1}
        self.initData(atype, atypes, ahash)
        
    def getPGhsKO(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        #atypes = ['MDA-MB-231_parental_0_FBS', 'MDA-MB-231_GIV_KO_0_FBS']
        #ahash = {'MDA-MB-231_parental_0_FBS': 0, 'MDA-MB-231_GIV_KO_0_FBS':1}
        atypes = ['MDA-MB-231_GIV_KO_10_FBS', 'MDA-MB-231_GIV_KO_0_FBS']
        ahash = {'MDA-MB-231_GIV_KO_10_FBS':0, 'MDA-MB-231_GIV_KO_0_FBS':1}
        self.initData(atype, atypes, ahash)       
        
    def getPGhsall(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        #atypes = ['MDA-MB-231_parental_0_FBS', 'MDA-MB-231_GIV_KO_0_FBS']
        #ahash = {'MDA-MB-231_parental_0_FBS': 0, 'MDA-MB-231_GIV_KO_0_FBS':1}
        atypes = ['MDA-MB-231_parental_10_FBS', 'MDA-MB-231_parental_0_FBS', 'MDA-MB-231_GIV_KO_10_FBS', 'MDA-MB-231_GIV_KO_0_FBS']
        ahash = {'MDA-MB-231_parental_10_FBS':0, 'MDA-MB-231_parental_0_FBS':1, 'MDA-MB-231_GIV_KO_10_FBS':2, 'MDA-MB-231_GIV_KO_0_FBS':3}
        self.initData(atype, atypes, ahash)           
        
    def getPGhshm(self, tn=1):
        self.prepareData("SS29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')

        atypes = ['MDA-MB-231_parental_0_FBS_R2_S12' ,'MDA-MB-231_parental_0_FBS_R1_S9' ,'MDA-MB-231_parental_0_FBS_R3_S54' ,'MDA-MB-231_GIV_KO_0_FBS_R1_S4' ,'MDA-MB-231_GIV_KO_0_FBS_R3_S18' ,'MDA-MB-231_GIV_KO_0_FBS_R2_S37']
        ahash = {'MDA-MB-231_parental_0_FBS_R2_S12':0,'MDA-MB-231_parental_0_FBS_R1_S9':1,'MDA-MB-231_parental_0_FBS_R3_S54':2,'MDA-MB-231_GIV_KO_0_FBS_R1_S4':3,'MDA-MB-231_GIV_KO_0_FBS_R3_S18':4,'MDA-MB-231_GIV_KO_0_FBS_R2_S37':5}
        self.initData(atype, atypes, ahash) 
        
#    def getTNB8(self):
#        self.getSurvival("TNB8")
        
#    def getSurvival(self, dbid = "TNB8"):
#        self.prepareData(dbid)
#        atype = self.h.getSurvName("status")
#        atypes = ['0', '1']
#        ahash = {"0": 0, "1":1}
#        self.initData(atype, atypes, ahash)
        
        
        
    def getshan2019(self, tn=1):
        self.prepareData("TNB12","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c tissue (ch1)')
        ahash = {'Breast':0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c dimension')
        atypes = ['2D', '3D']
        ahash = {'2D': 0, '3D':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getCreighton2009(self, tn=1):
        self.prepareData("TNB13","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c characteristics (ch1)')
        atypes = ['CD44+/CD24-', 'cells other than CD44+/CD24-']
        ahash = {'CD44+/CD24-': 0, 'cells other than CD44+/CD24-':1}
        self.initData(atype, atypes, ahash)
        
    def getboral2020(self, tn=1):
        self.prepareData("TNB14","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('CTCstatus')
        atypes = ['BMRC', 'CTC']
        ahash = {'BMRC': 0, 'CTC':1}
        self.initData(atype, atypes, ahash)
        
        
    def getboral2(self, tn=1):
        self.prepareData("TNB14","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('CTCstatus')
        ahash = {'BMRC':1}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c primary tumor subtype (ch1)')
        atypes = ['Triple-positive','TNBC']
        ahash = {'ER+/PR+': 0, 'HER2+':0, 'TNBC':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getboral3(self, tn=1):
        self.prepareData("TNB14","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('CTCstatus')
        ahash = {'CTC':1}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c primary tumor subtype (ch1)')
        atypes = ['Triple-positive','TNBC']
        ahash = {'ER+/PR+': 0, 'HER2+':0, 'TNBC':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getahn2020(self, tn=1):
        self.prepareData("BC69","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ajcc stage (ch1)')
        atypes = ['1', '2', '3', '4']
        ahash = {'1':0, '2':1, '3':2, '4':3}
        self.initData(atype, atypes, ahash)
        
    def getahn2(self, tn=1):
        self.prepareData("BC69","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c vital status (ch1)')
        ahash = {'death': 0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c ajcc stage (ch1)')
        atypes = ['1', '2', '3', '4']
        ahash = {'1':0, '2':1, '3':2, '4':3}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getahn3(self, tn=1):
        self.prepareData("BC69","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ajcc stage (ch1)')
        ahash = {'3': 0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c vital status (ch1)')
        atypes = ['alive', 'death']
        ahash = {'alive':0, 'death':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getBC13(self):
        self.getSurvival("BC13")

    def getPurrington2020(self, tn=1):
        self.prepareData("TNB17","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        ahash = {'Systemic therapy after surgery': 0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c vital status (ch1)')
        atypes = ['Alive', 'Dead']
        ahash = {'Alive':0, 'Dead':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
    def getPurrington2(self, tn=1):
        self.prepareData("TNB17","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c vital status (ch1)')
        atypes = ['Alive', 'Dead']
        ahash = {'Alive':0, 'Dead':1}
        self.initData(atype, atypes, ahash) 
        
    def getchen2020(self, tn=1):
        self.prepareData("TNB18","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c estrogen receptor status (ch1)')
        atypes = ['P', 'N']
        ahash = {'P':0, 'N':1}
        self.initData(atype, atypes, ahash) 
        
    def getprat2021(self, tn=1):
        self.prepareData("TNB16","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c de novo metastasis vs relapsed (ch1)')
        atypes = ['relapsed', 'de novo']
        ahash = {'relapsed':0, 'de novo':1}
        self.initData(atype, atypes, ahash) 
        
    def getNKI(self, tn=1):
        self.prepareData("BC7","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Chemo')
        atypes = ['Yes', 'No']
        ahash = {'Yes':0, 'No':1}
        self.initData(atype, atypes, ahash)
        

    def getHatzis(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c chemosensitivity_prediction')
        atypes = ['Rx Insensitive', 'Rx Sensitive']
        ahash = {'Rx Insensitive':0, 'Rx Sensitive':1}
        self.initData(atype, atypes, ahash)
        
    def getHatzis2(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c pam50_class')
        atypes = ['Normal','LumA','LumB','Her2']
        ahash = {'Normal':0,'LumA':1,'LumB':2,'Her2':3}
        self.initData(atype, atypes, ahash)
        
    def getHatzis_test(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf") 
        atype = self.h.getSurvName('c pam50_class')
        ahash = {'Basal':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c chemosensitivity_prediction')
        atypes = ['Rx Insensitive', 'Rx Sensitive']
        ahash = {'Rx Insensitive':0, 'Rx Sensitive':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)        

        
    def getHatzi3(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c dlda30_prediction')
        atypes = ['RD','pCR']
        ahash = {'RD':0,'pCR':1}
        self.initData(atype, atypes, ahash)
        
    def getHatzi4(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c clinical_nodal_status')
        atypes = ['N0','N1','N2','N3']
        ahash = {'N0':0,'N1':1,'N2':2,'N3':3}
        self.initData(atype, atypes, ahash)        
        
    def getcelltype(self, tn=1):
        self.prepareData("BC11","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c Title')
        atypes = ['Parental','Verapamel','Docetaxel','Paclitaxel']
        ahash = {'MDA-MB-231 Parental_AM':0, 'MDA-MB-231 Parental_AM_B':0,'Parental MDA-MB-231':0, 'MDA-MB-231 Parental':0,'MDA-MB-231_Verapamel_Rep2':1, 'MDA-MB-231_Docetaxel_Rep1':2, 'MDA-MB-231_Paclitaxel_Rep2':3, 'MDA-MB-231_Docetaxel_Rep2':2, 'MDA-MB-231_Verapamel_Rep1':1, 'MDA-MB-231_Docetaxel_Rep3':2, 'MDA-MB-231_Verapamel_Rep3':1, 'MDA-MB-231_Paclitaxel_Rep3':3, 'MDA-MB-231_Paclitaxel_Rep1':3}
        self.initData(atype, atypes, ahash)
        
        
    def getcelltype2(self, tn=1):
        self.prepareData("BC11","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c Title')
        atypes = ['Parental', 'Weakly Lung metastatic', 'Weakly Bone metastatic', 'Strongly lung metastatic', 'Strongly Bone metastatic']
        ahash = {'Parental MDA-MB-231':0 ,'MDA-MB-231 Parental_AM':0 ,'MDA-MB-231 Parental_AM_B':0 ,'Weakly Lung metastatic line SCP26':1 ,'Weakly Lung metastatic line SCP6':1 ,'Weakly Lung metastatic line SCP21':1 ,'Weakly Bone metastatic line 2293':2 ,'Weakly Bone metastatic line 2295':2 ,'Weakly Bone metastatic line 2296':2 ,'Strongly lung metastatic line 4175':3 ,'Strongly lung metastatic line 4173':3 ,'Strongly lung metastatic line 4142':3 ,'Strongly lung metastatic line 4180':3 ,'Strongly Bone metastatic line 1833':4 ,'Strongly Bone metastatic line 2274':4 ,'Strongly Bone metastatic line 2268':4 ,'Strongly Bone metastatic line 2269':4}
        self.initData(atype, atypes, ahash)
        
    def getkeene(self, tn=1):
        self.prepareData("TNB19","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c group_ch1')
        atypes = ['CTL','LRR','DM']
        ahash = {'CTL':0,'LRR':1,'DM':2}
        self.initData(atype, atypes, ahash)
        
    def getkeene2(self, tn=1):
        self.prepareData("TNB19","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c was hormonal therapy given?_ch1')
        ahash = {'Yes': 0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c group_ch1')
        atypes = ['CTL','LRR','DM']
        ahash = {'CTL':0,'LRR':1,'DM':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)       
        
        
    def getlang2018(self, tn=1):
        self.prepareData("TNB21","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c sample type_ch1')
#        atypes = ['peripheral blood','circulating tumor cells sorted from peripheral blood','formalin-fixed, paraffin embedded primary tumor']
#        ahash = {'peripheral blood':0,'circulating tumor cells sorted from peripheral blood':1,'formalin-fixed, paraffin embedded primary tumor':2}

#        atypes = ['formalin-fixed, paraffin embedded primary tumor','circulating tumor cells sorted from peripheral blood']
#        ahash = {'formalin-fixed, paraffin embedded primary tumor':0,'circulating tumor cells sorted from peripheral blood':0}
        
        atypes = ['formalin-fixed, paraffin embedded primary tumor','circulating tumor cells sorted from peripheral blood']
        ahash = {'formalin-fixed, paraffin embedded primary tumor':0,'circulating tumor cells sorted from peripheral blood':0}        
        self.initData(atype, atypes, ahash)
        
    def getlang2(self, tn=1):
        self.prepareData("TNB21","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['peripheral blood','sorted CTC']
        ahash = {'BLOOD_36581':0, 'BLOOD_36683':0, 'BLOOD_36828':0, 'BLOOD_47934':0, 'BLOOD_58029':0, 'BLOOD_68172':0, 'CTC_36581':1, 'CTC_36683':1, 'CTC_36828':1, 'CTC_47934':1, 'CTC_58029':1, 'CTC_68172':1}        
        self.initData(atype, atypes, ahash)
        
    def getCareySurv(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('status')
        atypes = ['0','1']
        ahash = {'0':0,'1':1}
        self.initData(atype, atypes, ahash)
        
        
    def getCarey2010(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c subtype (ch2)')
        atypes = ['Normal','Basal','Claudin-low']
        ahash = {'Normal':0,'Basal':1,'Claudin-low':2}
        self.initData(atype, atypes, ahash)
        
    def getCareyBasal(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c subtype (ch2)')
        atypes = ['Normal','Basal']
        ahash = {'Normal':0,'Basal':1}
        self.initData(atype, atypes, ahash)        

    def getCareyClaudinlow(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c subtype (ch2)')
        atypes = ['Normal','Claudin-low']
        ahash = {'Normal':0,'Claudin-low':1}
        self.initData(atype, atypes, ahash)        
        
        
    def getCarey2(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c subtype (ch2)')
        ahash = {'Claudin-low': 0}
        #ahash = {'Lung':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c source_name (ch2)')
        atypes = ['cetuximab+carboplatin','Untreated']
        ahash = {'LCCC-2002-BxComb':0,'LCCC-4003-BxComb':0,'LCCC-9001-BxComb':0,'LCCC-9006-BxComb':0,'LCCC-2002-Bx0':1,'LCCC-4003-Bx0':1,'LCCC-9001-Bx0':1,'LCCC-9006-Bx0':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getCarey3(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c subtype (ch2)')
        #ahash = {'Basal': 0}
        ahash = {'Claudin-low':1}
       # ahash = {'Claudin-low':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment')
        atypes = ['BxComb','BxSingle']
        ahash = {'BxComb':0,'BxSingle':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getCarey4(self, tn=1):
        self.prepareData("TNB22","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment')
        #ahash = {'Basal': 0}
        ahash = {'BxSingle':1}
       # ahash = {'Claudin-low':1}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c survival status (ch2)')
        atypes = ['Alive','Dead']
        ahash = {'Alive':0,'Dead':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    
    
    
    
    
    def getXiao2020(self, tn=1):
        self.prepareData("TNB27","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c class')
        atypes = ['ER/PR+','HER2+','TNBC']
        ahash = {'ER/PR+':0,'HER2+':1,'TNBC':2}
        self.initData(atype, atypes, ahash)
        
    def getXiao2(self, tn=1):
        self.prepareData("TNB27","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c class')
        atypes = ['HER2+','TNBC']
        ahash = {'HER2+':0,'TNBC':1}
        self.initData(atype, atypes, ahash)        
        
    def getInce2007(self, tn=1):
        self.prepareData("TNB23","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['BPE sample', 'BPLER sample', 'HME sample', 'HMLER sample']
        ahash = {'BPE sample 1':0, 'BPE sample 2':0, 'BPE sample 3':0, 'BPE sample 4':0, 'BPE sample 5':0, 'BPLER sample 1':1, 'BPLER sample 2':1, 'BPLER sample 3':1, 'BPLER sample 4':1, 'BPLER sample 5':1, 'BPLER sample 6':1, 'HME sample 1':2, 'HME sample 2':2, 'HME sample 3':2, 'HME sample 4':2, 'HMLER sample 1':3, 'HMLER sample 2':3, 'HMLER sample 3':3, 'HMLER sample 4':3, 'HMLER sample 5':3, 'HMLER sample 6':3}
        self.initData(atype, atypes, ahash)
        
    def getInce2(self, tn=1):
        self.prepareData("TNB23","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['BPLER sample', 'HMLER sample']
        ahash = {'BPLER sample 1':0, 'BPLER sample 2':0, 'BPLER sample 3':0, 'BPLER sample 4':0, 'BPLER sample 5':0, 'BPLER sample 6':0, 'HMLER sample 1':1, 'HMLER sample 2':1, 'HMLER sample 3':1, 'HMLER sample 4':1, 'HMLER sample 5':1, 'HMLER sample 6':1}
        self.initData(atype, atypes, ahash)
        
        
    def getArpaia(self, tn=1):
        self.prepareData("TNB24","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'MDA-MB-231 WT':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c growth condition (ch1)')
        atypes = ['Normoxia','Hypoxia']
        ahash = {'Normoxia':0,'Hypoxia':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getArpaia2(self, tn=1):
        self.prepareData("TNB24","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c growth condition (ch1)')
        atypes = ['Normoxia','Hypoxia']
        ahash = {'Normoxia':0,'Hypoxia':1}
        self.initData(atype, atypes, ahash)
        
    def getBarbazan(self, tn=1):
        self.prepareData("TNB25","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c cell type (ch1)')
        atypes = ['PB','CTC']
        ahash = {'non-specific immunoisolated cells':0,'circulating tumor cells':1}
        self.initData(atype, atypes, ahash)
        
    def getleon(self, tn=1):
        self.prepareData("TNB26","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['PB','CTC']
        ahash = {'Control 40':0, 'Control 42':0, 'Control 51':0, 'Control 52':0, 'Control 53':0, 'Control 54':0,'patient 9 progresion':1, 'patient 8 progresion':1, 'patient 24 progresion':1, 'patient 21 progresion':1, 'Patient 2 progresion':1, 'patient 17 progresion':1, 'patient 11 progresiton':1, 'Patien 5 progresion':1, 'patien 13 progresion':1}
        self.initData(atype, atypes, ahash)
        
    def getleonbasal(self, tn=1):
        self.prepareData("TNB26","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['PB','CTC']
        ahash = {'Control 40':0, 'Control 42':0, 'Control 51':0, 'Control 52':0, 'Control 53':0, 'Control 54':0,'patient 9 basal':1, 'patient 8 basal':1, 'patient 24 basal':1, 'patient 21 basal':1, 'Patient 2 basal':1, 'patient 17 basal':1, 'patient 11 progresiton':1, 'Patien 5 basal':1, 'patien 13 basal':1}
        self.initData(atype, atypes, ahash)
        
    def getleonbasalprogression(self, tn=1):
        self.prepareData("TNB26","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c title')
        atypes = ['Control','CTC_basal','CTC_Progression']
        ahash = {'Control 40':0, 'Control 42':0, 'Control 51':0, 'Control 52':0, 'Control 53':0, 'Control 54':0,'patient 9 basal':1, 'patient 8 basal':1, 'patient 24 basal':1, 'patient 21 basal':1, 'Patient 2 basal':1, 'patient 17 basal':1, 'patient 11 progresiton':1, 'Patien 5 basal':1, 'patien 13 basal':1,'patient 9 progresion':2, 'patient 8 progresion':2, 'patient 24 progresion':2, 'patient 21 progresion':2, 'Patient 2 progresion':2, 'patient 17 progresion':2, 'patient 11 progresiton':2, 'Patien 5 progresion':2, 'patien 13 progresion':2}
        self.initData(atype, atypes, ahash)
        
        
    def getMaheswaran(self, tn=1):
        self.prepareData("TNB28","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c rpl15 overexpressing (ch1)')
        atypes = ['control','RPL15 overexpressing']
        ahash = {'control':0,'RPL15 overexpressing':1}
        self.initData(atype, atypes, ahash)
        
    def getPopovici(self, tn=1):
        self.prepareData("BC30","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0': 0, '1':1}
        self.initData(atype, atypes, ahash)
        
    def gettests(self, tn=1, tb=0):
        self.prepareData("CRC11","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)
        
    def getWasnik2021(self, tn=1, tb=0):
        self.prepareData("SS36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c tissue source (ch1)')
        atypes = ['donor lungs', 'fibrotic (IPF) lungs']
        ahash = {'donor lungs':0, 'fibrotic (IPF) lungs':1}
        self.initData(atype, atypes, ahash)
        
    def getDeng2021(self, tn=1, tb=0):
        self.prepareData("SS37","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c diagnosis (ch1)')
        atypes = ['control donor', 'idiopathic pulmonary fibrosis (IPF)']
        ahash = {'control donor':0, 'idiopathic pulmonary fibrosis (IPF)':1}
        self.initData(atype, atypes, ahash)
        
    def getFurusawa2020(self, tn=1, tb=0):
        self.prepareData("SS39","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c diagnosis (ch1)')
        atypes = ['control', 'ipf']
        ahash = {'control':0, 'ipf':1}
        self.initData(atype, atypes, ahash)
        
    def getKonigsberg2020(self, tn=1, tb=0):
        self.prepareData("SS38","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['normal lung sample', 'IPF lung sample']
        ahash = {'normal lung sample':0, 'IPF lung sample':1}
        self.initData(atype, atypes, ahash)
        
    def getBoesch2020(self, tn=1, tb=0):
        self.prepareData("SS40","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c condition (ch1)')
        atypes = ['control', 'idiopathic pulmonary fibrosis (IPF)']
        ahash = {'control':0, 'idiopathic pulmonary fibrosis (IPF)':1}
        self.initData(atype, atypes, ahash)
        
    def getZhu2021(self, tn=1, tb=0):
        self.prepareData("SS41","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type (ch1)')
        atypes = ['normal', 'diseased']
        ahash = {'normal':0, 'diseased':1}
        self.initData(atype, atypes, ahash)
        
    def getPGNASH2(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')
        atypes = ['RL_726_CDAA_WT_4_S11', 'R729_CDAA_WT_3_S10', 'R726__CDAA_WT_2_S9', 'R733_CDAA_WT_5_S12', 'L724_CDAA_WT_1_S8', 'L_733_CDAA_KO_3_S15', 'L_719_CDAA_KO_2_S14', 'R_719_CDAA_KO_1_S13', 'RR733_CDAA_KO_5_S16']
        ahash = {'RL_726_CDAA_WT_4_S11':0, 'R729_CDAA_WT_3_S10':1, 'R726__CDAA_WT_2_S9':2, 'R733_CDAA_WT_5_S12':3, 'L724_CDAA_WT_1_S8':4, 'L_733_CDAA_KO_3_S15':5, 'L_719_CDAA_KO_2_S14':6, 'R_719_CDAA_KO_1_S13':7, 'RR733_CDAA_KO_5_S16':8}
        self.initData(atype, atypes, ahash)
        
    def getPGNASH3(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')
        atypes = ['CDAA-WT', 'CDAA-KO*']
        ahash = {'RL_726_CDAA_WT_4_S11':0, 'R729_CDAA_WT_3_S10':0, 'R726__CDAA_WT_2_S9':0, 'R733_CDAA_WT_5_S12':0, 'L724_CDAA_WT_1_S8':0, 'L_733_CDAA_KO_3_S15':1, 'L_719_CDAA_KO_2_S14':1, 'RR733_CDAA_KO_5_S16':1}
        self.initData(atype, atypes, ahash)
        
    def getPGNASH16(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')
        atypes = ['CDAA-WT', 'CDAA-KO*']
        ahash = {'RL_726_CDAA_WT_4_S11':0, 'R729_CDAA_WT_3_S10':0, 'R726__CDAA_WT_2_S9':0, 'L_733_CDAA_KO_3_S15':1, 'L_719_CDAA_KO_2_S14':1,'RR733_CDAA_KO_5_S16':1}
        self.initData(atype, atypes, ahash)        
        
    def getPGNASH13(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')
        atypes = ['CDAA-WT', 'CDAA-KO*']
        ahash = {'RL_726_CDAA_WT_4_S11':0, 'R729_CDAA_WT_3_S10':0, 'R726__CDAA_WT_2_S9':0, 'L_733_CDAA_KO_3_S15':1, 'L_719_CDAA_KO_2_S14':1,'R_719_CDAA_KO_1_S13':1}
        self.initData(atype, atypes, ahash)    
        
    def getPGNASH9101415(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('ArrayID')
        atypes = ['CDAA-WT', 'CDAA-KO*']
        ahash = {'R729_CDAA_WT_3_S10':0, 'R726__CDAA_WT_2_S9':0, 'L_733_CDAA_KO_3_S15':1, 'L_719_CDAA_KO_2_S14':1}
        self.initData(atype, atypes, ahash)          
        
    def getPGNASH(self, tn=1, tb=0):
        self.prepareData("PGSS1","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment')
        atypes = ['CDAA-WT', 'CDAA-KO']
        ahash = {'CDAA-WT':0, 'CDAA-KO':1}
        self.initData(atype, atypes, ahash)

    def getPGNASH2022(self, tn=1, tb=0):
        self.prepareData("PGSS3","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment')
        atypes = ['CSAA-WT', 'CSAA-KO', 'CDAA-WT', 'CDAA-KO']
        ahash = {'CSAA-WT':0, 'CSAA-KO':1, 'CDAA-WT':2, 'CDAA-KO':3}
        self.initData(atype, atypes, ahash)        
        
    def getLefebvre2017(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV3")
        atype = self.h.getSurvName('c liver status')
        atypes = ['no NASH', "NASH"]
        ahash = {'no NASH':0, "no NASH":1}
        self.initData(atype, atypes, ahash)   
        
        
    def getAhrens(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV9")
        atype = self.h.getSurvName('c group')
        atypes = ['Control', 'Nash']
        ahash = {'Control':0, 'Nash':1}
        self.initData(atype, atypes, ahash)           

    def getArendt(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV4")
        atype = self.h.getSurvName('c diagnosis')
        atypes = ['HC', 'NASH']
        ahash = {'HC':0, 'NASH':1}
        self.initData(atype, atypes, ahash)  
        
    def getArendt2(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV4")
        atype = self.h.getSurvName('c fibrosis (stage)')
        atypes = ['0', '1', '2', '3', '4']
        ahash = {'0':0, '1':1, '2':2, '3':3, '4':4}
        self.initData(atype, atypes, ahash)  
        
    def getPinyol(self, tn=1, ta=0, tb=0):
        self.prepareData("LIV53")
        atype = self.h.getSurvName('c tissue')
        atypes = ['Healthy liver', 'NASH liver']
        ahash = {'Healthy liver':0, 'NASH liver':1}
        self.initData(atype, atypes, ahash)        
        
    def getebright2(self, tn=1):
        self.prepareData("TNB29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c number uniquely mapped reads (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'pass':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c ptprc expression (ch1)')
        atypes = ['fail', 'pass']
        ahash = {'fail': 0, 'pass':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        
    def getebright3(self, tn=1):
        self.prepareData("TNB29","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        atypes = ['0', '1']
        ahash = {'0': 0, '1':1} 
        self.initData(atype, atypes, ahash)
        
    def getlips2021(self, tn=1):
        self.prepareData("TNB30","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c tissue type (ch1)')
        atypes = ['biopsy', 'surgery_specimen']
        ahash = {'biopsy': 0, 'surgery_specimen':1} 
        self.initData(atype, atypes, ahash)
        
    def getlips2(self, tn=1):
        self.prepareData("TNB30","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c chemotherapy (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'6x ddAC':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c recurrence (ch1)')
        atypes = ['0', '1']
        ahash = {'0': 0, '1':1} 
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getlips3(self, tn=1):
        self.prepareData("TNB30","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c neoadjuvant response index (nri) (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'0':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c tissue type (ch1)')
        atypes = ['biopsy', 'surgery_specimen']
        ahash = {'biopsy': 0, 'surgery_specimen':1} 
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getAceto2016(self, tn=1):
        self.prepareData("TNB31","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c progression (ch1)')
        atypes = ['Bone (+)', 'Visceral (+)']
        ahash = {'Bone (+)': 0, 'Visceral (+)':1} 
        self.initData(atype, atypes, ahash)
        
    def getDe2021(self, tn=1, tb=0):
        self.prepareData("SS42","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease (ch1)')
        atypes = ['DONOR', 'IPF']
        ahash = {'DONOR':0, 'IPF':1}
        self.initData(atype, atypes, ahash)
        
    def getDePianto(self, tn=1):
        self.prepareData("SS43","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'Biopsy':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease (ch1)')
        atypes = ['Normal', 'IPF']
        ahash = {'Normal': 0, 'IPF':1} 
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getDePianto2(self, tn=1):
        self.prepareData("SS43","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'Digest':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease (ch1)')
        atypes = ['Normal', 'IPF']
        ahash = {'Normal': 0, 'IPF':1} 
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getDePianto3(self, tn=1):
        self.prepareData("SS43","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'Bronchoalveolar lavage':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease (ch1)')
        atypes = ['Normal', 'IPF']
        ahash = {'Normal': 0, 'IPF':1} 
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)
        
    def getAnderson2021(self, tn=1, tb=0):
        self.prepareData("TNB32","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'EFM192A':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c genotype (ch1)')
        atypes = ['Parental', 'Drug-Tolerant Persister']
        ahash = {'P':0, 'DTP':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    

        
    def getFu2019(self, tn=1, tb=0):
        self.prepareData("TNB33","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c anti-her2 resistance (ch1)')
        #ahash = {'Normoxia': 0}
        ahash = {'Sensitive':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['Lapatinib plus trastuzumab, 24h', 'Lapatinib, 24h']
        ahash = {'Lapatinib plus trastuzumab, 24h':0, 'Lapatinib, 24h':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)

        
    def getBC16(self):
        self.getSurvival("BC16.1")
        


        
    def getGao2020ipf(self, tn=1, tb=0):
        self.prepareData("COV424","/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type')
        #ahash = {'Normoxia': 0}
        ahash = {'Unsorted':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease')
        atypes = ['Control', 'IPF']
        ahash = {'Control':0, 'IPF':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getGaoCD31(self, tn=1, tb=0):
        self.prepareData("COV424","/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type')
        #ahash = {'Normoxia': 0}
        ahash = {'CD31+':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease')
        atypes = ['Control', 'IPF']
        ahash = {'Control':0, 'IPF':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getGaoCD45(self, tn=1, tb=0):
        self.prepareData("COV424","/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type')
        #ahash = {'Normoxia': 0}
        ahash = {'CD45+':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease')
        atypes = ['Control', 'IPF']
        ahash = {'Control':0, 'IPF':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 

    def getGaoCD90(self, tn=1, tb=0):
        self.prepareData("COV424","/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type')
        #ahash = {'Normoxia': 0}
        ahash = {'CD90+':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease')
        atypes = ['Control', 'IPF']
        ahash = {'Control':0, 'IPF':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getGaoEPCAM(self, tn=1, tb=0):
        self.prepareData("COV424","/booleanfs2/sahoo/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type')
        #ahash = {'Normoxia': 0}
        ahash = {'EPCAM+':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c disease')
        atypes = ['Control', 'IPF']
        ahash = {'Control':0, 'IPF':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getmicrobes(self, tn=1, tb=0):
        self.prepareData("SS44","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ibd_or_not')
        atypes = ['Yes', 'No']
        ahash = {'Yes':0, 'No':1}
        self.initData(atype, atypes, ahash)     
        
        
    def getHoadley(self, tn=1, tb=0):
        self.prepareData("TNB34","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('status')
        #ahash = {'Normoxia': 0}
        ahash = {'0':0}
        hval = [1 if i in ahash else None for i in atype]
        atype = self.h.getSurvName('c TREATMENT')
        atypes = ['chemo only', 'chemo and tam']
        ahash = {'chemo only':0, 'chemo and tam':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)      
        
#    def getBC37(self, tn=1, tb=0):
#        self.prepareData("BC37","/Users/sinha7290/public_html/Hegemon/explore.conf")
#        atype = self.h.getSurvName('c Treatment')
#        atypes = ['HT', 'NONE']
#        ahash = {'HT':0, 'NONE':1}
#        self.initData(atype, atypes, ahash) 
        
    def getML8(self):
        self.getSurvival("ML8")
        
#    def getSurvival(self, dbid = "TNB34"):
#        self.prepareData(dbid)
#        atype = self.h.getSurvName("status")
#        atypes = ['Censor', 'Relapse']
#        ahash = {"0": 0, "1":1}
#        self.initData(atype, atypes, ahash)       

    def getwolf(self, tn=1, tb=0):
        self.prepareData("TNB36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c arm_ch1')
        #ahash = {'THP': 0}
        ahash = {'T-DM1/P':0}
        hval = [1 if i in ahash else None for i in atype]        
        atype = self.h.getSurvName('c pcr_ch1')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)



    def getwolf2(self, tn=1, tb=0):
        self.prepareData("TNB36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr_ch1')
        #ahash = {'Normoxia': 0}
        ahash = {'0':0}
        hval = [1 if i in ahash else None for i in atype]        
        atype = self.h.getSurvName('c arm_ch1')
        atypes = ['TH Control', 'THP', 'T-DM1/P']
        ahash = {'TH Control':0, 'THP':1, 'T-DM1/P':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)   
        
    def getwolf3(self, tn=1, tb=0):
        self.prepareData("TNB36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c hr_ch1')
        #ahash = {'Normoxia': 0}
        ahash = {'1':0}
        hval = [1 if i in ahash else None for i in atype]        
        atype = self.h.getSurvName('c pcr_ch1')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getwolf4(self, tn=1, tb=0):
        self.prepareData("TNB36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('hpcr')
        ahash = {'2':0}
        hval = [1 if i in ahash else None for i in atype]        
        atype = self.h.getSurvName('c arm_ch1')
        atypes = ['TH Control', 'THP', 'T-DM1/P']
        ahash = {'TH Control':0, 'THP':1, 'T-DM1/P':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)   
        
    def getwolf5(self, tn=1, tb=0):
        self.prepareData("TNB35","/Users/sinha7290/public_html/Hegemon/explore.conf")       
        atype = self.h.getSurvName('c arm_ch1')
        atypes = ['TH Control', 'THP', 'T-DM1/P']
        ahash = {'TH Control':0, 'THP':1, 'T-DM1/P':2} 
        self.initData(atype, atypes, ahash)         
        
    def getfina(self, tn=1, tb=0):
        self.prepareData("TNB37","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample class_ch1')
        atypes = ['MDA-MB-231', 'Primary tumor(I)', 'Lymph-node', 'CTC/DTC', 'Lung']
        ahash = {'MDA-MB-231':0, 'Primary tumor(I)':1, 'Lymph-node':2, 'CTC/DTC':3, 'Lung':4}
        self.initData(atype, atypes, ahash)                
        if (tn == 2):
            atypes = ['Primary tumor', 'Lymph-node']
            ahash = {'Primary tumor':0, 'Lymph-node':1}
        if (tn == 3):
            atypes = ['Lymph-node', 'CTC/DTC']
            ahash = {'Lymph-node':0, 'CTC/DTC':1}
        if (tn == 4):
            atypes = ['CTC/DTC', 'Lung']
            ahash = {'CTC/DTC':0, 'Lung':1}    
        if (tn == 5):
            atype = self.h.getSurvName('c tissue_ch1')
            atypes = ['Inguinal Primary Tumor', 'Axillary Lymph-Node']
            ahash = {'Inguinal Primary Tumor (d)':0, 'Axillary Lymph-Node (c)':1}            
        self.initData(atype, atypes, ahash) 
        
        
        
    def getfina2(self, tn=1, tb=0):
        self.prepareData("TNB38","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample class_ch1')
        atypes = ['MDA-MB-231', 'Primary tumor', 'Lymph-node', 'CTC', 'Lung']
        ahash = {'MDA-MB-231':0, 'Primary tumor':1, 'Lymph-node':2, 'CTC':3, 'Lung':4}
        self.initData(atype, atypes, ahash)                
        if (tn == 2):
            atypes = ['Primary tumor', 'Lymph-node']
            ahash = {'Primary tumor':0, 'Lymph-node':1}
        if (tn == 3):
            atypes = ['Lymph-node', 'CTC']
            ahash = {'Lymph-node':0, 'CTC':1}
        if (tn == 4):
            atypes = ['CTC', 'Lung']
            ahash = {'CTC':0, 'Lung':1}              
        self.initData(atype, atypes, ahash)  
        
    def getfina3(self, tn=1, tb=0):
        self.prepareData("TNB37","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample class_ch1')
        atypes = ['Primary tumor(I)', 'Lymph-node', 'CTC', 'Lung','MDA-MB-231']
        ahash = {'Primary tumor(I)':0, 'Lymph-node':1, 'CTC/DTC':2, 'Lung':3,'MDA-MB-231':4}
        self.initData(atype, atypes, ahash)                
        if (tn == 2):
            atypes = ['Primary tumor', 'Lymph-node']
            ahash = {'Primary tumor':0, 'Lymph-node':1}
        if (tn == 3):
            atypes = ['Lymph-node', 'CTC/DTC']
            ahash = {'Lymph-node':0, 'CTC/DTC':1}
        if (tn == 4):
            atypes = ['CTC/DTC', 'Lung']
            ahash = {'CTC/DTC':0, 'Lung':1}    
        if (tn == 5):
            atype = self.h.getSurvName('c tissue_ch1')
            atypes = ['Inguinal Primary Tumor', 'Axillary Lymph-Node']
            ahash = {'Inguinal Primary Tumor (d)':0, 'Axillary Lymph-Node (c)':1}            
        self.initData(atype, atypes, ahash)   
        
    def getFINADV(self, tn=1, tb=0):
        self.prepareData("TNB60","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('set')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'D':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c sample class (ch1)')
        atypes = ['Primary tumor(A)','Primary tumor(I)', 'Lymph-node', 'CTC', 'Lung','MDA-MB-231']
        ahash = {'Primary tumor(A)':0,'Primary tumor(I)':1, 'Lymph-node':2, 'CTC/DTC':3, 'Lung':4,'MDA-MB-231':5, 'CTC': 3}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)     
        
    def getFINADV3(self, tn=1, tb=0):
        self.prepareData("TNB60","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('set')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c sample class (ch1)')
        atypes = ['Primary tumor(A)','Primary tumor(I)', 'Lymph-node', 'CTC', 'Lung','MDA-MB-231']
        ahash = {'Primary tumor(A)':0,'Primary tumor(I)':1, 'Lymph-node':2, 'Lung':4,'MDA-MB-231':5, 'CTC': 3}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)          
        
        
    def getFINADV2(self, tn=1, tb=0):
        self.prepareData("TNB60","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('set')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'D':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c sample class (ch1)')
        atypes = ['MDA-MB-231','Primary tumor(A)','Primary tumor(I)', 'Lymph-node', 'CTC', 'Lung']
        ahash = {'MDA-MB-231':0,'Primary tumor(A)':1,'Primary tumor(I)':2, 'Lymph-node':3, 'CTC/DTC':4, 'Lung':5, 'CTC': 4}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)          
        
        
    def getMolloy(self, tn=1, tb=0):
        self.prepareData("TNB39","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ctc_status_ch1')
        atypes = ['No CTC', 'CTC']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)           
        
    def getMolloy2(self, tn=1, tb=0):
        self.prepareData("TNB39","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c lymph node metastasis_ch1')
        #ahash = {'Normoxia': 0}
        ahash = {'0':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c ctc_status_ch1')
        atypes = ['No CTC', 'CTC']
        ahash = {'0':0, '1':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getbardia1(self, tn=1, tb=0):
        self.prepareData("TNB20","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('patient_CTC')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'EPCAM CTC chip':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c blood draw_ch1')
        atypes = ['1', '2', '3', '4', '5']
        ahash = {'1':0, '2':1, '3':2, '4':3, '5':4}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    

    def getbardia0(self, tn=1, tb=0):
        self.prepareData("TNB20","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('healthy_CTC')
        #ahash = {'EPCAM CTC chip': 0}
        ahash = {'IgG CTC chip':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c blood draw_ch1')
        atypes = ['1', '2', '3', '4', '5']
        ahash = {'1':0, '2':1, '3':2, '4':3, '5':4}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)         
        
    def getbardia12(self, tn=1, tb=0):
        self.prepareData("TNB20","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('patient_CTC')
        #ahash = {'EPCAM CTC chip': 0}
        ahash = {'IgG CTC chip':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c blood draw_ch1')
        atypes = ['1', '2', '3', '4', '5']
        ahash = {'1':0, '2':1, '3':2, '4':3, '5':4}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 

        
    def getbardia3(self, tn=1, tb=0):
        self.prepareData("TNB20","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('patient_CTC time')      
        atypes = ['EPCAM CTC chip1', 'IgG CTC chip1', 'EPCAM CTC chip2', 'IgG CTC chip2', 'EPCAM CTC chip3', 'IgG CTC chip3', 'EPCAM CTC chip4', 'IgG CTC chip4', 'EPCAM CTC chip5', 'IgG CTC chip5']
        ahash = {'EPCAM CTC chip1':0, 'IgG CTC chip1':1, 'EPCAM CTC chip2':2, 'IgG CTC chip2':3, 'EPCAM CTC chip3':4, 'IgG CTC chip3':5, 'EPCAM CTC chip4':6, 'IgG CTC chip4':7, 'EPCAM CTC chip5':8, 'IgG CTC chip5':9}
        self.initData(atype, atypes, ahash) 
        
    def getbardiatest(self, tn=1, tb=0):
        self.prepareData("TNB20","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c blood processed by_ch1')
        ahash = {'IgG CTC chip': 0}
        #ahash = {'EPCAM CTC chip':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c blood draw_ch1')
        atypes = ['1', '2', '3', '4', '5']
        ahash = {'1':0, '2':1, '3':2, '4':3, '5':4}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)           
        
    def getfraley(self, tn=1, tb=0):
        self.prepareData("EMT3","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Title')
        atypes = ['KC1', 'KC2']
        ahash = {'KC1':0, 'KC2':1}
        self.initData(atype, atypes, ahash)        
        
    def getTsai1(self, tn=1, tb=0):
        self.prepareData("SS45","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c A_T')
        atypes = ['Young0', 'Young4', 'Young8', 'Young12', 'Young16', 'Young20', 'Young24']
        ahash = {'Young0':0, 'Young4':1, 'Young8':2, 'Young12':3, 'Young16':4, 'Young20':5, 'Young24':6}
        self.initData(atype, atypes, ahash)    
        
    def getTsai2(self, tn=1, tb=0):
        self.prepareData("SS45","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c A_T')
        atypes = ['Old0', 'Old4', 'Old8', 'Old12', 'Old16', 'Old20', 'Old24']
        ahash = {'Old0':0, 'Old4':1, 'Old8':2, 'Old12':3, 'Old16':4, 'Old20':5, 'Old24':6}
        self.initData(atype, atypes, ahash)   
        
    def getISPY1(self, tn=1, tb=0):
        self.prepareData("TNB41","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pathological complete response (pcr)_ch1')
        atypes = ['Yes', 'No']
        ahash = {'Yes':0, 'No':1}
        self.initData(atype, atypes, ahash)  
        
    def getISPY1_310(self, tn=1, tb=0):
        self.prepareData("TNB42","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pathologic_response_pcr_rd (ch1)')
        atypes = ['pCR', 'RD']
        ahash = {'pCR':0, 'RD':1}
        self.initData(atype, atypes, ahash)           
        
    def getISPY1_248(self, tn=1, tb=0):
        self.prepareData("TNB43","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c recurrence_yes is 1 (ch1)')
        atypes = ['1', '0']
        ahash = {'1':0, '0':1}
        self.initData(atype, atypes, ahash)  
        
    def getKhambata2007(self, tn=1, tb=0):
        self.prepareData("CRC11","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sex')
        atypes = ['MALE', 'FEMALE']
        ahash = {'MALE':0, 'FEMALE':1}
        self.initData(atype, atypes, ahash)           
        
    def getDas2020(self, tn=1, tb=0):
        self.prepareData("mlt.12mainF2", "/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c c1uster')
        atypes = ['Green','yellow', 'gray']
        ahash ={'org_H4_Colon_p35_S7_L003':0,'org_H14_Colon_p13_S1_L003':0,'org_CD21_LC_P6_S8_L003':1,'org_CD28_R_p8_S9_L003':1,'org_CD30_LC_p7_S10_L003':2,'org_CD47_LC_p8_S11_L003':2,'org_CD26_LC_p8_S12_L003':1,'org_CD59_R_p4_S13_L003':2,'org_CD60_LC_p6_S14_L003':2,'org_CD11_LC_p10_S16_L003':1,'org_CD20_LC_p7_S17_L003':1,'org_CD19_LC_p9_S18_L003':1,'org_CD24_RC_p9_S19_L003':2,'org_CD50_R_p6_S21_L003':2,'org_CD52_R_p5_S22_L003':2,'org_CD63_LC_p4_S23_L003':2,'org_CD27_R_NI_p4_S25_L003':2,'org_CD32_LC_p7_S28_L003':2,'org_CD35_LC_P6_S29_L003':2,'org_CD42_R_p11_S30_L003':2,'org_CD58_LC_p4_S31_L003':2,'org_CD62_R_p3_S33_L003':2,'H14_S90_L003':0,'H19_S95_L003':0}
        self.initData(atype, atypes, ahash)  
        
    def getDas2(self, tn=1, tb=0):
        self.prepareData("mlt.12mainF2", "/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        atypes = ['CD', 'H']
        ahash ={'EDM_CD21-LC p11':0, 'EDM_CD59-R':0, 'EDM_CD24-RC p10':0, 'EDM_CD32-LC p12':0, 'EDM_CD42-R p7':0, 'EDM_CD58-LC':0, 'EDM_H14-Colon':1, 'EDM_H12-Colon p14':1}
        self.initData(atype, atypes, ahash)    

    def getISPY2_1_1(self, tn=1, tb=0):
        self.prepareData("TNB35","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr_ch1')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)
  
    def getISPY2_1_2(self, tn=1, tb=0):
        self.prepareData("TNB36","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr_ch1')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)
        
        
    def getISPY2_2(self, tn=1, tb=0):
        self.prepareData("TNB44","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr (ch1)')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)  
        
    def getISPY2_3_1(self, tn=1, tb=0):
        self.prepareData("TNB45","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr (ch1)')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash) 
        
    def getISPY2_3_2(self, tn=1, tb=0):
        self.prepareData("TNB46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pcr (ch1)')
        atypes = ['0', '1']
        ahash = {'0':0, '1':1}
        self.initData(atype, atypes, ahash)      
        
    def getFrancesc2022(self, tn=1, tb=0):
        self.prepareData("TNB47","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type_ch1')
        atypes = ['ctc_single', 'ctc_cluster', 'ctc_cluster_wbc']
        ahash = {'ctc_single':0, 'ctc_cluster':1, 'ctc_cluster_wbc':2}
        self.initData(atype, atypes, ahash)   
        
    def getFrancesc2(self, tn=1, tb=0):
        self.prepareData("TNB47","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c origin_timepoint')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'xenograft_resting':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c sample type_ch1')
        atypes = ['ctc_single', 'ctc_cluster', 'ctc_cluster_wbc']
        ahash = {'ctc_single':0, 'ctc_cluster':1, 'ctc_cluster_wbc':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    
        
    def getFrancesc3(self, tn=1, tb=0):
        self.prepareData("TNB47","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type_ch1')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'ctc_cluster_wbc':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c timepoint_ch1')
        atypes = ['active', 'resting']
        ahash = {'active':0, 'resting':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    
        
    def getpawitan2(self, tn=1):
        self.prepareData("BC3","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c SUBTYPE')
        atypes = ['Normal Like', 'Basal', 'Luminal A', 'Luminal B', 'ERBB2']
        ahash = {'Normal Like':0, 'Basal':1, 'Luminal A':2, 'Luminal B':3, 'ERBB2':4}
        self.initData(atype, atypes, ahash)         
        
    def getsabatier(self, tn=1):
        self.prepareData("BC9","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c molecular subtype')
        atypes = ['Normal', 'Basal', 'LuminalA', 'LuminalB', 'ERBB2']
        ahash = {'Normal':0, 'Basal':1, 'LuminalA':2, 'LuminalB':3, 'ERBB2':4}
        self.initData(atype, atypes, ahash)    
        
    def getguedj(self, tn=1):
        self.prepareData("BC15","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c CIT classification')
        atypes = ['normL','basL','lumA', 'lumB', 'lumC']
        ahash = {'normL':0,'basL':1,'lumA':2, 'lumB':3, 'lumC':4}
        self.initData(atype, atypes, ahash) 
        
    def getdedeurwaerder(self, tn=1):
        self.prepareData("BC23","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c subtypeihc')
        atypes = ['Normal','Basal','LumA', 'LumB', 'HER2']
        ahash = {'Normal':0,'Basal':1,'LumA':2, 'LumB':3, 'HER2':4}
        self.initData(atype, atypes, ahash)    
        
    def getSircoulomb(self, tn=1):
        self.prepareData("BC24","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c type')
        atypes = ['Normal','Basal','LuminalA', 'ERBB2']
        ahash = {'Normal':0,'Basal':1,'LuminalA':2, 'ERBB2':3}
        self.initData(atype, atypes, ahash)   
        
    def getdeRonde(self, tn=1):
        self.prepareData("BC42","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c intrinsic molecular subtype')
        atypes = ['Normal','Basal','LumA', 'LumB', 'Her2']
        ahash = {'Normal':0,'Basal':1,'LumA':2, 'LumB':3, 'Her2':4}
        self.initData(atype, atypes, ahash)        
        
    def getAzim(self, tn=1):
        self.prepareData("BC46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50')
        atypes = ['Normal','Basal','LumA', 'LumB', 'Her2']
        ahash = {'Normal':0,'Basal':1,'LumA':2, 'LumB':3, 'Her2':4}
        self.initData(atype, atypes, ahash)     
        
    def getAzimBasal(self, tn=1):
        self.prepareData("BC46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50')
        atypes = ['Normal','Basal']
        ahash = {'Normal':0,'Basal':1}
        self.initData(atype, atypes, ahash)         
        
    def getAzimLumA(self, tn=1):
        self.prepareData("BC46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50')
        atypes = ['Normal','LumA']
        ahash = {'Normal':0,'LumA':1}
        self.initData(atype, atypes, ahash) 
        
    def getAzimLumB(self, tn=1):
        self.prepareData("BC46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50')
        atypes = ['Normal', 'LumB']
        ahash = {'Normal':0, 'LumB':1}
        self.initData(atype, atypes, ahash) 
        
    def getAzimHer2(self, tn=1):
        self.prepareData("BC46","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50')
        atypes = ['Normal', 'Her2']
        ahash = {'Normal':0, 'Her2':1}
        self.initData(atype, atypes, ahash)         
               
        
    def getjonsson2012(self, tn=1):
        self.prepareData("BC58","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c hu subtype')
        atypes = ['Normal','Basal','LumA', 'LumB', 'ERBB2']
        ahash = {'Normal':0,'Basal':1,'LumA':2, 'LumB':3, 'ERBB2':4}
        self.initData(atype, atypes, ahash)   
        
    def getxavier(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        atypes = ['Normal','Basal','LumA', 'LumB', 'Her2']
        ahash = {'Normal':0,'Basal':1,'LumA':2, 'LumB':3, 'Her2':4}
        self.initData(atype, atypes, ahash)           
        
        
    def getxavier2(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'Her2':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c immune cluster (ch1)')
        atypes = ['ClusterA','ClusterB','ClusterC']
        ahash = {'ClusterA':0,'ClusterB':1,'ClusterC':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getcampo2022(self, tn=1):
        self.prepareData("TNB49","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c subject id_ch1')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'19065':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('ArrayId')
        atypes = ['circulating tumor cells sorted from peripheral blood','Ascites', 'peripheral blood; normal leukocyte population']
        #atypes = ['circulating tumor cells sorted from peripheral blood', 'peripheral blood; normal leukocyte population']
        #ahash = {'circulating tumor cells sorted from peripheral blood':0,'Pleural effusion':1,'peripheral blood; normal leukocyte population':2}
        ahash = {'GSM3122803':0,'GSM3122831':1,'GSM3122861':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getcampo2(self, tn=1):
        self.prepareData("TNB49","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name_ch1')
        atypes = ['peripheral blood','metastases', 'circulating tumor cells sorted from peripheral blood']
        ahash = {'peripheral blood':0,'metastases':1, 'circulating tumor cells sorted from peripheral blood':2}
        self.initData(atype, atypes, ahash)        
        
    def getcampo3(self, tn=1):
        self.prepareData("TNB49","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name_ch1')
        atypes = ['metastases', 'circulating tumor cells sorted from peripheral blood']
        ahash = {'metastases':0, 'circulating tumor cells sorted from peripheral blood':1}
        self.initData(atype, atypes, ahash)         
        
    def getshen2022(self, tn=1):
        self.prepareData("TNB50","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment.type (ch1)')
        atypes = ['FEC/TX','FEC/TX+H']
        ahash = {'FEC/TX':0,'FEC/TX+H':1}
        self.initData(atype, atypes, ahash)     
        
    def getPGTAMS(self, tn=1):
        self.prepareData("SS22","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        atypes = ['GIVKO', 'WT']
        ahash = {'GIVKO':0,'WT':1}
        self.initData(atype, atypes, ahash) 
    def getTUMOR(self, tn=1):
        self.prepareData("SS21","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Type')
        atypes = ['GIVKO', 'WT']
        ahash = {'GIVKO':0,'WT':1}
        self.initData(atype, atypes, ahash)        
        
    def getHatzisBasal(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c pam50_class')
        atypes = ['Normal','Basal']
        ahash = {'Normal':0,'Basal':1}
        self.initData(atype, atypes, ahash)

    def getHatzisLumA(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c pam50_class')
        atypes = ['Normal','LumA']
        ahash = {'Normal':0,'LumA':1}
        self.initData(atype, atypes, ahash)

    def getHatzisLumB(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c pam50_class')
        atypes = ['Normal','LumB']
        ahash = {'Normal':0,'LumB':1}
        self.initData(atype, atypes, ahash)

    def getHatzisHer2(self, tn=1):
        self.prepareData("BC8","/Users/sinha7290/public_html/Hegemon/explore.conf")   
        atype = self.h.getSurvName('c pam50_class')
        atypes = ['Normal','Her2']
        ahash = {'Normal':0,'Her2':1}
        self.initData(atype, atypes, ahash)


    def getxavierBasal(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        atypes = ['Normal','Basal']
        ahash = {'Normal':0,'Basal':1}
        self.initData(atype, atypes, ahash)
        
    def getxavierLumA(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        atypes = ['Normal','LumA']
        ahash = {'Normal':0,'LumA':1}
        self.initData(atype, atypes, ahash)   
        
    def getxavierLumB(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        atypes = ['Normal', 'LumB']
        ahash = {'Normal':0, 'LumB':1}
        self.initData(atype, atypes, ahash)         
        
    def getxavierHer2(self, tn=1):
        self.prepareData("TNB48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pam50 (ch1)')
        atypes = ['Normal', 'Her2']
        ahash = {'Normal':0, 'Her2':1}
        self.initData(atype, atypes, ahash)    
        
    def getsun2019(self, tn=1):
        self.prepareData("TNB51","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['4.3% ethanol treatment', 'Early recovery ', 'Late recovery','Mock treatment', ]
        ahash = {'mock treatment (growth medium only) for 3hrs':3, '4.3% ethanol treatment for 3hrs':0, '1hr recovery after 3hr ethanol treatment':1, '2hr recovery after 3hr ethanol treatment':1, '3hr recovery after 3hr ethanol treatment':1, '4hr recovery after 3hr ethanol treatment':2, '8hr recovery after 3hr ethanol treatment':2, '12hr recovery after 3hr ethanol treatment':2}
        self.initData(atype, atypes, ahash) 
        
    def getsun2(self, tn=1):
        self.prepareData("TNB51","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['4.3% ethanol treatment for 3hrs', '1hr recovery after 3hr ethanol treatment', '2hr recovery after 3hr ethanol treatment', '3hr recovery after 3hr ethanol treatment', '4hr recovery after 3hr ethanol treatment', '8hr recovery after 3hr ethanol treatment', '12hr recovery after 3hr ethanol treatment']
        ahash = {'4.3% ethanol treatment for 3hrs':0, '1hr recovery after 3hr ethanol treatment':1, '2hr recovery after 3hr ethanol treatment':2, '3hr recovery after 3hr ethanol treatment':3, '4hr recovery after 3hr ethanol treatment':4, '8hr recovery after 3hr ethanol treatment':5, '12hr recovery after 3hr ethanol treatment':6}
        self.initData(atype, atypes, ahash)  
        
    def getsun3(self, tn=1):
        self.prepareData("TNB51","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['Treatment', 'Late recovery']
        ahash = {'4.3% ethanol treatment for 3hrs':0, '3hr recovery after 3hr ethanol treatment':1, '4hr recovery after 3hr ethanol treatment':1, '8hr recovery after 3hr ethanol treatment':1, '12hr recovery after 3hr ethanol treatment':1}
        self.initData(atype, atypes, ahash)        
        
    def gettang2022(self, tn=1):
        self.prepareData("TNB52","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['4.5% EtOH, 5Hr', '4.5% EtOH, 5Hr, fresh medium 3Hr', '4.5% EtOH, 5Hr, fresh medium 6Hr', '4.5% EtOH, 5Hr, fresh medium 24Hr', '4.5% EtOH, 5Hr, fresh medium 48Hr']
        ahash = {'4.5% EtOH, 5Hr':0, '4.5% EtOH, 5Hr, fresh medium 3Hr':1, '4.5% EtOH, 5Hr, fresh medium 6Hr':2, '4.5% EtOH, 5Hr, fresh medium 24Hr':3, '4.5% EtOH, 5Hr, fresh medium 48Hr':4}
        self.initData(atype, atypes, ahash) 
        
    def getSoltysova(self, tn=1):
        self.prepareData("TNB53","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ctc marker positivity (ch1)')
        atypes = ['normal tissue', 'CTC negative', 'BrCA with CTC_EMT']
        ahash = {'normal tissue':0, 'CTC negative':1, 'CTC EMT':2, 'CTC EMT, epithelial':2, 'CTC epithelial':2}
        self.initData(atype, atypes, ahash)
        
    def getDonato(self, tn=1):
        self.prepareData("TNB58","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c model (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'Br16':0, 'LM2':0, 'Br61':0}
        #ahash = {'Br61':0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c hypoxia (ch1)')
        atypes = ['Negative', 'Positive']
        ahash = {'Negative':0, 'Positive':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)   
        
    def getDonato2(self, tn=1):
        self.prepareData("TNB58","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'CD 190' : 0, 'CD 208' : 0, 'CD 222' : 0, 'CD 224' : 0, 'CD 229' : 0, 'CD 264' : 0, 'CD 571' : 0, 'CD 574' : 0, 'CD 580' : 0, 'CD 583' : 0, 'CD 585' : 0, 'CD 590' : 0, 'CD 591' : 0, 'CD 592' : 0, 'CD 597' : 0, 'CD 599' : 0, 'CD 601' : 0, 'CD 605' : 0, 'CD 609' : 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c hypoxia (ch1)')
        atypes = ['Negative', 'Positive']
        ahash = {'Negative':0, 'Positive':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)          
        
        
        
    def getSoltysova2(self, tn=1):
        self.prepareData("TNB53","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ctc marker positivity (ch1)')
        atypes = ['CTC negative', 'CTC EMT', 'CTC EMT, epithelial', 'CTC epithelial']
        ahash = {'CTC negative':0, 'CTC EMT':1, 'CTC EMT, epithelial':2, 'CTC epithelial':3}
        self.initData(atype, atypes, ahash)  
        
    def getSoltysova3(self, tn=1):
        self.prepareData("TNB53","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ctc marker positivity (ch1)')
        atypes = ['CTC negative', 'CTC positive']
        ahash = {'CTC negative':0, 'CTC EMT':1, 'CTC EMT, epithelial':1, 'CTC epithelial':1}
        self.initData(atype, atypes, ahash)          
        
    def getKnudsen2019(self, tn=1):
        self.prepareData("TNB55","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c responder status (us)at last visit (ch1)')
        atypes = ['PR', 'PD', 'NE', 'SD']
        ahash = {'PR':0, 'PD':1, 'NE':2, 'SD':3}
        self.initData(atype, atypes, ahash)    
         
        
    def getsuppli11(self, tn=1):
        self.prepareData("LIV14","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c disease')
        atypes = ['NASH', 'NAFLD']
        ahash = {'NASH':0, 'NAFLD':1}
        self.initData(atype, atypes, ahash)    

    def getCagnin2012(self, tn=1):
        self.prepareData("SS49","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c patient number_ch1')
        atypes = ['H', 'P']
        ahash = {'':0, '1':1, '2':1, '3':1, '4':1, '5':1, '6':1, '7':1, '8':1}
        self.initData(atype, atypes, ahash) 
        
    def getEijg(self, tn=1):
        self.prepareData("SS48","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c characteristics (ch2)')
        atypes = ['atherosclerotic plaque-derived macrophages', 'Lung macrophages','Splenic macrophages', 'Liver macrophages']
        ahash = {'atherosclerotic plaque-derived macrophages':0, 'Lung macrophages':1,'Splenic macrophages':2, 'Liver macrophages':3}
        self.initData(atype, atypes, ahash) 
        
    def gethela(self, tn=1):
        self.prepareData("PGSS4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['HP', 'HK']
        ahash = {'HP1':0, 'HP1':0, 'HP2':0, 'HP2':0, 'HP3':0, 'HP3':0, 'HK1':1, 'HK1':1, 'HK2':1, 'HK3':1, 'HK3':1}
        self.initData(atype, atypes, ahash) 

    def getprodiff_T(self, tn=1):
        self.prepareData("PGSS4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['Sen_T', 'Res_T']
        ahash = {'P1T':0, 'P2T':0, 'P2T':0, 'P3T':0, 'P3T':0, 'P4T':0, 'P4T':0, 'P5T':1, 'P6T':1, 'P6T':1, 'P7T':1, 'P8T':1}
        self.initData(atype, atypes, ahash)        

    def getprodiff_C(self, tn=1):
        self.prepareData("PGSS4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['Sen_C', 'Res_C']
        ahash = {'P1C':0, 'P1C':0, 'P2C':0, 'P2C':0, 'P3C':0, 'P3C':0, 'P4C':0, 'P4C':0, 'P5C':1, 'P6C':1, 'P7C':1, 'P7C':1, 'P8C':1, 'P8C':1}
        self.initData(atype, atypes, ahash)  
        
    def getprodiff_S(self, tn=1):
        self.prepareData("PGSS4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['Un', 'Tr']
        ahash = {'P1C':0, 'P1C':0, 'P2C':0, 'P2C':0, 'P3C':0, 'P3C':0, 'P4C':0, 'P4C':0,'P1T':1, 'P2T':1, 'P2T':1, 'P3T':1, 'P3T':1, 'P4T':1, 'P4T':1}
        self.initData(atype, atypes, ahash)         
        
    def getprodiff_all(self, tn=1):
        self.prepareData("PGSS4","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['Sen_C', 'Sen_T', 'Res_C', 'Res_T', 'H_C', 'H_T']
        ahash = {'P1C':0, 'P1C':0, 'P2C':0, 'P2C':0, 'P3C':0, 'P3C':0, 'P4C':0, 'P4C':0,'P1T':1, 'P2T':1, 'P2T':1, 'P3T':1, 'P3T':1, 'P4T':1, 'P4T':1,'P5C':2, 'P6C':2, 'P7C':2, 'P7C':2, 'P8C':2, 'P8C':2, 'P5T':3, 'P6T':3, 'P6T':3, 'P7T':3, 'P8T':3, 'H1C': 4, 'H1C': 4, 'H2C': 4, 'H2C': 4, 'H3C': 4, 'H3C': 4, 'H1T': 5, 'H1T': 5, 'H2T': 5, 'H2T': 5, 'H3T': 5}
        self.initData(atype, atypes, ahash)           
        
    def gethelanew(self, tn=1):
        self.prepareData("PGSS5","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['HP', 'HK']
        ahash = {'HP1':0,'HP2':0,'HP3':0, 'HK1':1, 'HK3':1}
        self.initData(atype, atypes, ahash)         
        
    def getBasnet(self, tn=1):
        self.prepareData("TNB59","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c metastatic derivative (ch1)')
        atypes = ['Fat pad', 'Lung', 'Brain']
        ahash = {'Fat pad':0, 'Lung':1, 'Brain':2}
        self.initData(atype, atypes, ahash)    
        
    def getLUNG2014(self, tn=1):
        self.prepareData("TNB61","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c tissue (ch1)')
        atypes = ['Breast Tissue', 'blood of breast cancer patient']
        ahash = {'Breast Tissue':0, 'blood of breast cancer patient':1}
        self.initData(atype, atypes, ahash)   
        
    def getFINA2017(self, tn=1):
        self.prepareData("TNB62","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c sample type (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'blood with AdnaWash' : 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c tissue (ch1)')
        atypes = ['blood', 'circulating tumor cells']
        ahash = {'blood':0, 'circulating tumor cells':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getFINA2017_1(self, tn=1):
        self.prepareData("TNB62","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c tissue (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'blood': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c sample type (ch1)')
        atypes = ['spiked cells', 'RNA']
        ahash = {'spiked cells':0, 'RNA':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)         
        
        
    def getFINA2017_2(self, tn=1):
        self.prepareData("TNB62","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('type')
        atypes = ['Control', '50', '100', 'CTC']
        ahash = {'Control':0, '50':1, '100':2, 'CTC007':3,'CTC006':3,'CTC005':3,'CTC004':3,'CTC003':3,'CTC002':3,'CTC001':3}
        self.initData(atype, atypes, ahash)  
        
    def getFuji2022(self, tn=1):
        self.prepareData("SS50","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c fibrosis stage (ch1)')
        atypes = ['0', '1', '2', '3', '4']
        ahash = {'0':0, '1':1, '2':2, '3':3, '4':4}
        self.initData(atype, atypes, ahash) 
        
    def getKaur2021(self, tn=1):
        self.prepareData("TNB63","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['MDA-231_attached', 'MDA-231_suspension', 'MDA-231_suspension_B6H12 Antibody', 'MDA-231_suspension_Control Antibody']
        ahash = {'MDA-231_attached':0, 'MDA-231_suspension':1, 'MDA-231_suspension_B6H12 Antibody':2, 'MDA-231_suspension_Control Antibody':3}
        self.initData(atype, atypes, ahash)    
        
    def getBliss2019(self, tn=1):
        self.prepareData("TNB64","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_low-Oct4', 'MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_medium-Oct4', 'MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_high-Oct4']
        ahash = {'MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_low-Oct4':0, 'MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_medium-Oct4':1, 'MDA-MB-231_transfected with pEGFP1-Oct3/4 and then selected with neomycin_high-Oct4':2}
        self.initData(atype, atypes, ahash)    
        
    def getSarioglu2019(self, tn=1):
        self.prepareData("TNB65","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['White blood cell', 'CTC cluster']
        ahash = {'White blood cell':0, 'CTC cluster':1}
        self.initData(atype, atypes, ahash)   
        
    def getBentires(self, tn=1):
        self.prepareData("TNB66","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        atypes = ['MDAMB231_CTC', 'MDAMB231_Liver', 'MDAMB231_Lungs', 'MDAMB231_Spleen', 'MDAMB231_Tumor']
        ahash = {'MDAMB231_CTC_1': 0, 'MDAMB231_CTC_2': 0, 'MDAMB231_CTC_3': 0, 'MDAMB231_Liver_1': 1, 'MDAMB231_Liver_2': 1, 'MDAMB231_Liver_3': 1, 'MDAMB231_Lungs_1': 2, 'MDAMB231_Lungs_2': 2, 'MDAMB231_Lungs_3': 2, 'MDAMB231_Spleen_1': 3, 'MDAMB231_Spleen_2': 3, 'MDAMB231_Spleen_3': 3, 'MDAMB231_Tumor_1': 4, 'MDAMB231_Tumor_2': 4, 'MDAMB231_Tumor_3': 4}
        self.initData(atype, atypes, ahash)
        
    def getSugiyama(self, tn=1):
        self.prepareData("TNB67","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c system (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'TENA': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c population (ch1)')
        atypes = ['GFP+', 'mCherry+', 'Both++']
        ahash = {'P15':0, 'P10':0, 'P16':0, 'P9':0, 'P5':1, 'P17':1, 'P18':1, 'P22':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getSugiyama2(self, tn=1):
        self.prepareData("TNB67","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c passedqc (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'TRUE': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c population (ch1)')
        atypes = ['++', 'EMT', 'BC_Ep']
        ahash = {'P22':0, 'P15':1, 'P18':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
        
    def getSugiyama3(self, tn=1):
        self.prepareData("TNB67","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c system (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'TENA': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c new')
        atypes = ['P22','P15', 'P10', 'P16', 'P09', 'P05', 'P17', 'P18']
        ahash = {'P22_TRUE':0,'P15_TRUE':1, 'P10_TRUE':2, 'P16_TRUE':3, 'P09_TRUE':4, 'P05_TRUE':5, 'P17_TRUE':6, 'P18_TRUE':7}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)    
        
        
    def getSugiyama4(self, tn=1):
        self.prepareData("TNB67","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c system (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'Cdh': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c new')
        atypes = ['P15','P05','P18']
        ahash = {'P15_TRUE':0,'P05_TRUE':1,'P18_TRUE':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)   
        
    def getLee2022(self, tn=1):
        self.prepareData("TNB68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['Clone EM1', 'Clone EM2', 'Clone EM3']
        ahash = {'Clone EM1':0, 'Clone EM2':1, 'Clone EM3':2}
        self.initData(atype, atypes, ahash)    
        
    def getLee2022_2(self, tn=1):
        self.prepareData("TNB68","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['Clone E','Parental','Clone EM1', 'Clone EM2', 'Clone EM3','Clone M1','Clone M2']
        ahash = {'Clone E':0,'Parental':1,'Clone EM1':2, 'Clone EM2':3, 'Clone EM3':4,'Clone M1':5,'Clone M2':6}
        self.initData(atype, atypes, ahash)  
        
    def getShinde2019(self, tn=1):
        self.prepareData("TNB69","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c group (ch1)')
        atypes = ['Parental cell line','TGFB','LAPR', 'BM']
        ahash = {'Parental cell line':0,'TGFB':1,'LAPR':2, 'BM':3}
        self.initData(atype, atypes, ahash)    
        
    def getXu20122(self, tn=1):
        self.prepareData("TNB70","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB231 breast cancer': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c stress (ch1)')
        atypes = ['normoxia','hypoxia']
        ahash = {'normoxia':0,'hypoxia':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getZanotto2019(self, tn=1):
        self.prepareData("TNB71","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        atypes = ['MDA-MB231_Control', 'MDA-MB231_Paclitaxel', 'MDA-MB231_MMS_8hr', 'MDA-MB231_Etoposide', 'MDA-MB231_Doxorubicin', 'MDA-MB231_Ctrl_8hr']
        ahash = {'MDA-MB231_Control':0, 'MDA-MB231_Paclitaxel':1, 'MDA-MB231_MMS_8hr':2, 'MDA-MB231_Etoposide':3, 'MDA-MB231_Doxorubicin':4, 'MDA-MB231_Ctrl_8hr':5}
        self.initData(atype, atypes, ahash)         

    def getJordan2020(self, tn=1):
        self.prepareData("TNB72","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ptprc (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'low': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c erbb2 (ch1)')
        atypes = ['low','high']
        ahash = {'low':0,'high':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
        
    def getJordan2(self, tn=1):
        self.prepareData("TNB72","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c ptprc (ch1)')
        atypes = ['low','high']
        ahash = {'low':0,'high':1}
        self.initData(atype, atypes, ahash)  
        
    def getJohnson2022(self, tn=1):
        self.prepareData("TNB73","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['Untreated MCF10A mRNA','2d +TGFB MCF10A mRNA','4d +TGFB MCF10A mRNA', '6d +TGFB MCF10A mRNA','8d +TGFB MCF10A mRNA','10d +TGFB MCF10A mRNA', '10d +TGFB, 4d -TGFB MCF10A mRNA', '10d +TGFB, 6d -TGFB MCF10A mRNA', '10d +TGFB, 10d -TGFB MCF10A mRNA', '4d +TGFB, 2d -TGFB MCF10A mRNA', '4d +TGFB, 4d -TGFB MCF10A mRNA', '4d +TGFB, 6d -TGFB MCF10A mRNA', '4d +TGFB, 10d -TGFB MCF10A mRNA']
        ahash = {'Untreated MCF10A mRNA':0,'2d +TGFB MCF10A mRNA':1,'4d +TGFB MCF10A mRNA':2, '6d +TGFB MCF10A mRNA':3,'8d +TGFB MCF10A mRNA':4,'10d +TGFB MCF10A mRNA':5, '10d +TGFB, 4d -TGFB MCF10A mRNA':6, '10d +TGFB, 6d -TGFB MCF10A mRNA':7, '10d +TGFB, 10d -TGFB MCF10A mRNA':8, '4d +TGFB, 2d -TGFB MCF10A mRNA':9, '4d +TGFB, 4d -TGFB MCF10A mRNA':10, '4d +TGFB, 6d -TGFB MCF10A mRNA':11, '4d +TGFB, 10d -TGFB MCF10A mRNA':12}
        self.initData(atype, atypes, ahash)         
        
    def getEnzo2019(self, tn=1):
        self.prepareData("TNB75","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB-231': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['water (vehicle)', '2DG']
        ahash = {'water (vehicle)':0, '2DG':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getKang2022(self, tn=1):
        self.prepareData("TEST8","/Users/dtv004/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'HCT116': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['IgG', 'IgG+H2O2', 'anti-Hsp60N+H2O2']
        ahash = {'IgG':0, 'IgG+H2O2':1, 'anti-Hsp60N+H2O2':2}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getKang2(self, tn=1):
        self.prepareData("TEST8","/Users/dtv004/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line_ch1')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB-231': 0}
        hval = [1 if i in ahash else None for i in atype]          
        atype = self.h.getSurvName('c treatment_ch1')
        atypes = ['IgG', 'IgG+H2O2']
        ahash = {'IgG':0, 'IgG+H2O2':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getKimBG(self, tn=1):
        self.prepareData("TNB78","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB-231': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c stress (ch1)')
        atypes = ['High', 'low']
        ahash = {'0 kPa':1, '0.386 kPa':1, '0.773 kPa':1, '1.546 kPa':0, '3.866 kPa':0, '7.732 kPa':0}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getLiuX(self, tn=1):
        self.prepareData("TNB79","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB-231': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c agent (ch1)')
        atypes = ['Vehicle DMSO', 'ER-stress']
        ahash = {'Vehicle DMSO':0, 'ERX-41 2h treatment':1, 'ERX-41 4h treatment':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getCaffa(self, tn=1):
        self.prepareData("TNB80","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('Treatment')
        atypes = [' control', ' starvation', ' Tamoxifen', ' starvation and Tamoxifen']
        ahash = {' control':0, ' starvation':1, ' Tamoxifen':2, ' starvation and Tamoxifen':3}
        self.initData(atype, atypes, ahash)        

    def getFranses2020(self, tn=1):
        self.prepareData("TNB81","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c patient diagnosis (ch1)')
        atypes = ['healthy donor', 'localized PDAC', 'metastatic PDAC']
        ahash = {'healthy donor':0, 'localized PDAC':1, 'metastatic PDAC':2}
        self.initData(atype, atypes, ahash)   
        
    def getTing2019(self, tn=1):
        self.prepareData("TNB82","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c subject status (ch1)')
        atypes = ['healthy donor (HD)', 'intraductal papillary mucinous neoplasm (IPMN), low-risk', 'intraductal papillary mucinous neoplasm (IPMN), high-risk', 'pancreatic ductal adenocarcinoma (PDAC)']
        ahash = {'healthy donor (HD)':0, 'intraductal papillary mucinous neoplasm (IPMN), low-risk':1, 'intraductal papillary mucinous neoplasm (IPMN), high-risk':2, 'pancreatic ductal adenocarcinoma (PDAC)':3}
        self.initData(atype, atypes, ahash)   
        
        
    def getBoya2022(self, tn=1):
        self.prepareData("TNB83","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type (ch1)')
        atypes = ['circulating tumor cell cluster', 'cell line', 'white blood cell']
        ahash = {'circulating tumor cell cluster':0, 'cell line':1, 'white blood cell':2}
        self.initData(atype, atypes, ahash)     
        
    def getXiang2019(self, tn=1):
        self.prepareData("TNB84","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c pre/post therapy (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'Pre': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c dtc positive(ihc) (ch1)')
        atypes = ['Yes', 'No']
        ahash = {'Yes':0, 'No':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getMsaki2018(self, tn=1):
        self.prepareData("TNB85","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type (ch1)')
        atypes = ['Early disseminated tumor cells', 'Tumor derived from TUBO cells']
        ahash = {'Early disseminated tumor cells':0, 'Tumor derived from TUBO cells':1}
        self.initData(atype, atypes, ahash)
        
    def getGreer2022(self, tn=1):
        self.prepareData("TNB86","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell type (ch1)')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'Triple Negative Breast Cancer, Basal A': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c title')
        atypes = ['AU565', 'BT20', 'BT474', 'HCC1500', 'HCC1937', 'HCC1954', 'HCC38', 'Hs578T', 'MCF7', 'MDA-MB-231', 'MDA-MB-436', 'MDA-MB-453', 'MDA-MB-468', 'T47D', 'ZR-75-1']
        ahash = {'AU565':0, 'BT20':1, 'BT474':2, 'HCC1500':3, 'HCC1937':4, 'HCC1954':5, 'HCC38':6, 'Hs578T':7, 'MCF7':8, 'MDA-MB-231':9, 'MDA-MB-436':10, 'MDA-MB-453':11, 'MDA-MB-468':12, 'T47D':13, 'ZR-75-1':14}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash)  
        
    def getGreer2(self, tn=1):
        self.prepareData("TNB86","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        atypes = ['AU565', 'BT20', 'BT474', 'HCC1500', 'HCC1937', 'HCC1954', 'HCC38', 'Hs578T', 'MCF7', 'MDA-MB-231', 'MDA-MB-436', 'MDA-MB-453', 'MDA-MB-468', 'T47D', 'ZR-75-1']
        ahash = {'AU565':0, 'BT20':1, 'BT474':2, 'HCC1500':3, 'HCC1937':4, 'HCC1954':5, 'HCC38':6, 'Hs578T':7, 'MCF7':8, 'MDA-MB-231':9, 'MDA-MB-436':10, 'MDA-MB-453':11, 'MDA-MB-468':12, 'T47D':13, 'ZR-75-1':14}
        self.initData(atype, atypes, ahash)      
        
        
    def getVelez2017(self, tn=1):
        self.prepareData("EMT2","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c cell line')
        #ahash = {'IgG CTC chip': 0}
        ahash = {'MDA-MB-231': 0}
        hval = [1 if i in ahash else None for i in atype]         
        atype = self.h.getSurvName('c culture condition')
        atypes = ['2.5 mg/mL collagen matrix', '6 mg/mL collagen matrix']
        ahash = {'2.5 mg/mL collagen matrix':0, '6 mg/mL collagen matrix':1}
        atype = [atype[i] if hval[i] == 1
                else None for i in range(len(atype))]   
        self.initData(atype, atypes, ahash) 
        
    def getHCT116_Sw480(self, tn=1):
        self.prepareData("PG60","/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Cell protocol')
        atypes = ['SW480', 'SW480 PF', 'HCT116', 'HCT116 PF']
        ahash = {'SW480':0, 'SW480 PF':1, 'HCT116':2, 'HCT116 PF':3}
        self.initData(atype, atypes, ahash)  
        
    def getxeno_PF(self, tn=1):
        self.prepareData("PG20.m","/Users/sataheri/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c Sample Code')
        atypes = ['XC', 'XPF']
        ahash = {'XC1':0, 'XC20':0, 'XC3':0, 'XPF1':1, 'XPF2':1, 'XPF3':1}
        self.initData(atype, atypes, ahash)        
                
    def getMarisa2013(self, tn=1):
        self.prepareData("CRC54","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c kras.mutation')
        atypes = ['WT', 'M']
        ahash = {'WT':0, 'M':1}
        self.initData(atype, atypes, ahash)   
        
    def getbozon2022(self, tn=1):
        self.prepareData("SS53","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c treatment (ch1)')
        atypes = ['Transfected with shRNA targetting Cdx2 (mCherry postive), sorted by FACS', 'Non-transfected scrambled shRNA (mCherry negative), sorted by FACS','Transfected with scrambled shRNA (mCherry positive), sorted by FACS','Untransfected, sorted by FACs']
        ahash = {'Transfected with shRNA targetting Cdx2 (mCherry postive), sorted by FACS':0, 'Non-transfected scrambled shRNA (mCherry negative), sorted by FACS':1,'Transfected with scrambled shRNA (mCherry positive), sorted by FACS':2,'Untransfected, sorted by FACs':3}
        self.initData(atype, atypes, ahash)  

    def getBalbinot(self, tn=1):
        self.prepareData("BL06","/Users/ssaha/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['Cecum cdx2 +/- and cecum WT', 'cecum conditional cdx2-/-', 'cecum conditional cdx2-/- ; Apc+/-']
        ahash = {'Cecum cdx2 +/-':0, 'cecum WT':0, 'cecum conditional cdx2-/-':1, 'cecum conditional cdx2-/- ; Apc+/-':2}
        self.initData(atype, atypes, ahash)    
        
        
    def getsakamoto(self, tn=1):
        self.prepareData("BL08","/Users/ssaha/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c title')
        atypes = ['control_sample', 'Cdx2-/-_sample', 'Braf_V600E_sample', 'Cdx2-/-+Braf_V600E_sample', 'Apc-/-_sample']
        ahash = {'control_sample1': 0, 'control_sample2': 0, 'control_sample3': 0, 'Cdx2-/-_sample1': 1, 'Cdx2-/-_sample2': 1, 'Cdx2-/-_sample3': 1, 'Braf_V600E_sample1': 2, 'Braf_V600E_sample2': 2, 'Braf_V600E_sample3': 2, 'Cdx2-/-+Braf_V600E_sample1': 3, 'Cdx2-/-+Braf_V600E_sample2': 3, 'Cdx2-/-+Braf_V600E_sample3': 3, 'Apc-/-_sample1': 4, 'Apc-/-_sample2': 4, 'Apc-/-_sample3': 4}
        self.initData(atype, atypes, ahash)    
        
        
    def getverzi(self, tn=1):
        self.prepareData("BL10","/Users/ssaha/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype (ch1)')
        atypes = ['Wild-Type', 'Wild-Type (Cdx2 f/f)', 'Cdx2-KO(Shh-cre; Cdx2f/f)', 'Wild-Type(Cdx2 f/f; vil-Cre-ERT2 X Cdx2 f/f; vil-Cre-ERT2)']
        ahash = {'Wild-Type':0, 'Wild-Type (Cdx2 f/f)':1, 'Cdx2-KO(Shh-cre; Cdx2f/f)':2, 'Wild-Type(Cdx2 f/f; vil-Cre-ERT2 X Cdx2 f/f; vil-Cre-ERT2)':3}
        self.initData(atype, atypes, ahash)
        
    def getverzi2(self, tn=1):
        self.prepareData("BL11","/Users/ssaha/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c genotype/variation')
        atypes = ['Cdx2 knockout', 'Cdx2-Hnf4a Double knockout', 'Hnf4a knockout', 'littermate controls']
        ahash = {'Cdx2 knockout':0, 'Cdx2-Hnf4a Double knockout':1, 'Hnf4a knockout':2, 'littermate controls':3}
        self.initData(atype, atypes, ahash)    
        
    def getchemJ2022(self, tn=1):
        self.prepareData("TNB87","/Users/sinha7290/public_html/Hegemon/explore.conf")
        atype = self.h.getSurvName('c source_name (ch1)')
        atypes = ['mammary gland','fat pad xenograft', 'circulating tumor cell', 'lung metastasis']
        ahash = {'mammary gland':0,'fat pad xenograft':1, 'circulating tumor cell':2, 'lung metastasis':3}
        self.initData(atype, atypes, ahash)        