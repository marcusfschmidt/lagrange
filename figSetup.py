#%%

import matplotlib.pyplot as plt
import numpy as np

def figSetup(nrow = 1, ncol = 1,legend = 14, small = 16, medium = 18, bigger = 24, sharex = False, sharey = False):

    w = 25/2.54
    if ncol == 1 and nrow == 1:
        h = w*0.7
    else:
        h = w*0.8 / ncol*nrow
    #Change height if nrow = 1

    correctionFactor = (ncol + np.log(1/ncol))**(-1)

    LEGEND_SIZE = legend*correctionFactor
    SMALL_SIZE = small*correctionFactor
    MEDIUM_SIZE = medium*correctionFactor
    BIGGER_SIZE = bigger*correctionFactor

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    fig, ax = plt.subplots(nrow, ncol, figsize = (w, h), sharex = sharex, sharey = sharey)
    # stylePath = os.environ["ONEDRIVE"] + '\Speciale\Data\plotstyle.mplstyle'
    # plt.style.use(stylePath)
    # 
    return fig, ax

