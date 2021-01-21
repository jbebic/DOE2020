# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:59:20 2020

@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com

v1.0 JZB20201207
Illustration of heatmap plotting using reactive compensation additions from UTK.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as dpdf # pdf output
from matplotlib import cm, colors

from datetime import datetime
import os # operating system interface
import sys # errors handling and paths to local modules

import logging

def load_result_matrices(dirin, fnameroot):
    df1 = pd.read_csv(os.path.join(dirin,fnameroot+'mx1.csv'))
    # Mx1 is a 3-dimensional data structure set up as a row-stacked matrix where
    # the first column carries the iteration number. For a given iteration number the 
    # rows are in order of contingencies (line outages). The columns are bus numbers and
    # the values are pu bus voltages that are achievable with the existing shunt compensation
    # but they are only recorded if the algorithm decides that adding another capacitor 
    # is necessary. Otherwise the voltage value is left as NaN
    # Mx1 values are used to prove that more compensation is needed. 
    
    df2 = pd.read_csv(os.path.join(dirin,fnameroot+'mx2.csv'))
    # Mx2 has the same structure as Mx1, but the values are the number of capacitors that were activated
    
    df3 = pd.read_csv(os.path.join(dirin,fnameroot+'mx3.csv'))
    # Mx3 has the same structure as Mx1, but the values are the number of requested capacitors

    df4 = pd.read_csv(os.path.join(dirin,fnameroot+'mx4.csv'))
    # Mx5 has columns that correspond to bus numbers, rows are iteration numbers and values are
    # the modes of non-zero requests

    df5 = pd.read_csv(os.path.join(dirin,fnameroot+'mx5.csv'))
    # Mx5 has columns that correspond to bus numbers, rows are iteration numbers and values are
    # number of added capacitors
    
    return df1, df2, df3, df4, df5

def output_continuous_heatmap_page(pltPdf, 
                                   df, 
                                   xymax, 
                                   xylabels = ['Contingency number', 'Bus number'],
                                   pagetitle='', 
                                   axistitle='', 
                                   colormap='viridis', 
                                   crange=None, 
                                   precision=0.05, 
                                   figsize = (6.4, 4.8), 
                                   dpi=300):

    # define the figure
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi, sharex=False)
    
    fig.suptitle(pagetitle) # This titles the figure
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
    
    ax0.set_title(axistitle)

    # define color range
    # round pu color scale to precision
    if crange == None:
        cmin = np.floor(df.min().min()/precision)*precision
        cmax = np.ceil( df.max().max()/precision)*precision
    else:
        cmin = crange[0]
        cmax = crange[1]
    
    # define image
    im0 = ax0.imshow(df.iloc[:,:].transpose(),
                     interpolation='none', #'nearest'
                     cmap=colormap, 
                     origin='lower', 
                     # extent=(0, xymax[1], 0, xymax[0]),
                     vmin = cmin, 
                     vmax = cmax)

    fig.colorbar(im0, ax=[ax0])
    ax0.set_xlabel(xylabels[0])
    ax0.set_ylabel(xylabels[1])
    ax0.set_aspect('equal')
    
    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.close() # Closes fig to release memory

    return

def output_discrete_heatmap_page(pltPdf, 
                                 df, 
                                 xymax, 
                                 xylabels = ['Contingency number', 'Bus number'],
                                 pagetitle='', 
                                 axistitle='', 
                                 colormap='tab10', 
                                 crange=4, 
                                 figsize = (6.4, 4.8), dpi=300):

    # define the figure
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi, sharex=False)
    
    fig.suptitle(pagetitle) # This titles the figure
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
    
    ax0.set_title(axistitle)

    # define color range and boundaries
    cmap = cm.get_cmap(colormap, 10)
    clist = ['#ffffff'] # adding white for level zero
    for i in range(crange):
        rgba = cmap(i)
        # print('color(%d):' %i, rgba*255)
        clist.append(colors.rgb2hex(rgba))
    
    cmap = colors.ListedColormap(clist)
    tarray = np.arange(crange+2)
    bounds = (tarray-0.5).tolist()
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # define image
    im0 = ax0.imshow(df.iloc[:,:].transpose(),
                     interpolation='none', #'nearest'
                     cmap=cmap, 
                     norm = norm,
                     origin='lower', 
                     extent=(0, xymax[0], 0, xymax[1]),
                     )
    
    # formating the colorbar
    cbar = fig.colorbar(im0, ax=[ax0], ticks=tarray[0:-1].tolist())
    temp = [str(e) for e in tarray[0:-1].tolist()]
    temp[-1] = '>=' + temp[-1]
    cbar.ax.set_yticklabels(temp)

    ax0.set_xlabel(xylabels[0])
    ax0.set_ylabel(xylabels[1])
    ax0.grid(True, which='both', axis='both')
    ax0.set_aspect('auto')

    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.close() # Closes fig to release memory

    return


def open_plot_file(dirplots, fpltroot, fpltsuf):
    temp = os.path.join(dirplots,fpltroot + fpltsuf)
    logging.info('Opening plot file: %s' %(temp))
    try:
        pltPdf = dpdf.PdfPages(temp) # opens a pdf file
    except:
        logging.error('Unexpected error occured:', sys.exc_info()[0])
        raise
    return pltPdf

if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/severity.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    if True:
        
        dirin = 'results/' # location of result matrices
        finroot = '0v93_full_' # the actual result file names have mx1 through mx5 added to this root of a filename
        
        dirplots = 'plots/' # must create this relative directory path before running
        fpltroot = 'heatmap' # file name to save the plot
        pltPdf1 = open_plot_file(dirplots, fpltroot, '_mx1.pdf')
        pltPdf2 = open_plot_file(dirplots, fpltroot, '_mx2.pdf')
        pltPdf3 = open_plot_file(dirplots, fpltroot, '_mx3.pdf')
        pltPdf4 = open_plot_file(dirplots, fpltroot, '_mx4.pdf')
        pltPdf5 = open_plot_file(dirplots, fpltroot, '_mx5.pdf')

        logging.info('Reading results files in: %s, named %s' %(dirin, finroot))
        try:
            df1, df2, df3, df4, df5 = load_result_matrices(dirin, finroot) # description of content given in the function
        except:
            logging.error('Unexpected error occured:', sys.exc_info()[0])
            raise

        # Plotting
        # Generating plots by iteration number
        for inum in df1.iloc[:,0].unique():
            try:
                df1a = df1[df1.iloc[:,0] == inum]
                xymax = df1a.shape # total number of contingencies and buses as a tuple
                ix1 = df1a[df1a.iloc[:,1:].isna().sum(axis=1)==0].index # index of rows without any NaN
                ix2 = df1a[df1a.iloc[:,1:].isna().sum(axis=1)>0].index # index of rows with some NaN

                df2a = df2[df2.iloc[:,0] == inum]
                # df2a.loc[ix1] = np.nan
                df3a = df3[df3.iloc[:,0] == inum]
                title = 'Iteration %d: %d contingencies requesting additional capacitors' %(inum, ix1.size)
                logging.info(title)
                output_continuous_heatmap_page(pltPdf1, df1a.iloc[:,1:], xymax, pagetitle=title, axistitle='Voltage (before new capacitors are added) [pu]')
                output_discrete_heatmap_page(pltPdf2, df2a.iloc[:,1:], xymax, pagetitle=title, axistitle='Number of activated capacitors', crange=4, colormap='tab10')
                output_discrete_heatmap_page(pltPdf3, df3a.iloc[:,1:], xymax, pagetitle=title, axistitle='Number of newly requested capacitors', crange=4, colormap='tab10')
                
            except:
                logging.info('** something failed in iteration %d' %(inum))

        # Mx4 and 5 are 2-D with iteration numbers in the rows and bus number in the columns
        xymax = df4.shape
        output_discrete_heatmap_page(pltPdf4, df4, xymax, crange=10, colormap='tab10', xylabels = ['Iteration number', 'Bus number'], pagetitle='Modes of new capacitor requests')
        
        # Mx5 only has iteration on the x-axis
        xymax = df5.shape
        output_discrete_heatmap_page(pltPdf5, df5, xymax, crange=4, colormap='tab10', xylabels = ['Iteration number', 'Bus number'], pagetitle='Number of added capacitors in each iteration')
        
        # Preparing for exit
        pltPdf1.close() # closes a pdf file
        pltPdf2.close() # closes a pdf file
        pltPdf3.close() # closes a pdf file
        pltPdf4.close() # closes a pdf file
        pltPdf5.close() # closes a pdf file

        
    # preparing for exit
    logging.shutdown()
