# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:14:24 2020

@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com

v1.0 JZB20201203
Illustration of vectorized calculations of severity metrics using numpy and plotting of 
the severity transfer functions as an illustration of matplotlib plotting
Heavily commented to aid in the comprehension.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as dpdf # pdf output

from datetime import datetime
import os # operating system interface
import sys # errors handling and paths to local modules

import logging

#%% calculate severity at specified dv breakpoints
def calculate_severity_breakpoints(dv, ks):
    sv = np.zeros_like(dv) # fill the array with zeros, then populate indices from 1 to end of array in the for loop
    for i in range(1, len(dv)):
        sv[i] = (dv[i]-dv[i-1])*ks[i-1] + sv[i-1]
        
    return sv
    
#%% 
def calculate_voltage_severity(v, dv, ks, vnom=1.0):
    # v is the vector of per-unit bus voltages
    # dv and ks define a piece-wise linear lookup table measuring severity based on voltage values
    # dv is the array of absolute voltage errors relative to 1.0 (specified for positive values only)
    # ks is the array of the corresponding severity gradients.
    # Both dv and ks must be non-decreasing, meaning: dv[i] <= dv[i+1], for any i. Ditto for ks(i)
    # vnom is the nominal voltage value, defined as an optional function parameter with the default value of 1.0
    
    # Calculate severity values at dv breakpoints and place into an array called sv.
    # This is needed to make the subsequent vector calcuations non-recursive.
    # placed it into a separate function to enable calling it from plot_severity_tf function
    sv = calculate_severity_breakpoints(dv, ks)
    
    # In severity calculations, all line segmets of the piecewise linear transfer function are treated as 
    # separate functions and severity s(v) is solved for each one as:
    # s(v) = (|v-vnom| - dv)*ks + sv; where: dv, ks, and sv are row vectors defining the transfer function
    
    # Calculate absolute values of voltage errors
    verr = np.abs(v-vnom)
    # change verr into a column vector
    verr = verr.reshape(-1, 1)
    # replicate this column vector as many times as there are segments on the severity curve
    verr = np.tile(verr, (1,np.size(dv)))
    
    # At this point verr is a matrix with identical columns, and dv, ks, and sv are 
    # row vectors defining the piecewise linear line segments of the severity transfer function.
    # We calculate the severity metric for each column (the operations are elementwise)
    severity = (verr-dv)*ks + sv
    # at this point severity is a matrix where each row is a value of severity looked up from each
    # line segment. Row 1 corresponds to voltage on bus, and severity[1,1] is the severity value read 
    # from the first curve 
    
    severity = np.max(severity, axis=1) # row-wise max selects the severity line segment with the highest value and returns a row vector
    zero_row = np.zeros_like(severity) # need a zero row to limit severity to zero when |v-vnom| < dv[0]
    severity = np.stack([severity, zero_row]) # stacks two rows into a matrix
    severity = np.max(severity, axis = 0) # column-wise maximum of severity and the zero row
    return severity # this returns severity by bus.

#%% plot severity transfer function
def plot_severity_tf(pltPdf, dv, ks, title='Voltage Severity TF'):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.8), dpi=300) # , sharex=True
    plt.suptitle(title)
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
   
    sv = calculate_severity_breakpoints(dv, ks)
    v = np.linspace(0.8, 1.2, 100) # creating a regularly spaced numpy array with 100 data points
    s = calculate_voltage_severity(v, dv, ks)
    
    ax0.plot(v, s, label='Vseverity')
    ax0.plot(1+dv, sv, 'o', color='C1') # labeling with circles
    ax0.plot(1-dv, sv, 'd', color='C1') # labeling with diamonds
    ax0.grid(True, which='both', axis='both')
    ax0.set_ylim([0,0.3]) # when range not specified, the graph autoscales 
    ax0.set_ylabel('Severity [pu]')
    ax0.legend()
    #ax0.set_xlabel('Voltage [pu]')
    ax0.set_xlabel('Voltage [pu]')
    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.close() # Closes fig to clean up memory
    return

if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/severity.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    if True:
        # define the transfer function of severity (positive values only, non-decreasing)
        dv_values = np.array([0.03, 0.08])
        ks_values = np.array([1., 2.])
        # defining arbitrary voltages to evaluate 
        v_bus = np.array([0.91, 1.01, 0.94, 1.05, 1.1, 0.99]) 
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'voltage_severity.pdf' # file name to save the plot
        logging.info('Opening plot file: %s' %(os.path.join(dirplots,fnameplot)))
        try:
            pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
            plot_severity_tf(pltPdf, dv_values, ks_values) # places a plot page into the pdf file
            plot_severity_tf(pltPdf, dv_values, ks_values/2) # places a plot page into the pdf file
            pltPdf.close() # closes a pdf file
        except:
            logging.error('Unexpected error occured:', sys.exc_info()[0])
            raise
        
    # preparing for exit
    logging.shutdown()
