# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:21:12 2020

@author: txia4@vols.utk.edu

v1.3 TX20201222
Delete the non-converge cases

v1.2 TX20201220
Computhe the cumulative severity of each contigency. The result is saved in powerflow_severity_cumulative.pdf
Compute  the cumulative severity of all contigency. The result is saved in powerflow_severity_cumulative_sum.pdf

v1.1 TX20201216
There is no bug in saving files now
compute the powerflow severity



v1.0 TX20201215
read the powerflow file. Plot the result
There is a bug in saving files


"""


import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy.matlib 
import matplotlib.backends.backend_pdf as dpdf # pdf output

from datetime import datetime # time stamps
import os # operating system interface

# import andes
# from andes.core.var import BaseVar, Algeb, ExtAlgeb

from npcc_voltages_severity import read_cont_mx1, calculate_voltage_severity
from review_shunt_compensation import output_continuous_heatmap_page

#%% read_contingencies
# def read_cont_mx(dirin, fname):
    
#     logging.info('reading contigency matrix')
#     df = pd.read_csv(os.path.join(dirin, fname))
#     return df

#%% calculate severity at specified dv breakpoints
# def calculate_severity_breakpoints(dv, ks):
#     sv = np.zeros_like(dv) # fill the array with zeros, then populate indices from 1 to end of array in the for loop
#     for i in range(1, len(dv)):
#         sv[i] = (dv[i]-dv[i-1])*ks[i-1] + sv[i-1]
        
#     return sv
    
#%% 
# def calculate_powerflow_severity(v, dv, ks, vnom=0):
#     # v is the vector of per-unit bus voltages
#     # dv and ks define a piece-wise linear lookup table measuring severity based on voltage values
#     # dv is the array of absolute voltage errors relative to 1.0 (specified for positive values only)
#     # ks is the array of the corresponding severity gradients.
#     # Both dv and ks must be non-decreasing, meaning: dv[i] <= dv[i+1], for any i. Ditto for ks(i)
#     # vnom is the nominal voltage value, defined as an optional function parameter with the default value of 1.0
    
#     # Calculate severity values at dv breakpoints and place into an array called sv.
#     # This is needed to make the subsequent vector calcuations non-recursive.
#     # placed it into a separate function to enable calling it from plot_severity_tf function
#     sv = calculate_severity_breakpoints(dv, ks)
    
#     # In severity calculations, all line segmets of the piecewise linear transfer function are treated as 
#     # separate functions and severity s(v) is solved for each one as:
#     # s(v) = (|v-vnom| - dv)*ks + sv; where: dv, ks, and sv are row vectors defining the transfer function
    
#     # Calculate absolute values of voltage errors
#     verr = np.abs(v-vnom)
#     # change verr into a column vector
#     verr = verr.reshape(-1, 1)
#     # replicate this column vector as many times as there are segments on the severity curve
#     verr = np.tile(verr, (1,np.size(dv)))
    
#     # At this point verr is a matrix with identical columns, and dv, ks, and sv are 
#     # row vectors defining the piecewise linear line segments of the severity transfer function.
#     # We calculate the severity metric for each column (the operations are elementwise)
#     severity = (verr-dv)*ks + sv
#     # at this point severity is a matrix where each row is a value of severity looked up from each
#     # line segment. Row 1 corresponds to voltage on bus, and severity[1,1] is the severity value read 
#     # from the first curve 
    
#     severity = np.max(severity, axis=1) # row-wise max selects the severity line segment with the highest value and returns a row vector
#     zero_row = np.zeros_like(severity) # need a zero row to limit severity to zero when |v-vnom| < dv[0]
#     severity = np.stack([severity, zero_row]) # stacks two rows into a matrix
#     severity = np.max(severity, axis = 0) # column-wise maximum of severity and the zero row
#     return severity # this returns severity by bus.
#%% 
def plot_severity(pltPdf, plot_x, plot_y, title=''):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.8), dpi=300) # , sharex=True
    plt.suptitle(title)
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
    
    # ax0.bar(plot_x, plot_y, label='Pseverity')
    ax0.step(plot_x, plot_y, label='Pseverity')


    ax0.grid(True, which='both', axis='both')
    #ax0.set_ylim([0,0.3]) # when range not specified, the graph autoscales 
    ax0.set_ylabel('Apparent Power Severity')
    #ax0.set_xlabel('Voltage [pu]')
    ax0.set_xlabel('Contigency number')
    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.title(title)
    plt.close() # Closes fig to clean up memory
    return
#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)
       
    if True: # loading the contingency results and test voltage severtiy
        dirin = 'results/'
        fnamein = 'contigency_apparentpower.csv'
        
        # loding in Mx1
        dfMx = read_cont_mx1(dirin, fnamein)
        print()
        #print('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        #print('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        logging.info('The shape of Mx is (%d, %d)' %(dfMx.shape[0], dfMx.shape[1]))
        logging.info('There are %d unique iteration numbers' %(len(dfMx.iloc[:,0].unique())))
        
        npMx = dfMx.to_numpy()
        list_convergy = npMx[:,0]
        baseflow = npMx[0,:]
        nonconver_list = np.asarray(np.where(npMx[:,0]==0))
        
        npMx2 = np.delete(npMx,nonconver_list,axis=0)   #delete the nonconverge row
        npMx2 = np.delete(npMx2,[0],axis=0)   #delete the base case row
        npMx2 = np.delete(npMx2,[0],axis=1)   #delete the fisrt column
        
    if True:  # plot contigency
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'powerflow_result.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
       
    
        for plot_num in range(npMx2.shape[0]):
            plot_x = np.linspace(1,npMx2[plot_num,:].shape[0],npMx2[plot_num,:].shape[0])
            plot_y = npMx2[plot_num,:]
            plot_x = plot_x.reshape(plot_y.shape)
            title_temp = 'The power flow of contigency  %d' %(plot_num+1)
            plot_severity(pltPdf, plot_x, plot_y,title_temp) # places a plot page into the pdf file                              
            
        pltPdf.close() # closes a pdf file
        
    if True:  # contigency severity 
       
        dirin = 'results/'
        fnamein = 'powerflow_limit.csv'
        dfMx_limit = read_cont_mx1(dirin, fnamein)
        npMx_limit = dfMx_limit.to_numpy()
        npMx_limit = npMx_limit.T
    
        dv_values = np.array([0.80, 1.0])
        ks_values = np.array([1., 2.])
        
        npMx2_norm = npMx2/npMx_limit
        # severity_matrix = calculate_powerflow_severity(npMx2_norm, dv_values, ks_values)
        severity_matrix = calculate_voltage_severity(npMx2_norm, dv_values, ks_values, vnom=0)
        severity_matrix = severity_matrix.reshape(npMx2_norm.shape)
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'powerflow_severity.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
       
    
        for plot_num in range(severity_matrix.shape[0]):
            print('Processing contingency %d of %d' %(plot_num+1, severity_matrix.shape[0]+1))
            plot_x = np.linspace(1,severity_matrix[plot_num,:].shape[0],severity_matrix[plot_num,:].shape[0])
            plot_y = severity_matrix[plot_num,:]
            plot_x = plot_x.reshape(plot_y.shape)
            title_temp = 'The power flow severity of contigency  %d' %(plot_num+1)
            plot_severity(pltPdf, plot_x, plot_y,title_temp) # places a plot page into the pdf file                              
            
        pltPdf.close() # closes a pdf file
        
    if True: # contigency severity cumsum
        
        dirin = 'results/'
        fnamein = 'powerflow_limit.csv'
        dfMx_limit = read_cont_mx1(dirin, fnamein)
        npMx_limit = dfMx_limit.to_numpy()
        npMx_limit = npMx_limit.T
    
        dv_values = np.array([0.80, 1.0])
        ks_values = np.array([1., 2.])
        
        npMx2_norm = npMx2/npMx_limit
        # severity_matrix = calculate_powerflow_severity(npMx2_norm, dv_values, ks_values)
        severity_matrix = calculate_voltage_severity(npMx2_norm, dv_values, ks_values, vnom=0)
        severity_matrix = severity_matrix.reshape(npMx2_norm.shape)
    if True:    
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'powerflow_severity_cumulative.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
       
    
        for plot_num in range(severity_matrix.shape[0]):
            print('Processing contingency %d of %d' %(plot_num+1, severity_matrix.shape[0]+1))
            plot_x = np.linspace(1,severity_matrix[plot_num,:].shape[0],severity_matrix[plot_num,:].shape[0])
            plot_y = severity_matrix[plot_num,:]
            plot_x = plot_x.reshape(plot_y.shape)
            plot_y = np.cumsum(plot_y, axis=0)
            title_temp = 'The power flow cumulative severity of contigency  %d' %(plot_num+1)
            plot_severity(pltPdf, plot_x, plot_y,title_temp) # places a plot page into the pdf file                              
            
        pltPdf.close() # closes a pdf file
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'powerflow_severity_cumulative_sum.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
       
    
        plot_x = np.linspace(1,severity_matrix[:,0].shape[0],severity_matrix[:,0].shape[0])
        plot_y = np.sum(severity_matrix, axis = 1)
        plot_x = plot_x.reshape(plot_y.shape)
        #plot_y = np.cumsum(plot_y, axis=0)
        title_temp = 'The power flow severity of contigency on all line ' 
        plot_severity(pltPdf, plot_x, plot_y,title_temp) # places a plot page into the pdf file                              
            
        pltPdf.close() # closes a pdf file
    
    
    if True: # plot heat map
        try:
            dirplots = 'plots/' # must create this relative directory path before running
            fnameplot = 'powerflow_severity_heatmap.pdf' # file name to save the plot
            pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
            severity_matrix_df = pd.DataFrame(severity_matrix)
            xymax = severity_matrix_df.shape 
            title = 'Heat map of power flow severity'
            output_continuous_heatmap_page(pltPdf, severity_matrix_df,xymax,pagetitle=title)
            pltPdf.close() # closes a pdf file
        except:
            logging.info('** something failed' )
    
    
    
    logging.shutdown()
    print('end')
        
        
        