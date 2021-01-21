# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:45:06 2020

@author: txia4@vols.utk.edu

v1.7 TX20210106
change the title and label based on nomenclature 
add Bus-voltage severity difference of full system part


V1.6 TX20210105
Add the nonconver case by the whtie stripe
Correct the title

v1.5 TX20201220
Computhe the cumulative severity of each iteration. The result is saved in powerflow_severity_cumulative.pdf

v1.4 TX20201214
Add the reference of each figure

v1.3 TX20201211
Change the title of figures

v1.2 TX20201210
Save the file in PDF

v1.0 TX20201205
Combine the vector computation and serverity together
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy.matlib 
import matplotlib.backends.backend_pdf as dpdf # pdf output

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb
from review_shunt_compensation import output_continuous_heatmap_page

#%% read_contingencies
def read_cont_mx1(dirin, fname):
    
    logging.info('reading contigency matrix')
    df = pd.read_csv(os.path.join(dirin, fname))
    return df


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
#%% 
def plot_severity(pltPdf, plot_x, plot_y, title='', xlabel='Contigency number',ylabel='Apparent Power Severity',ref = False, sort = False,step=False):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.8), dpi=300) # , sharex=True
    plt.suptitle(title)
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
    if step:
        ax0.step(plot_x, plot_y, label='Vseverity')
    else:
        ax0.plot(plot_x, plot_y, label='Vseverity')
    if sort:
       dy = np.diff(plot_y.T).T
       dys = np.sort(dy,axis=0)
       plot_y2 = plot_y[0]+np.cumsum(dys)
       plot_y2a = np.hstack((plot_y[0], plot_y2))
       plot_y2b = plot_y2a.T
       ax0.plot(plot_x, plot_y2b)
       ax0.set_xticks(np.arange(0,22,2))
 
    if ref:
        plot_ref(ax0, ref_matrix)
        ax0.set_xticks(np.arange(0,22,2))

    ax0.grid(True, which='both', axis='both')
    #ax0.set_ylim([0,0.3]) # when range not specified, the graph autoscales 
    ax0.set_ylabel(ylabel)
    
    ax0.legend(('Vseverity','0.90','0.92','0.95'))
    if sort:
        ax0.legend(('Original','Optimized','0.90','0.92','0.95'))
        
    #ax0.set_xlabel('Voltage [pu]')
    ax0.set_xlabel(xlabel)
    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.title(title)
    plt.close() # Closes fig to clean up memory
    return

 #%%   
def plot_ref(ax0, ref_matrix):
    colorbase = ['r', 'g', 'm']
    for i in range(ref_matrix.shape[1]):
        plot_xref = np.linspace(1,ref_matrix.shape[0],ref_matrix.shape[0])
        plot_yref = ref_matrix[:,i]
        c = colorbase[i]
        ax0.plot(plot_xref, plot_yref,c)
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
        fnamein = '0v93_full_mx1.csv'
        
        # loding in Mx1
        dfMx1 = read_cont_mx1(dirin, fnamein)
        print()
        #print('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        #print('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        logging.info('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        logging.info('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        
        # define the transfer function of severity (positive values only, non-decreasing)
        dv_values = np.array([0.03, 0.05, 0.08])
        ks_values = np.array([5, 10, 15])
        
        npMx1 = dfMx1.to_numpy()
        npMx1 = np.delete(npMx1, 0, axis=1)             # the first column is the iteration number so delete it
        bus_totalnum = npMx1.shape[1]
        iteration_totalnum = 21
        contigency_totalnum = int(npMx1.shape[0]/iteration_totalnum)
        
        severity_matrix = calculate_voltage_severity(npMx1, dv_values, ks_values)
        severity_matrix = severity_matrix.reshape(npMx1.shape)
        
        ref_val = np.array([0.9, 0.92, 0.95])
        refMx1 = ref_val*np.ones((iteration_totalnum,ref_val.shape[0]))
        ref_matrix = calculate_voltage_severity(refMx1, dv_values, ks_values)
        ref_matrix = ref_matrix.reshape(refMx1.shape)
        
        
         # compute severity for each contigency
        weight_vector = np.matlib.ones((severity_matrix.shape[1],1))
        severity_vector = np.matlib.zeros((dfMx1.shape[0],1))
        for i in range(severity_matrix.shape[0]):
            temp_dot = np.dot(severity_matrix[i,:],weight_vector)
            severity_vector[i,0] = np.sum(temp_dot)
        
        severity_vector = severity_vector.reshape(iteration_totalnum,contigency_totalnum)
        severity_vector = np.nan_to_num(severity_vector)   # replace the Nan with 0
        severity_vector_sum = np.sum(severity_vector, axis = 1) 
 
    if False:    #plot the severity of different iteration under certain contigency
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Bus-voltage severity.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        for plot_num in range(severity_vector.shape[1]):
            print('Processing contingency %d' %(plot_num))
            plot_x = np.linspace(1,severity_vector[:,plot_num].shape[0],severity_vector[:,plot_num].shape[0])
            plot_y = severity_vector[:,plot_num]
            plot_x = plot_x.reshape(plot_y.shape)
            title_temp = 'Line contigency # %d' %(plot_num+1)
            plot_severity(pltPdf, plot_x, plot_y,title_temp,'Iteration number','Bus-voltage severity',ref = 1) # places a plot page into the pdf file                              
            #plt.title('Line contigency # %d' %(plot_num))
        
        pltPdf.close() # closes a pdf file
     
              
        plot_x = np.linspace(1,severity_vector_sum.shape[0],severity_vector_sum.shape[0])
        plot_y = severity_vector_sum
        plot_x = plot_x.reshape(plot_y.shape)
        ref_matrix = ref_matrix*contigency_totalnum       
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Bus-voltage severity of full system.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        plot_severity(pltPdf, plot_x, plot_y,'Bus-voltage severity','Iteration number','Bus-voltage severity',ref=True) # places a plot page into the pdf file
        plot_severity(pltPdf, plot_x, plot_y,'Bus-voltage severity','Iteration number','Bus-voltage severity',ref=True,sort=True) # places a plot page into the pdf file
        pltPdf.close() # closes a pdf file
        
        plot_x = np.linspace(2,severity_vector_sum.shape[0],severity_vector_sum.shape[0]-1)
        plot_y = -np.diff(severity_vector_sum.T).T
        plot_x = plot_x.reshape(plot_y.shape)
                  
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Bus-voltage severity difference of full system.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        plot_severity(pltPdf, plot_x, plot_y,'Bus-voltage severity','Iteration number','Bus-voltage severity') # places a plot page into the pdf file
        
        pltPdf.close() # closes a pdf file
       
        
       
        
        
    if True: # loading the contingency results and test voltage severtiy
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Bus-voltage severity sum.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        severity_matrix = numpy.nan_to_num(severity_matrix)
        for iter_num in range(iteration_totalnum):
            plot_x = np.linspace(1,contigency_totalnum, contigency_totalnum)
            severity_matrix_piece = severity_matrix[iter_num*contigency_totalnum:(iter_num+1)*contigency_totalnum,:]
            plot_y = np.sum(severity_matrix_piece, axis = 1)
            title_temp = 'Iteration # %d' %(iter_num+1)
            plot_severity(pltPdf, plot_x, plot_y,title_temp,'Contigency number','Bus-voltage severity',ref = False, sort = False,step=True) # places a plot page into the pdf file  
            
        pltPdf.close() # closes a pdf file
      
    if True: # plot heat map
        
        dirin = 'results/'
        fnamein = 'contigency_apparentpower.csv'
        dfMx = read_cont_mx1(dirin, fnamein)
        npMx = dfMx.to_numpy()
        nonconver_list = np.asarray(np.where(npMx[:,0]==0)) # get the nonconvergy list
        
        try:
            dirplots = 'plots/' # must create this relative directory path before running
            fnameplot = 'Bus-voltage severity heatmap.pdf' # file name to save the plot
            pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
            
            
            for iter_num in range(iteration_totalnum):
                severity_matrix_piece = severity_matrix[iter_num*contigency_totalnum:(iter_num+1)*contigency_totalnum,:] 
                severity_matrix_piece = numpy.nan_to_num(severity_matrix_piece)
                nonconver_list = nonconver_list-2*np.ones(nonconver_list.shape)  # the matrix start from 0
                nonconver_list[0,0] = nonconver_list[0,0]+1
                nonconver_list = nonconver_list.astype('int64')
                severity_matrix_piece[nonconver_list,:] = np.nan   #replace the nonconverge row
                severity_matrix_piece_df = pd.DataFrame(severity_matrix_piece)
                xymax = severity_matrix_piece_df.shape 
                title = 'Heat map of Bus-voltage severity'
                subtitle = 'Iteration %d' %(iter_num+1)
                output_continuous_heatmap_page(pltPdf, severity_matrix_piece_df, xymax, pagetitle=title, axistitle=subtitle)
            pltPdf.close() # closes a pdf file
        except:
            logging.info('** something failed' ) 


    print('end')
    # preparing for exit
    logging.shutdown()
