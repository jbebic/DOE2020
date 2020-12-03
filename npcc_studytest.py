# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:15:04 2020

@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com; txia4@vols.utk.edu

v1.2 TX20201203


v1.1 Tx20201130

Add the code for ploting the matrix1
Try to understand the structure of adnes result

v1.0 TX20201130

For study purpose.
Importing the npcc case in matpower format, solving the loadflow, and saving 
the results.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy.matlib 

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb
#%% function of voltage severity
def voltage_severity(V_value =  1):
    k1 = 0.5
    k2 = 2
    threshold1 = 0.05
    threshold2 = 0.1
    V_temp = abs(V_value) - 1
    if abs(V_temp) < threshold1:
        result = 0
    elif abs(V_temp) < threshold2:
        result = (abs(V_temp) - threshold1) * k1
    else:
        result =(abs(V_temp)- threshold2) * k2 +(threshold2 - threshold1) * k1
    return result

#%% read_contingencies
def read_cont_mx1(dirin, fname):
    
    logging.debug('reading contigency matrix')
    df = pd.read_csv(os.path.join(dirin, fname))
    return df

#%% save_results
def save_results(ss:andes.system, dirout:str, foutroot:str):

    logging.debug('saving ANDES voltage magnitudes data')
    # fetch the names of the buses and set up as columns of a dataframe
    dfBus = pd.DataFrame(columns=ss.Bus.name.v)
    # add voltage magnitudes as rows of the dataframe
    fout = os.path.join(dirout,foutroot + '.vmag.csv')
    dfBus.loc[len(dfBus)] = ss.Bus.v.v
    dfBus.to_csv(fout, index=False)

    return

#%% save_rinputs
def save_inputs(ss:andes.system, dirout:str, foutroot:str):
    # Optional debugging message
    logging.debug('entered save_inputs function')

    # saving ANDES bus data
    logging.debug('saving ANDES bus data')
    fout = os.path.join(dirout,foutroot + '.bus.csv')
    dfBus = ss.Bus.as_df()
    dfBus.to_csv(fout, index=False)

    # saving ANDES lines data
    logging.debug('saving ANDES line data')
    fout = os.path.join(dirout,foutroot + '.line.csv')
    dfLine = ss.Line.as_df()
    dfLine.to_csv(fout, index=False)

    # saving ANDES generator data
    logging.debug('saving ANDES generator data')
    fout = os.path.join(dirout,foutroot + '.gen.csv')
    dfGen = ss.PV.as_df()
    dfGen.to_csv(fout, index=False)
    
    return

#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    if False: # loading the contingency results
        dirin = 'results/'
        fnamein = '0v93_full_mx1.csv'
        
        # loding in Mx1
        dfMx1 = read_cont_mx1(dirin, fnamein)
        print()
        print('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        print('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        
        plot_x = np.linspace(1,140,140)
        iter_num = 1
        bus_num = 2
        plot_y = dfMx1.iloc[(iter_num-1)*140:iter_num*140,bus_num+1]
        figure1 = plt.plot(plot_x, plot_y,'C5')
        plt.xlabel('contigency number')
        plt.ylabel('bus voltage')
        plt.title('The bus voltage')
  
        
        # We can now pass the dfMx1 to a function that can calculate things based on the contingency data
        # for example the severity metric.

    if False: # loading the matpower file
        dirin = 'cases/'
        fnamein = 'caseNPCC.m'
        
        dirout = 'output/' # output directory
        # ANDES output file names are derived from the input file name with an added suffix
        # The suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
        
        # deriving the root name for output files based on the input file name
        foutroot = fnamein.replace('.m', '')
        save_results(ss, dirout, foutroot)
  
    if False: # Loading the PSS/E file
        dirin = 'cases/'
        fnamein = 'npcc.raw'
        
        dirout = 'output/' # output directory
        # ANDES output file names are derived from the input file name with an added suffix
        # The suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
        
        # deriving the root name for output files based on the input file name
        foutroot = fnamein.replace('.m', '')
        save_results(ss, dirout, foutroot)
        
    if True: # loading the contingency results and test voltage severtiy
        dirin = 'results/'
        fnamein = '0v93_full_mx1.csv'
        
        # loding in Mx1
        dfMx1 = read_cont_mx1(dirin, fnamein)
        print()
        print('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        print('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        
        voltage_sevrity = np.matlib.zeros((4640,141))
        for i in range(dfMx1.shape[0]):
            for j in range(dfMx1.shape[1]):
                temp = dfMx1.iloc[i,j]
                if np.isnan(temp):
                    voltage_sevrity[i,j] = np.nan
                else:
                    voltage_sevrity[i,j] = voltage_severity(temp)
                    

                
                
        
        
    
    


    # preparing for exit
    logging.shutdown()
