# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 08:41:52 2020
@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com

vX.Y JZBYYYYMMDD
As we make changes the latest revisions are kept on top.

v1.0 JZB20201125
Importing the npcc case in matpower format, solving the loadflow, and saving 
the results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb

from vectorized_severity import calculate_voltage_severity

#%% read_contingencies
def read_cont_mx1(dirin, fname):
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
    
    # calculating the number of engaged elements based on standard ANDES vars
    temp = ss.ShuntSw.bs.v*ss.ShuntSw.ns.v # b*n, where b is an element's admittance and n is a number of elements
    bav = np.array(np.sum(arr) for arr in temp) # total b avaialable on a given bus
    ix = np.array(ss.ShuntSw.bus.v, dtype=int) # bus numbers of switched shunt devices
    
    
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

    if True: # loading the npcc system model with shunt capacitors
        dirin = 'cases/'
        fnamein = 'caseNPCC.xlsx'
        
        dirout = 'output/' # output directory
        foutroot = fnamein.replace('.xlsx', '')
        # Solving the load flow with all shunt capacitors active

		ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
		if ss.exit_code == 0: 
			save_results(ss, dirout, foutroot)
		else:
			logging.info('Load flow did not converge with all switched shunts in service')

		# 20210111 TODO: Tianwei
		# intialize the numpy matrices to record the results
		# create a for-loop that:
		#   - applies a line outage
		#   - solves the load flow
		#   - adds bus voltages and line loadings as rows to the numpy matrices, but
		#     checks for unconverged case and stuffs the row with np.nan
		# 
		# after the for-loop is done: 
		# make dataframes from the numpy matrices and 
		# save them into csv files.
		# 
		# 20210111 TODO: Jovan
		# intialize the numpy matrices to record the results
		# create a for-loop that:
		#   - applies a generator outage
		#   - rallocates the lost output to other generators as discussed in Daily Call Notes 20210111
		#   - solves the load flow
		#   - adds bus voltages and line loadings as rows to the numpy matrices, but
		#     checks for unconverged case and stuffs the row with np.nan
		# 
		# after the for-loop is done: 
		# make dataframes from the numpy matrices and 
		# save them into csv files.
		

    if False: # loading the andes file with manually added switched capacitors
        dirin = 'cases/'
        fnamein = 'caseNPCC.xlsx'
        
        dirout = 'output/' # output directory
        # ANDES output file names are derived from the input file name with an added suffix
        # The suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
        
        # deriving the root name for output files based on the input file name
        foutroot = fnamein.replace('.xlsx', '')
        save_results(ss, dirout, foutroot)

    if False: # extracting the buses with added capacitors
        dirin = 'results/'
        fnamein = '0v93_full_mx5.csv'
        dfMx5 = read_cont_mx1(dirin, fnamein)
        print()
        print('The shape of Mx5 is (%d, %d)' %(dfMx5.shape[0], dfMx5.shape[1]))
        # find columns with non-zero sums
        df1 = dfMx5.sum(axis=0) 
        ix = df1[df1.values != 0].index
        busnums = [int(num.split('_')[1]) for num in ix.tolist()]
        capcounts = df1.loc[ix].tolist()
        print('Bus numbers: ', busnums)
        print('Cap counts:  ', capcounts)
        # Used this to set up the Excel file with switched shunt compensation
        
    if False: # loading the contingency results
        dirin = 'results/'
        fnamein = '0v93_full_mx1.csv'
        
        # loding in Mx1
        dfMx1 = read_cont_mx1(dirin, fnamein)
        print()
        print('The shape of Mx1 is (%d, %d)' %(dfMx1.shape[0], dfMx1.shape[1]))
        print('There are %d unique iteration numbers' %(len(dfMx1.iloc[:,0].unique())))
        
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


    # preparing for exit
    logging.shutdown()
