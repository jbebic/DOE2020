# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:06:23 2021

@author: txia4@vols.utk.edu

v1.0 TX20210115
This file is compute the severity based on the bus voltage and line flow data
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import matplotlib.backends.backend_pdf as dpdf # pdf output

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb

from vectorized_severity import calculate_voltage_severity
from npcc_powerflow_severity import read_cont_mx1
from review_shunt_compensation import output_continuous_heatmap_page

#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    if True: # read the information from Matpower
        dirin = 'output/'
        
        # loding in Mx1
        fnamein = 'basecase_apparentpower.csv'
        dfMx_base_S = read_cont_mx1(dirin, fnamein)
        npMx_base_S = dfMx_base_S.to_numpy()
        
        fnamein = 'changecase_apparentpower.csv'
        dfMx_change_S = read_cont_mx1(dirin, fnamein)
        npMx_change_S = dfMx_change_S.to_numpy()
        
        fnamein = 'basecase_busvoltage.csv'
        dfMx_base_V = read_cont_mx1(dirin, fnamein)
        npMx_base_V = dfMx_base_V.to_numpy()
        
        fnamein = 'changecase_busvoltage.csv'
        dfMx_change_V = read_cont_mx1(dirin, fnamein)
        npMx_change_V = dfMx_change_V.to_numpy()

 
    if False: # plot heat map of power
        
        # converge_list_matpower is [1 14 15 16 17 20 25 33 34 47 48 57 60 110 111 113 117 139 141 194 208 233]-1
        # converge_list_andes is [1 2 16 17 20 22 25 33 34 39 48 110 111 113 117 194 233]
        mannul_nonconver_list = [0,14,15,16,19,24,32,33,46,47,56,59,109,110,112,116,138,140,193,207,232]  
        npMx_base_S[mannul_nonconver_list,:] = np.nan
        npMx_change_S[mannul_nonconver_list,:] = np.nan
        
        nonconver_list = np.where(np.isnan(npMx_base_S[:,0]))
        #read the limit
        dirin = 'results/'
        fnamein = 'powerflow_limit.csv'
        dfMx_limit = read_cont_mx1(dirin, fnamein)
        npMx_limit = dfMx_limit.to_numpy()
        npMx_limit = np.dot(npMx_limit,0.01)
        npMx_limit = npMx_limit.T

    
        dv_values = np.array([0.8, 1.0, 1.25, 1.725])
        ks_values = np.array([5, 10, 15, 20]) 
        
        npMx_base_S_norm = npMx_base_S/npMx_limit
        npMx_change_S_norm = npMx_change_S/npMx_limit
        # severity_matrix = calculate_powerflow_severity(npMx2_norm, dv_values, ks_values)
        severity_matrix_base = calculate_voltage_severity(npMx_base_S_norm, dv_values, ks_values, vnom=0)
        severity_matrix_base = severity_matrix_base.reshape(npMx_base_S_norm.shape)
        severity_matrix_change = calculate_voltage_severity(npMx_change_S_norm, dv_values, ks_values, vnom=0)
        severity_matrix_change = severity_matrix_change.reshape(npMx_change_S_norm.shape)
           
        try:
            dirplots = 'plots/' # must create this relative directory path before running
            fnameplot = 'Line-loading severity heatmap based on andes.pdf' # file name to save the plot
            pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
            severity_matrix_base_df = pd.DataFrame(severity_matrix_base)
            severity_matrix_change_df = pd.DataFrame(severity_matrix_change)
            xymax = severity_matrix_base_df.shape 
            title = 'Heat map of line-loading severity of base case'
            output_continuous_heatmap_page(pltPdf, severity_matrix_base_df,xymax, xylabels = ['Contingency number', 'Line number'],pagetitle=title)
            title = 'Heat map of line-loading severity of change case'
            output_continuous_heatmap_page(pltPdf, severity_matrix_change_df,xymax, xylabels = ['Contingency number', 'Line number'],pagetitle=title)
            pltPdf.close() # closes a pdf file
        except:
            logging.info('** something failed' )
            
    if True: # plot heat map of voltage
        
        # converge_list_matpower is [1 14 15 16 17 20 25 33 34 47 48 57 60 110 111 113 117 139 141 194 208 233]-1
        # converge_list_andes is [1 2 16 17 20 22 25 33 34 39 48 110 111 113 117 194 233]
        mannul_nonconver_list = [0,14,15,16,19,24,32,33,46,47,56,59,109,110,112,116,138,140,193,207,232]              
        # they are actually non_converge
        npMx_base_V[mannul_nonconver_list,:] = np.nan
        npMx_change_V[mannul_nonconver_list,:] = np.nan
        
        nonconver_list = np.where(np.isnan(npMx_base_S[:,0]))
        
        dv_values = np.array([0.03, 0.05, 0.08])
        ks_values = np.array([5, 10, 15])
        
        severity_matrix_base = calculate_voltage_severity(npMx_base_V, dv_values, ks_values, vnom=1)
        severity_matrix_base = severity_matrix_base.reshape(npMx_base_V.shape)
        severity_matrix_change = calculate_voltage_severity(npMx_change_V, dv_values, ks_values, vnom=1)
        severity_matrix_change = severity_matrix_change.reshape(npMx_change_V.shape)
        
        try:
            dirplots = 'plots/' # must create this relative directory path before running
            fnameplot = 'Bus voltage severity heatmap based on andes.pdf' # file name to save the plot
            pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
            severity_matrix_base_df = pd.DataFrame(severity_matrix_base)
            severity_matrix_change_df = pd.DataFrame(severity_matrix_change)
            xymax = severity_matrix_base_df.shape 
            title = 'Heat map of bus_voltage severity of base case'
            output_continuous_heatmap_page(pltPdf, severity_matrix_base_df,xymax, xylabels = ['Contingency number', 'Line number'],pagetitle=title)
            title = 'Heat map of bus_voltage severity of change case'
            output_continuous_heatmap_page(pltPdf, severity_matrix_change_df,xymax, xylabels = ['Contingency number', 'Line number'],pagetitle=title)
            pltPdf.close() # closes a pdf file
        except:
            logging.info('** something failed' )


    print('end')
    # preparing for exit
    logging.shutdown()
    