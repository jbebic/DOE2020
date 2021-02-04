# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:13:40 2021

@author: txia4@vols.utk.edu

v1.0 TX20210127
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

from reconfigure_case import area_generators_indices,area_loads_indices
from vectorized_severity import calculate_voltage_severity
from npcc_powerflow_severity import read_cont_mx1
from review_shunt_compensation import output_continuous_heatmap_page
from reconfigure_case import area_interface_lines_indices, area_generators_indices

#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    
    if  True: # read the information from Matpower
        

        
        casedir = 'cases/'
        casefile = 'caseNPCC_wAreas.xlsx'
        outdir = 'results/'
        ss = andes.load(casefile, input_path=casedir, setup=False)
        
        
        area_idx = 2
        line_total_num = len(ss.Line.u.v)
        bus_total_num = len(ss.Bus.u.v)
        select_bus_list = area_generators_indices(ss, area_idx)
        gen_area_toal_num = len(select_bus_list)
        
    
    
        dirin = 'output/'
        
        # loding in Mx1
        fnamein = 'basecase_apparentpower.csv'
        dfMx_base_S = read_cont_mx1(dirin, fnamein)
        npMx_base_S = dfMx_base_S.to_numpy()
        
        fnamein = 'changecase_apparentpower.csv'
        dfMx_change_S = read_cont_mx1(dirin, fnamein)
        npMx_change_S = dfMx_change_S.to_numpy()
        apparentpower_database = npMx_change_S.reshape(gen_area_toal_num,line_total_num,line_total_num)
        # npMx_change_S = apparentpower_database[4,:,:]
        
        fnamein = 'basecase_busvoltage.csv'
        dfMx_base_V = read_cont_mx1(dirin, fnamein)
        npMx_base_V = dfMx_base_V.to_numpy()
        
        fnamein = 'changecase_busvoltage.csv'
        dfMx_change_V = read_cont_mx1(dirin, fnamein)
        npMx_change_V = dfMx_change_V.to_numpy()
        busvoltage_database = npMx_change_V.reshape(gen_area_toal_num,line_total_num,bus_total_num)
        # npMx_change_V = busvoltage_database[4,:,:]
        
        # converge_list_matpower is [1 14 15 16 17 20 25 33 34 47 48 57 60 110 111 113 117 139 141 194 208 233]-1
        # converge_list_andes is [1 2 16 17 20 22 25 33 34 39 48 110 111 113 117 194 233]
        # mannul_nonconver_list = [0,14,15,16,19,24,32,33,46,47,56,59,109,110,112,116,138,140,193,207,232]  

 
    if True: # plot heat map of power
        
        # nonconver_list = np.where(np.isnan(npMx_base_S[:,0]))
        #read the limit
        dirin = 'results/'
        fnamein = 'line_loading_limits.csv'
        # dfMx_limit = read_cont_mx1(dirin, fnamein)
        # npMx_limit = dfMx_limit.to_numpy()
        # npMx_limit = np.dot(npMx_limit,0.01)
        # npMx_limit = npMx_limit.T
        dfLineLims = read_cont_mx1(dirin, fnamein)
        npMx_limit = dfLineLims['new_limit'].values/100

        casedir = 'cases/'
        casefile = 'caseNPCC_wAreas.xlsx'
        outdir = 'results/'
        
        print('generators in area ',area_generators_indices(ss,area_idx))
        il_idxs = area_interface_lines_indices(ss, area_idx)
        il_lims = [npMx_limit[i] for i in il_idxs]
        print('Interface line limits')
        for name, lim in zip([ss.Line.name.v[i] for i in il_idxs], il_lims):
            print('  %s: %f [pu]' %(name, lim))
    
        dp_values = np.array([0.8, 1.0, 1.25, 1.725])
        ks_values = np.array([5, 10, 15, 20])
        
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Line-loading severity heatmap based on andes.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        # npMx_base_S[mannul_nonconver_list,:] = np.nan
        npMx_base_S_norm = npMx_base_S/npMx_limit
        severity_matrix_base = calculate_voltage_severity(npMx_base_S_norm, dp_values, ks_values, vnom=0)
        severity_matrix_base = severity_matrix_base.reshape(npMx_base_S_norm.shape)
        # temp = calculate_voltage_severity(npMx_base_S_norm, dv_values, ks_values, vnom=0)
        # severity_matrix_base = temp.reshape(npMx_base_S_norm.shape)
        if False:
            severity_matrix_base_df = pd.DataFrame(severity_matrix_base)
            title = 'Line-loading severity, base case'
        else:
            severity_matrix_base_df = pd.DataFrame(npMx_base_S_norm)
            title = 'Line-loading, base case'
        xymax = severity_matrix_base_df.shape 
        output_continuous_heatmap_page(pltPdf, 
                                       severity_matrix_base_df, 
                                       xymax, 
                                       xylabels = ['Contingency number', 'Line number'], 
                                       pagetitle=title, 
                                       crange=[0,2]
                                       )        
        for select_gen_num in range(gen_area_toal_num):  
            
            npMx_change_S = apparentpower_database[select_gen_num,:,:]
            # npMx_change_S[mannul_nonconver_list,:] = np.nan
            npMx_change_S_norm = npMx_change_S/npMx_limit
            severity_matrix_change = calculate_voltage_severity(npMx_change_S_norm, dp_values, ks_values, vnom=0)
            severity_matrix_change = severity_matrix_change.reshape(npMx_change_S_norm.shape)           
            try:
                if False:
                    severity_matrix_change_df = pd.DataFrame(severity_matrix_change)
                    title = 'Line-loading severity, change case %d' %(select_gen_num+1)
                else:
                    severity_matrix_change_df = pd.DataFrame(npMx_change_S_norm)
                    title = 'Line-loading, change case %d' %(select_gen_num+1)
                output_continuous_heatmap_page(pltPdf, 
                                               severity_matrix_change_df, 
                                               xymax, 
                                               xylabels = ['Contingency number', 'Line number'], 
                                               pagetitle=title, 
                                               crange=[0,2]
                                               )
              
            except:
                logging.info('** something failed' )
      
        pltPdf.close() # closes a pdf file
       
            
    if False: # plot heat map of voltage
        
            
        # they are actually non_converge
     
        # nonconver_list = np.where(np.isnan(npMx_base_S[:,0]))
        
        dv_values = np.array([0.03, 0.05, 0.08])
        ks_values = np.array([5, 10, 15])
        
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Bus voltage severity heatmap based on andes.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        npMx_base_V[mannul_nonconver_list,:] = np.nan
        severity_matrix_base = calculate_voltage_severity(npMx_base_V, dv_values, ks_values, vnom=1)
        severity_matrix_base = severity_matrix_base.reshape(npMx_base_V.shape)
        severity_matrix_base_df = pd.DataFrame(severity_matrix_base)
        xymax = severity_matrix_base_df.shape 
        title = 'Heat map of bus_voltage severity of base case'
        output_continuous_heatmap_page(pltPdf, severity_matrix_base_df,xymax, xylabels = ['Contingency number', 'Line number'], pagetitle=title)
        for select_gen_num in range(gen_area_toal_num):
            
            npMx_change_V = busvoltage_database[select_gen_num,:,:]
            npMx_change_V[mannul_nonconver_list,:] = np.nan
            severity_matrix_change = calculate_voltage_severity(npMx_change_V, dv_values, ks_values, vnom=1)
            severity_matrix_change = severity_matrix_change.reshape(npMx_change_V.shape)
            try:
                severity_matrix_change_df = pd.DataFrame(severity_matrix_change)
                title = 'Heat map of bus_voltage severity of change case %d' %(select_gen_num+1)
                output_continuous_heatmap_page(pltPdf, severity_matrix_change_df, xymax, xylabels = ['Contingency number', 'Line number'], pagetitle=title)
            except:
                logging.info('** something failed' )
        pltPdf.close() # closes a pdf file


    print('end')
    # preparing for exit
    logging.shutdown()
    