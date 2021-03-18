# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:08:53 2021



@author: txia4@vols.utk.edu
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
        
        # loding database
        fnamein = 'bus_voltages.csv'
        dfMX_V_database = read_cont_mx1(dirin, fnamein)
        npMX_V_database = dfMX_V_database.to_numpy()
        npMX_V_database = np.delete(npMX_V_database,0,axis = 1) # delete the first column because that's the label of generator
        npMX_V_database = npMX_V_database.reshape(gen_area_toal_num+1,line_total_num+1,bus_total_num) #+1 becasue we have the base case
        
        
        npMx_base_V = npMX_V_database[0,:,:]
        npMX_displace_V = npMX_V_database[1:,:,:]
        
        fnamein = 'line_flows.csv'
        dfMX_S_database = read_cont_mx1(dirin, fnamein)
        npMX_S_database = dfMX_S_database.to_numpy()
        
        npMX_S_database = np.delete(npMX_S_database,0,axis = 1) # delete the first column because that's the label of generator
        npMX_S_database = npMX_S_database.reshape(gen_area_toal_num+1,line_total_num+1,line_total_num) #+1 becasue we have the base case
        
        
        npMx_base_S = npMX_S_database[0,:,:]
        npMX_displace_S = npMX_S_database[1:,:,:]
        
 
    if True: # plot heat map of power
        
        # nonconver_list = np.where(np.isnan(npMx_base_S[:,0]))
        #read the limit
        dirin = 'results/'
        fnamein = 'line_loading_limits.csv'

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
        fnameplot = 'flow_heatmap_absolute.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        # npMx_base_S[mannul_nonconver_list,:] = np.nan
        npMx_base_S_norm = npMx_base_S/npMx_limit
        flow_matrix_base_df = pd.DataFrame(npMx_base_S_norm)
        title = 'Line-loading, base case, PV0'
        xymax = flow_matrix_base_df.shape 
        output_continuous_heatmap_page(pltPdf, 
                                       flow_matrix_base_df, 
                                       xymax, 
                                       xylabels = ['Contingency number', 'Line number'], 
                                       pagetitle=title, 
                                       crange=[0,2]
                                       )        
        for select_gen_num in range(gen_area_toal_num):  
            
            npMx_change_S = npMX_displace_S[select_gen_num,:,:]
            # npMx_change_S[mannul_nonconver_list,:] = np.nan
            npMx_change_S_norm = npMx_change_S/npMx_limit       
            flow_matrix_change_df = pd.DataFrame(npMx_change_S_norm)
            title = 'Line-loading,  change case, PV %d' %(ss.PV.bus.v[select_gen_num])
            output_continuous_heatmap_page(pltPdf, 
                                           flow_matrix_change_df, 
                                           xymax, 
                                           xylabels = ['Contingency number', 'Line number'], 
                                           pagetitle=title, 
                                           crange=[0,2]
                                           )
                    
        pltPdf.close() # closes a pdf file
        
   

    print('end')
    # preparing for exit
    logging.shutdown()