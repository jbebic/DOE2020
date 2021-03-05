# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:51:56 2021

@author: tianw
"""

import logging
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt # plotting
# import matplotlib.backends.backend_pdf as dpdf # pdf output

# from datetime import datetime # time stamps
# import os # operating system interface

import andes
# from andes.core.var import BaseVar, Algeb, ExtAlgeb

# from vectorized_severity import calculate_voltage_severity
# from reconfigure_case import area_generators_indices,area_loads_indices
# from npcc_powerflow_severity import read_cont_mx1
# from npcc_contigency_test import compute_lineapparentpower
# from review_shunt_compensation import output_continuous_heatmap_page
# from npcc_contigency_test import save_database
# from npcc_contigency_areastudy import area_num_detect,sort_line


#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    casedir = 'cases/'
    casefile = 'caseNPCC_wAreas.xlsx'
    outdir = 'results/'
    ss = andes.load(casefile, input_path=casedir, setup=False)

    
    ss.PFlow.config.max_iter = 100
    ss.setup()      # setup system  
    ss.PFlow.run()  # run power flow  

 
    if True: # the generation displacement "inner loop" "
            
            # islanding_list = [0,13,14,15,16,19,21,24,32,33,38,47,109,110,112,116,193,232]
            # islanding_generator_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
           
            # for line_con_num in islanding_list:
            for line_con_num in range(1):       # only conider the first contigency line_con_num = 0
                
                ss.Line.u.v[line_con_num] = 0  #  trip the transmission line #line_con_num

                # shut down the generator     
                islanding_gen_uid = 0                    
                ss.PV.u.v[islanding_gen_uid] = 0
                
                #run pf
                ss.PFlow.init() # helps with the initial guess
                ss.PFlow.run() # run power flow
                
                #recover the system back to original    
                ss.PV.u.v[islanding_gen_uid] = 1
                ss.Line.u.v[line_con_num] = 1
                  
  
    # flatten/reshape for saving data     
    # preparing for exit
    logging.shutdown()
    print('end')
