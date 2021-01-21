# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:49:22 2021

@author: txia4@vols.utk.edu


non-converge case for Hantao
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb

# andes.main.config_logger(stream_level=10)

#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)     
            
    if True: # loading modified case
        
        dirin = 'cases/'
        fnamein = 'caseNPCC.xlsx'
        dirout = 'output/' # output directory
        foutroot = fnamein.replace('.xlsx', '')
        
        ss = andes.load(fnamein, setup=False)  # please change the path to case file. Note `setup=False`.
        line_total_num = 234-1    # there are 234 transmission line

        ss.ShuntSw.u.v = np.zeros_like(ss.ShuntSw.u.v).tolist() # shut down all capacitor
        
        # converge_list_matpower is [1 14 15 16 17 20 25 33 34 47 48 57 60 110 111 113 117 139 141 194 208 233]
        # converge_list_andes is [1 2 16 17 20 22 25 33 34 39 48 110 111 113 117 194 233]
        
        # ss.add("PV", dict(bus=38, p0=0.1, v0=1.0))
        # a = 1  # immutable
        # b = np.array([1, 2, 3])  # mutable
        
        # a = 2  # new allocation
        # b = np.array([4, 5])  # memory allocation, points `b` to the new location
        # b[:] = [5, 6]  # no memory allocation, modified in-place
        # c = [1, 2, 3]
        # c[0] = 4
        
        # for line_con_num in range(line_total_num):
        for line_con_num in [20]:
            ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
            try:
                ss.setup()      # setup system      
                ss.PFlow.run() # run power flow
            except:
                logging.info('Load flow did not converge in contigency %d' %(line_con_num+1))
                print('Load flow did not converge in contigency %d' %(line_con_num+1))
            ss.Line.u.v[line_con_num] = 1
        
        print('end')            
            
            

 