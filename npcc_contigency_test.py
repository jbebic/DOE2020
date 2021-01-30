# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:11:20 2021


@author: txia4@vols.utk.edu

V1.0 TX20210115
Fix the bug.
Compute the bus voltage as well.


v1.0 TX20210111
Try to achieve line outage on each transmission line
There is bug in the loop
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
from npcc_powerflow_severity import read_cont_mx1

#%% save_results
def save_database(input_database, dirout:str, foutroot:str):

    logging.debug('saving ANDES data')
    # fetch the names of the buses and set up as columns of a dataframe
    dfBus = pd.DataFrame(input_database)
    # add voltage magnitudes as rows of the dataframe
    fout = os.path.join(dirout,foutroot + '.csv')
    dfBus.to_csv(fout, index=False)
       
    return

#%% line apparent power
def compute_lineapparentpower(ss:andes.system):
    
    p1 = ss.Line.a1.e
    p2 = ss.Line.a2.e
    q1 = ss.Line.v1.e
    q2 = ss.Line.v2.e
    s1 = np.square(p1)+np.square(q1)
    s2 = np.square(p2)+np.square(q2)
    s = np.sqrt(np.amax(np.stack((s1,s2)),axis = 0))
    return s


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
        # andes.main.config_logger()       
        ss = andes.load(fnamein, setup=False)  # please change the path to case file. Note `setup=False`.
        line_total_num = 234-1
        bus_total_num = 140
        apparentpower_database = np.zeros((line_total_num,line_total_num))
        busvoltage_database = np.zeros((line_total_num,bus_total_num))
        ss.ShuntSw.u.v = np.zeros_like(ss.ShuntSw.u.v).tolist()
        for line_con_num in range(line_total_num):
            ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
            try:
                ss.setup()      # setup system      
                ss.PFlow.run() # run power flow
                apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
                bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
                apparentpower_database[line_con_num,:] = apparent_power       
                busvoltage_database[line_con_num,:] = bus_voltage 
            except:
                apparentpower_database[line_con_num,:] = np.nan
                busvoltage_database[line_con_num,:] = np.nan
                logging.info('Load flow did not converge in contigency %d' %(line_con_num+1))
                print('Load flow did not converge in contigency %d' %(line_con_num+1))
            ss.Line.u.v[line_con_num] = 1
        dirout = 'output/' # output directory
        foutroot = 'basecase_apparentpower'
        save_database(apparentpower_database, dirout, foutroot)
        foutroot = 'basecase_busvoltage'
        save_database(busvoltage_database, dirout, foutroot)
           
            
    if False: # loading modified case
        
        dirin = 'cases/'
        fnamein = 'caseNPCC.xlsx'
        dirout = 'output/' # output directory
        foutroot = fnamein.replace('.xlsx', '')
        ss = andes.load(fnamein, setup=False)  # please change the path to case file. Note `setup=False`.
        
        bus_shuntdown = 3-1    # generator id =  3
        bus_open = 46-1
        ss.PV.u.v[bus_shuntdown] = 0
        ss.PV.u.v[bus_open] = 1
        
        line_total_num = 234-1
        bus_total_num = 140
        apparentpower_database = np.zeros((line_total_num,line_total_num))
        busvoltage_database = np.zeros((line_total_num,bus_total_num))
        ss.ShuntSw.u.v = np.zeros_like(ss.ShuntSw.u.v).tolist()
        for line_con_num in range(line_total_num):
            ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
            try:
                ss.setup()      # setup system      
                ss.PFlow.run() # run power flow
                apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
                bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
                apparentpower_database[line_con_num,:] = apparent_power
                busvoltage_database[line_con_num,:] = bus_voltage                
            except:
                apparentpower_database[line_con_num,:] = np.nan
                busvoltage_database[line_con_num,:] = np.nan
                logging.info('Load flow did not converge in contigency %d' %(line_con_num+1))
                print('Load flow did not converge in contigency %d' %(line_con_num+1))
            ss.Line.u.v[line_con_num] = 1
            
        #change the system back
        ss.PV.u.v[bus_shuntdown] = 1
        ss.PV.u.v[bus_open] = 0  
        
        dirout = 'output/' # output directory
        foutroot = 'changecase_apparentpower'
        save_database(apparentpower_database, dirout, foutroot)
        foutroot = 'changecase_busvoltage'
        save_database(busvoltage_database, dirout, foutroot)
        
        
        

    if False: # read the information from Matpower
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
        npMx = np.delete(npMx,[0],axis=0)   #delete the base case row
        conver_list = np.asarray(np.where(npMx[:,0]==1))          


        # ss = andes.run(fnamein,input_path=dirin, output_path=dirout)
        # if ss.exit_code == 0: 
        #     save_results(ss, dirout, foutroot)
        # else:
        #     logging.info('Load flow did not converge with all switched shunts in service')

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
        
    # preparing for exit
    logging.shutdown()
    print('end')
