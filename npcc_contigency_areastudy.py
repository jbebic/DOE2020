# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:39:16 2021

@author: txia4@vols.utk.edu


V1.1 TX20210120
Sort the line.

V1.0 TX20210119
Fix the bug.
Compute the Link between two area.


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
from npcc_contigency_test import compute_lineapparentpower
from review_shunt_compensation import output_continuous_heatmap_page
from npcc_contigency_test import save_database

#%% find the link of two area | returen uid
def area_num_detect(ss:andes.system,area1_num, area2_num):

    line_total_num = len(ss.Line.u.v)
    line_area_num_database = np.zeros((line_total_num,2))
    for i in range(line_total_num): 
        # print(i)
        line_area_num_database[i,:] = area_num_for_line(ss,i)
        
    result = np.where((line_area_num_database == (area1_num, area2_num)).all(axis=1)|(line_area_num_database == (area2_num, area1_num)).all(axis=1))
    # result1 = np.where((line_area_num_database == (area1_num, area2_num)).all(axis=1))
    # result2 = np.where((line_area_num_database == (area2_num, area1_num)).all(axis=1))
    return result
#%% find the area the first bus of one line
def area_num_for_line(ss:andes.system,line_num=0):  # start from line 0

    bus1_num = ss.Line.bus1.v[line_num]
    bus2_num = ss.Line.bus2.v[line_num]
    
    area1_num = ss.Bus.area.v[bus1_num-1]     # uid = bus_num(bus name) - 1
    area2_num = ss.Bus.area.v[bus2_num-1]
    area_num = [area1_num,area2_num]
    return area_num
#%% sort the line based on the capacity
def sort_line(line_database,line_capacity_database):  # start from line 0

    line_capacity_database_temp = line_capacity_database[line_database]
    line_database = np.asarray(line_database).T
    temp = np.append(line_database,line_capacity_database_temp,axis=1)
    temp2 = temp[np.lexsort(-temp.T)]

    return temp2

#%% sort the line based on the capacity
def islanding_list(ss:andes.system):  # start from line 0

    toal_line_num = len(ss.Line.uid)-1
    islanding_case = []
    ss.setup()
    for i in range(toal_line_num):
        ss.Line.u.v[i] = 0
        ss.connectivity()
        if ss.Bus.n_islanded_buses:
           islanding_case.append(i)
        ss.Line.u.v[i] = 1
           
    return islanding_case
#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)      

    if True: # loading the npcc system model with shunt capacitors
        
        dirin = 'results/'
        fnamein = 'powerflow_limit.csv'
        dfMx_limit = read_cont_mx1(dirin, fnamein)
        npMx_limit = dfMx_limit.to_numpy()
        npMx_limit = npMx_limit

        fnamein = 'caseNPCC.xlsx'
        ss = andes.load(fnamein, setup=False)  # please change the path to case file. Note `setup=False`.
        connection_line = area_num_detect(ss,1,2)
        temp2 = sort_line(connection_line,npMx_limit)
     
    if True: #basic information study
        
        target_bus  = 39 - 1
        # select_bus = 3 - 1
        select_bus_list = np.array([1,2,3,4,5,6,7,8])-1

        #add a generator and let is work
        
        ss.add('PV', dict(bus=target_bus, p0=0, v0=1))
        ss.setup()      # setup system      
        ss.PFlow.run() # run power flow
        
        PVP = ss.PV.p.v
        PVQ = ss.PV.q.v
        PQP = ss.PQ.Ppf.v
        PQQ = ss.PQ.Qpf.v
        LineP = ss.Line.a1.e
        
        Total_gen = np.sum(PVP[0:8])
        Total_load = np.sum(PQP[0:17])
        Total_input = LineP[42]+LineP[99]
        print('Total Generatorion = %f' %Total_gen)
        print('Total Load = %f' %Total_load)
        print('Total Input = %f' %Total_input)
 
    if True: # here study the 
        
        #database initilization
        line_total_num = 234-1
        bus_total_num = 140
        gen_area_toal_num = select_bus_list.shape[0]
        apparentpower_database = np.zeros((gen_area_toal_num,line_total_num,line_total_num))
        busvoltage_database = np.zeros((gen_area_toal_num,line_total_num,bus_total_num))
        
        # select_gen_num = 0
        for select_gen_num in range(gen_area_toal_num):
            
            select_bus = select_bus_list[select_gen_num]
            factor_temp = 1    # reduce factor <=1
            power_change = ss.PV.p0.v[select_bus] * factor_temp
            ss.PV.u.v[-1] = 1          #activate the new generator
            ss.PV.p0.v[select_bus] = ss.PV.p0.v[select_bus] - power_change
            ss.PV.p0.v[target_bus] = power_change
            
            for line_con_num in range(line_total_num):
                # print('Line contigency = %d' %line_con_num)
                ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
                #ss.conectivtiy
                # use the continue
                ss.connectivity()
                if ss.Bus.n_islanded_buses:
                    logging.info('Contingency %d creates an island - skipping' %(line_con_num+1))
                    print('Contingency %d creates an island - skipping' %(line_con_num+1))
                    apparentpower_database[select_gen_num,line_con_num,:] = np.nan
                    busvoltage_database[select_gen_num,line_con_num,:] = np.nan
                    ss.Line.u.v[line_con_num] = 1
                    continue
                else:
                    try:
                        print('Line contigency = %d' %(line_con_num+1))
                        # ss.setup()      # setup system      
                        ss.PFlow.run() # run power flow
                        apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
                        bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
                        apparentpower_database[select_gen_num,line_con_num,:] = apparent_power       
                        busvoltage_database[select_gen_num,line_con_num,:] = bus_voltage
                    except:
                        # apparentpower_database[line_con_num,:] = np.nan
                        # busvoltage_database[line_con_num,:] = np.nan
                        logging.info('Load flow did not converge in contigency %d' %(line_con_num+1))
                        print('Load flow did not converge in contigency %d' %(line_con_num+1))
                ss.Line.u.v[line_con_num] = 1
            # let the power change to be zero and prepare for the next change of slect bus        
            ss.PV.p0.v[select_bus] = ss.PV.p0.v[select_bus] + power_change
            ss.PV.p0.v[target_bus] = 0
      
    if True:
        dirout = 'output/' # output directory
        foutroot = 'changecase_apparentpower'
        apparentpower_database2 = apparentpower_database.reshape(-1,line_total_num)
        save_database(apparentpower_database2, dirout, foutroot)
        foutroot = 'changecase_busvoltage'
        busvoltage_database2 = busvoltage_database.reshape(-1,bus_total_num)
        save_database(busvoltage_database2, dirout, foutroot)
        
        apparentpower_database3 = apparentpower_database2.reshape(gen_area_toal_num,line_total_num,line_total_num)
        busvoltage_database2 = busvoltage_database.reshape(gen_area_toal_num,line_total_num,bus_total_num)
        
  
    # flatten/reshape for saving data     
    # preparing for exit
    logging.shutdown()
    print('end')
