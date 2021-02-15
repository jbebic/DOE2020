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
from reconfigure_case import area_generators_indices,area_loads_indices
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

    # Read in the line loading limits
    datadir = 'results/'
    datafile = 'line_loading_limits.csv'
    dfMx_limit = read_cont_mx1(datadir, datafile)
    npMx_limit = dfMx_limit.to_numpy()
    npMx_limit = npMx_limit

    # Read in the case file
    casedir = 'cases/'
    casefile = 'caseNPCC_wAreas.xlsx'
    outdir = 'results/'
    ss = andes.load(casefile, input_path=casedir, setup=False)
    connection_line = area_num_detect(ss,1,2)
    temp2 = sort_line(connection_line, npMx_limit)
    area_import_limit = sum(temp2[:,1])
    print('Area import limit: %f MVA' %(area_import_limit))
    
    ss.PFlow.config.max_iter = 100
    
    if True: 
        
        # This selects all generators in the specified area (no need for manual lookup)
        area_idx = 2
        area_gens_indices = area_generators_indices(ss, area_idx)
        print(area_gens_indices)
        area_load_indices = area_loads_indices(ss, area_idx)
        print(area_load_indices)

        #add a generator to target bus
        target_bus_idx = 39 # use the bus number as shown on the system map
        
        newgen_idx = ss.add('PV', dict(bus=target_bus_idx, # we specify a bus idx, not index
                                       p0=0, v0=1, # initial values for load flow
                                       u=0, # starting out with an added generator disabled
                                       idx=98001) # 98 prefix is used to designate displaced units, and
                           )                      # 001 is the index of a displaced unit.
        newgen_ix = ss.PV.idx.v.index(newgen_idx)
        
        # add a moving PQ bus
        ss.add('PQ', dict(bus=10, p0=0.1, v0=1))
        ss.PQ.u.v[-1] = 0 # set the last element to be 0
        
        ss.setup()      # setup system      
        ss.PFlow.run()  # run power flow

        # Collecting info about the area's total generation, load, and imports
        PVP = ss.PV.p.v
        PVQ = ss.PV.q.v
        PQP = ss.PQ.Ppf.v
        PQQ = ss.PQ.Qpf.v
        LineP = ss.Line.a1.e
        
        Total_gen = np.sum(PVP[area_gens_indices])
        Total_load = np.sum(PQP[area_load_indices])
        Total_import = np.sum(LineP[connection_line])
        print('Total Generation = %f' %Total_gen)
        print('Total Load = %f' %Total_load)
        print('Total Import = %f' %Total_import)
        
        #save the data of N-0 starting case
        StartingCase_BusV = ss.Bus.v.v
        StartingCase_BusA = ss.Bus.a.v
 
    if True: # the generation displacement "inner loop" "
        
        
        # database initialization
        line_total_num = len(ss.Line) # 234-1
        bus_total_num = len(ss.Bus) # 140
        gen_area_total_num = len(area_gens_indices) # select_bus_list.shape[0]
        N1_apparentpower_database = np.zeros((gen_area_total_num, line_total_num, line_total_num))
        N1_busvoltage_database = np.zeros((gen_area_total_num, line_total_num, bus_total_num))
     
        
        for ig, gen_ix in enumerate(area_gens_indices):
            
            # first loop: check the convergency
            factor_list = np.arange(0.1,1.1,0.1)
          
            ss.PV.u.v[gen_ix] = 0 # disable the unit being displaced
            ss.PQ.u.v[-1] = 1 # enable the virtual PQ bus
            ss.PQ.bus.v[-1] = ss.PV.bus.v[gen_ix]
            
            for displacement_factor in factor_list:
                
                ppower_change = ss.PV.p0.v[gen_ix] * displacement_factor
                qpower_change = ss.PV.q0.v[gen_ix] * displacement_factor
                ss.PQ.p0.v[-1] = ss.PV.p0.v[gen_ix] - ppower_change
                ss.PQ.q0.v[-1] = ss.PV.q0.v[gen_ix] - qpower_change
                #change the parameter of new generator
                ss.PV.p0.v[newgen_ix] = ppower_change
                ss.PV.q0.v[newgen_ix] = qpower_change
               
                print('Displacing %f pu power from generator %s' %(ppower_change, ss.PV.name.v[gen_ix]))
                
                ss.PFlow.init() # helps with the initial guess
                ss.PFlow.run() # run power flow
                
                logging.info('Displacing unit %s' %(ss.PV.name.v[ig]))
                
                #update the intial guess
                ss.Bus.v0.v[:] = ss.Bus.v.v
                ss.Bus.a0.v[:] = ss.Bus.a.v
                

           
            # Replacing power displacement by a complete unit shutdown
            # displacement_factor = 1    # reduce factor <=1
            
            ss.PV.u.v[gen_ix] = 0 # disable the unit being displaced
            ss.PQ.u.v[-1] = 0 # disable the virtual PQ bus 
            ss.PV.u.v[newgen_ix] = 1 # enable the substitute unit. Already enabled. Do it again for safe.
            ss.PV.p0.v[newgen_ix] = ss.PV.p.v[gen_ix] # set initial power output of the added generator equal to power output of the displaced one. 
            print('Displacing unit %s' %(ss.PV.name.v[ig]))     
            
            # we do not need to change the intial guess here becuase it is not a hard copy
            # ss.PFlow.init()
            # ss.PFlow.run() # run power flow
            N0_apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
            N0_bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
            
            for line_con_num in range(line_total_num):
                # print('Line contigency = %d' %line_con_num)
                ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
                ss.connectivity() # look for islands
                # if ss.Bus.island_sets:
                if len(ss.Bus.islands) > 1:
                    logging.info('  Contingency on %s creates islands - skipping' %(ss.Line.name.v[line_con_num]))
                    print('  Contingency on %s creates islands - skipping' %(ss.Line.name.v[line_con_num]))
                    N1_apparentpower_database[ig, line_con_num, :] = np.nan
                    N1_busvoltage_database[ig, line_con_num, :] = np.nan
                    ss.Line.u.v[line_con_num] = 1
                    continue
                else:
                    try:
                        # print('Line contigency = %d' %(line_con_num+1))
                        # ss.setup()      # setup system
                        ss.PFlow.init() # helps with the initial guess
                        ss.PFlow.run() # run power flow
                        if ss.PFlow.converged:
                            apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
                            bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
                            N1_apparentpower_database[ig,line_con_num,:] = apparent_power
                            N1_busvoltage_database[ig,line_con_num,:] = bus_voltage
                        else:
                            logging.info('  Load flow did not converge for contigency on %s' %(ss.Line.name.v[line_con_num]))
                            print('  Load flow did not converge for contigency on %s' %(ss.Line.name.v[line_con_num]))
                            N1_apparentpower_database[ig, line_con_num, :] = np.nan
                            N1_busvoltage_database[ig, line_con_num, :] = np.nan
                    except:
                        # apparentpower_database[line_con_num,:] = np.nan
                        # busvoltage_database[line_con_num,:] = np.nan
                        logging.info('  Load flow did not solve for contigency on %s' %(ss.Line.name.v[line_con_num]))
                        print('  Load flow did not solve for contigency on %s' %(ss.Line.name.v[line_con_num]))
                ss.Line.u.v[line_con_num] = 1
            # let the power change to be zero and prepare for the next change of slect bus        
           
            
            ss.PV.u.v[gen_ix] = 1 # re-enable the original unit
            ss.PV.u.v[newgen_ix] = 0 # disable the replacement unit
            #restore the intial guess
            ss.Bus.v0.v[:] = StartingCase_BusV
            ss.Bus.a0.v[:] = StartingCase_BusA
      
    if True:
        dirout = 'output/' # output directory
        foutroot = 'changecase_apparentpower'
        N1_apparentpower_database2 = N1_apparentpower_database.reshape(-1,line_total_num)
        save_database(N1_apparentpower_database2, dirout, foutroot)
        foutroot = 'changecase_busvoltage'
        N1_busvoltage_database2 = N1_busvoltage_database.reshape(-1,bus_total_num)
        save_database(N1_busvoltage_database2, dirout, foutroot)
        
        N1_apparentpower_database3 = N1_apparentpower_database2.reshape(gen_area_total_num,line_total_num,line_total_num)
        N1_busvoltage_database2 = N1_busvoltage_database.reshape(gen_area_total_num,line_total_num,bus_total_num)
    
        
        
  
    # flatten/reshape for saving data     
    # preparing for exit
    logging.shutdown()
    print('end')
