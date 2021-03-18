# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:39:16 2021

@author: txia4@vols.utk.edu



"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import matplotlib.backends.backend_pdf as dpdf # pdf output
from andes.io.xlsx import write

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
#%% contiency analysis
def contingency_analysis(ss:andes.system): # system model
    
    line_flows_2D = np.zeros((line_total_num, line_total_num)) # +1 to hold N-0 solution| NO +1 HERE
    bus_voltages_2D = np.zeros((line_total_num, bus_total_num)) # ditto
        
    for line_con_num in range(line_total_num):  
        # print('Line contingency = %d' %line_con_num)
        ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
        ss.connectivity() # look for islands
        # if ss.Bus.island_sets:
        if len(ss.Bus.islands) > 1:
            logging.info('  Contingency on %s creates islands - changing case' %(ss.Line.name.v[line_con_num]))
            # print('  Contingency on %s creates islands - changing case' %(ss.Line.name.v[line_con_num])) 
            if slack_bus in ss.Bus.islands[0]:
                # add slack bus in islands[1]
                bix = ss.Bus.islands[1][0]
            else:
                #add slack bus in islands[0]                        
                bix = ss.Bus.islands[0][0]
                                        
            newslack_bus_idx = ss.Bus.idx.v[bix]
            logging.info('  Adding slack generator to %s' %(newslack_bus_idx))
            # print('  Adding slack generator to %s' %(newslack_bus_idx))
                      
            # ss_temp = andes.load('temp.xlsx', input_path=casedir, setup=False)
            ss_temp = andes.load('temp.xlsx', setup=False)
            newslack_idx = ss_temp.add('Slack', dict(bus=newslack_bus_idx, p0=0, v0=1, u=1, idx=95001,name='slack 95001'))
            ss_temp.setup()
            # make the ss_temp same as the temp expect the 
            ss_temp.Line.u.v[line_con_num] = 0 
            ss_temp.connectivity()
            # u flags are not saved, restoring all u flags to match the ss case
            # ss_temp.PV.u.v[gen_ix] = 0 # disable the unit being displaced
            # ss_temp.PQ.u.v[Total_PQ+ig] = 1 # enable the virtual PQ bus
            # ss_temp.PQ.bus.v[Total_PQ+ig] = ss_temp.PV.bus.v[gen_ix]
            # ss_temp.PV.p0.v[newgen_ix] = ss.PV.p.v[gen_ix] # set initial power output of the added generator equal to power output of the displaced one. 
                   
                    
            try:

                ss_temp.PFlow.run() # run power flow
                if ss_temp.PFlow.converged:
                    apparent_power = compute_lineapparentpower(ss_temp).reshape((1,line_total_num))
                    bus_voltage = ss_temp.Bus.v.v.reshape((1,bus_total_num))
                    line_flows_2D[line_con_num,:] = apparent_power # +1 to account for N-0 solution
                    bus_voltages_2D[line_con_num,:] = bus_voltage # ditto
                else:
                    logging.info('  Load flow did not converge for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                    print('  Load flow did not converge for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                    line_flows_2D[line_con_num, :] = 9999.9 # +1 to account for N-0 solution
                    bus_voltages_2D[line_con_num, :] = 9999.9 # ditto
            except:

                logging.info('  Load flow did not solve for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                print('  Load flow did not solve for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                        
            ss.Line.u.v[line_con_num] = 1
                    
            continue
        else:
            try:
               
                ss.PFlow.run() # run power flow
                if ss.PFlow.converged:
                    apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
                    bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
                    line_flows_2D[line_con_num,:] = apparent_power # +1 to account for N-0 solution
                    bus_voltages_2D[line_con_num,:] = bus_voltage # ditto
                else:
                    logging.info('  Load flow did not converge for contingency on %s' %(ss.Line.name.v[line_con_num]))
                    print('  Load flow did not converge for contingency on %s' %(ss.Line.name.v[line_con_num]))
                    line_flows_2D[line_con_num, :] = 9999.9 # +1 to account for N-0 solution
                    bus_voltages_2D[line_con_num, :] = 9999.9 # ditto
            except:
                
                logging.info('  Load flow did not solve for contingency on %s' %(ss.Line.name.v[line_con_num]))
                print('  Load flow did not solve for contingency on %s' %(ss.Line.name.v[line_con_num]))
                
            ss.Line.u.v[line_con_num] = 1


    return line_flows_2D, bus_voltages_2D

#%% line loading severity
def calculate_line_loading_severity(line_flows,line_flow_limit,dp_values = np.array([0.8, 1.0, 1.25, 1.725]), ks_values = np.array([5, 10, 15, 20])): # system model
    
    temp = np.ones_like(line_flows)
    line_flows_severity = -temp
    line_flows_norm = line_flows/line_flow_limit
    (rows, cols) = line_flows.shape
    for r in range(rows):        
        line_flows_severity[r,:]  = calculate_voltage_severity(line_flows_norm[r,:], dp_values, ks_values, vnom=0)
           
    return line_flows_severity    
#%% bus voltage severity
def calculate_bus_voltage_severity(bus_voltage,dv_values = np.array([0.03, 0.05, 0.08]),ks_values = np.array([5, 10, 15])  ): # system model
     
    temp = np.ones_like(bus_voltage)
    bus_voltage_severity = -temp 
    (rows, cols) = bus_voltage.shape
    for r in range(rows):        
        bus_voltage_severity[r,:]  = calculate_voltage_severity(bus_voltage[r,:], dv_values, ks_values, vnom=1)
           
    return bus_voltage_severity
#%% bus voltage severity
def displace_generator(ss:andes.system, # system
			ig, # generator to displace
			target_bus_idx, # target bus
			target_power = 1): # pu value of power on system base) 
    
    # target_percentage = target_power# Target_percentange_database[0,ig] 
    
    factor_list = np.linspace(0, target_power, 10)
    
    displacement_factor_z1 = factor_list[0]
    for displacement_factor in factor_list:
        
        if displacement_factor>target_power: break        
        ppower_change = ss.PV.p0.v[gen_ix] * displacement_factor
        qpower_change = ss.PV.q0.v[gen_ix] * displacement_factor
        ss.PQ.p0.v[-1] = ss.PV.p0.v[gen_ix] - ppower_change
        ss.PQ.q0.v[-1] = ss.PV.q0.v[gen_ix] - qpower_change
        #change the parameter of new generator
        ss.PV.p0.v[newgen_ix] = ppower_change
        ss.PV.q0.v[newgen_ix] = qpower_change
                
        logging.info('Displacing %f pu power from generator %s' %(ppower_change, ss.PV.name.v[gen_ix]))
        print('Displacing %f pu power from generator %s' %(ppower_change, ss.PV.name.v[gen_ix]))
                
        ss.PFlow.init() # helps with the initial guess
        ss.PFlow.run() # run power flow
        
        # if not converge change the target value
        if not ss.PFlow.converged: 
            target_power = displacement_factor_z1
            break        
               
        #update the intial guess
        ss.Bus.v0.v[:] = ss.Bus.v.v
        ss.Bus.a0.v[:] = ss.Bus.a.v
        displacement_factor_z1 = displacement_factor

    return target_power
#%% check_v_severit
def check_v_severity(bus_voltage_severity,ind_threshold = 99999999.9,acc_threshold = 99999.99): # 
    logging.info('bus voltage severity = %g' %(np.max(bus_voltage_severity)))
    print('bus voltage severity = %g' %(np.max(bus_voltage_severity))) 
    check_result = True   # the default value is True means we don't need to worry about it
    # It checks mulative and and individual values
    if np.max(bus_voltage_severity)-ind_threshold>0.1:
        check_result = False   
    
    # severity_matrix = bus_voltage_severity
    # weight_vector = np.matlib.ones((severity_matrix.shape[1],1))
    # severity_vector = np.matlib.zeros((severity_matrix.shape[0],1))
    # for i in range(severity_matrix.shape[0]):
    #     temp_dot = np.dot(severity_matrix[i,:],weight_vector)
    #     severity_vector[i,0] = np.sum(temp_dot)
       
    # if np.max(severity_vector)>=acc_threshold:
    #     check_result = False   

    return check_result # True if severity acceptable, False if not.

#%% check_v_severit
def check_flow_severity(line_flow_severity,ind_threshold = 99999999.9,acc_threshold = 99999.99): # 
    logging.info('flow severity = %g ' %(np.max(line_flow_severity)))
    print('flow severity = %g ' %(np.max(line_flow_severity))) 
    check_result = True   # the default value is True means we don't need to worry about it
    # It checks mulative and and individual values
    if np.max(line_flow_severity)-ind_threshold>0.1:
        check_result = False   
    
    # severity_matrix = line_flow_severity
    # weight_vector = np.matlib.ones((severity_matrix.shape[1],1))
    # severity_vector = np.matlib.zeros((severity_matrix.shape[0],1))
    # for i in range(severity_matrix.shape[0]):
    #     temp_dot = np.dot(severity_matrix[i,:],weight_vector)
    #     severity_vector[i,0] = np.sum(temp_dot)
    
    # if np.max(severity_vector)>=acc_threshold:
    #     check_result = False   

    return check_result # True if severity acceptable, False if not.

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
    line_flow_limit = (npMx_limit/100).T

    # Read in the case file
    casedir = 'cases/'
    casefile = 'caseNPCC_wAreas.xlsx'
    outdir = 'results/'
    ss = andes.load(casefile, input_path=casedir, setup=False)
    ## calculate the area input limit as the thermal capability of interface lines
    connection_line = area_num_detect(ss,1,2)
    temp2 = sort_line(connection_line, npMx_limit)
    area_import_limit = sum(temp2[:,1])
    print('Area import limit: %f MVA' %(area_import_limit))
    
    ss.PFlow.config.max_iter = 100
    # find the slack bus idx
    slack_bus = ss.Bus.idx.v.index(ss.Slack.bus.v[0])  # uid of slack bus
    
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
        Total_PQ = len(ss.PQ.u.v)           # w
        for i in area_gens_indices:
            ss.add('PQ', dict(bus=ss.PV.bus.v[i], p0=0, v0=1,u=0))
            
        # check the result
        print(ss.PV.bus.v)
        print(ss.PQ.bus.v)
        print(ss.PQ.u.v)
        
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
        line_total_num = len(ss.Line)  # 234-1 # 3 means fake line
        bus_total_num = len(ss.Bus) # 140
        gen_area_total_num = len(area_gens_indices) # select_bus_list.shape[0]
        N1_apparentpower_database = np.zeros((gen_area_total_num+1, line_total_num+1, line_total_num)) #first +1 to hold the original case;second +1 to hold N-0 solution
        N1_busvoltage_database = np.zeros((gen_area_total_num+1, line_total_num+1, bus_total_num)) # ditto
        N1_flow_severity_database = np.zeros((gen_area_total_num+1, line_total_num+1, line_total_num)) #first +1 to hold the original case;second +1 to hold N-0 solution
        N1_voltage_severity_database = np.zeros((gen_area_total_num+1, line_total_num+1, bus_total_num)) # ditto
        Target_percentange_database = np.ones_like(area_gens_indices)
        
        
        # here is the N-1 contigency study for the base original case without displacement
        ss.PFlow.init() # helps with the initial guess
        ss.PFlow.run() # run power flow
        apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
        bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
        N1_apparentpower_database[0,0,:] = apparent_power # saving N-0 solution
        N1_busvoltage_database[0,0,:] = bus_voltage # ditto
        
        write(ss, 'temp.xlsx',overwrite = True) # copy the original case
        # gen_ix = 0
        # ig = 0
        if True: # debug (todo)
            line_flows_2D, bus_voltages_2D = contingency_analysis(ss)
        
        # here is the severity for N-1 contingency for original case
       
            flow_sev_mx = calculate_line_loading_severity(line_flows_2D,line_flow_limit)
            v_sev_mx  = calculate_bus_voltage_severity(bus_voltages_2D)
        # extract the individual max severity for voltage and flow    
            ind_voltage_severity_ref = np.max( v_sev_mx )
            ind_flow_severity_ref = np.max( flow_sev_mx )
        logging.info('max flow severity = %g' %(ind_flow_severity_ref))
        print('max flow severity = %g' %(ind_flow_severity_ref) )
        logging.info('max bus voltage severity %g = ' %(ind_voltage_severity_ref))
        print('max bus voltage severity %g =' %(ind_voltage_severity_ref) )
        # here is the displacment case       
    if True:
        
        for ig, gen_ix in enumerate(area_gens_indices):
            
            # first loop: check the convergency
           
            # ss.PV.u.v[gen_ix] = 0 # disable the unit being displaced
            ss.PV.alter('u',ss.PV.idx.v[gen_ix],0)
            # ss.PQ.u.v[Total_PQ+ig] = 1 # enable the virtual PQ bus
            ss.PQ.alter('u',ss.PQ.idx.v[Total_PQ+ig],1)
            ss.PV.alter('u',98001,1) # enable the target generator (39)
            # ss.PQ.bus.v[Total_PQ+ig] = ss.PV.bus.v[gen_ix]
            
            logging.info('Displacing unit %s' %(ss.PV.name.v[ig]))
            print('Displacing unit %s' %(ss.PV.name.v[ig])) 
           
            # target_power  = displace_generator(ss, ig,target_bus_idx, 1)  #Target_percentange_database
            target_power  = displace_generator(ss, ig,target_bus_idx, Target_percentange_database[ig])  #
            Target_percentange_database[ig] = target_power
            displaceable_power = target_power
            ddp = displaceable_power # ddp = delta_displaceable_power; always >
              
            
            # save the N-0 case
            apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
            bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
            N1_apparentpower_database[ig+1,0,:] = apparent_power # saving N-0 solution
            N1_busvoltage_database[ig+1,0,:] = bus_voltage # ditto
            
            write(ss, 'temp.xlsx',overwrite = True) # save the N-0 case with displace generator

            ind_power_severity_threshold = ind_flow_severity_ref*1
            ind_voltage_severity_threshold = ind_voltage_severity_ref*1

            eps = 0.125  # the threshold of interval halving
            eps2 = 0.06  # the threshold of power reduction
           # N-1 contigency
            while True:          
                line_flows_2D, bus_voltages_2D = contingency_analysis(ss)
                
                flow_sev_mx = calculate_line_loading_severity(line_flows_2D,line_flow_limit)
                v_sev_mx  = calculate_bus_voltage_severity(bus_voltages_2D)
                # let the power change to be zero and prepare for the next change of slect bus 
                check_flow_result = check_flow_severity(flow_sev_mx,ind_power_severity_threshold)
                check_v_result = check_v_severity(v_sev_mx,ind_voltage_severity_threshold)
                #logging.info('  voltage severity = %g' %(check_flow_result))  # sum of the entire matrix
                #print('Displacing unit %s' %(ss.PV.name.v[ig])) 
                             
                if (check_flow_result == False) or (check_v_result == False):
                    if (displaceable_power < eps2): 
                        print('  reach the reduction limit, exiting')
                        logging.debug('  reach the reduction limit, exiting')
                        break
                    ddp = ddp/2 # adjust delta displaceable power
                    #can’t exit onon ddp < eps, because we just failed
                    displaceable_power -= ddp
                    print('  reducing power by %g' %(ddp))
                    logging.debug('  reducing power by %g' %(ddp))

                else:
                    ddp = ddp/2 # adjust delta displaceable power				
                    if ddp<eps: 
                        print('  precision reached, exiting')
                        logging.debug('  precision reached, exiting')
                        break
                    if (displaceable_power + ddp > target_power): 
                        print('  success on the first try, exiting')
                        logging.debug('  success on the first try, exiting')
                        break
                    displaceable_power += ddp
                    print('  increasing power by %g' %(ddp))
                    logging.debug('  increasing power by %g' %(ddp))
                
                #undoing the last displacement
                ss.PQ.p0.v[-1] = 0
                ss.PQ.q0.v[-1] = 0
                ss.PV.p0.v[gen_ix] = ss.PV.p0.v[newgen_ix]
                ss.PV.q0.v[gen_ix] = ss.PV.q0.v[newgen_ix]
                ss.PV.p0.v[newgen_ix] = 0
                ss.PV.q0.v[newgen_ix] = 0
                Target_percentange_database[ig] = displace_generator(ss, ig,target_bus_idx,  displaceable_power)
                write(ss, 'temp.xlsx',overwrite = True) # save the N-0 case with displace generator

            # save the data after the interval halving
            N1_apparentpower_database[ig+1,1:,:] = line_flows_2D # +1 to account for N-0 solution
            N1_busvoltage_database[ig+1,1:,:] = bus_voltages_2D # ditto
            N1_flow_severity_database[ig+1,1:,:] = flow_sev_mx
            N1_voltage_severity_database[ig+1,1:,:] = v_sev_mx 
                
            # ss.PV.u.v[gen_ix] = 1 # re-enable the original unit
            # ss.PV.u.v[newgen_ix] = 0 # disable the replacement unit
            ss.PV.alter('u',ss.PV.idx.v[gen_ix],1)
            ss.PQ.alter('u',ss.PQ.idx.v[Total_PQ+ig],0)
            #restore the intial guess
            ss.Bus.v0.v[:] = StartingCase_BusV
            ss.Bus.a0.v[:] = StartingCase_BusA
      
    if False:
        dirout = 'output/' # output directory
        foutroot = 'changecase_apparentpower'
        N1_apparentpower_database2 = N1_apparentpower_database.reshape(-1,line_total_num)
        save_database(N1_apparentpower_database2, dirout, foutroot)
        foutroot = 'changecase_busvoltage'
        N1_busvoltage_database2 = N1_busvoltage_database.reshape(-1,bus_total_num)
        save_database(N1_busvoltage_database2, dirout, foutroot)
        
        # reshape line flows and bus voltages into 2D data structures
        N1_flows = N1_apparentpower_database.reshape((gen_area_total_num+1) * (line_total_num+1), line_total_num) # +1 to account for N-0 solution
        N1_voltages = N1_busvoltage_database.reshape((gen_area_total_num+1) * (line_total_num+1), bus_total_num) # ditto
        # prepend an index column to the 2d data structures to denote the idx of each moved generator
        gidx = [ss.PV.idx.v[i] for i in area_gens_indices] #  extract idxs of all generators in the area
        gidx = np.hstack(([0],gidx)) # add 0 for the original case
        N1_index = np.repeat(gidx, line_total_num+1).reshape(-1,1) # expand it to match the 2d solution arrays, then turn into a column-vector
        temp = np.hstack((N1_index,N1_flows)) # prepend the index column to line flows
        save_database(temp, dirout, 'line_flows') # save line flows
        temp = np.hstack((N1_index,N1_voltages)) # prepend the index column to bus voltages 
        save_database(temp, dirout, 'bus_voltages') # save it
    
        #severity version
        dirout = 'output/' # output directory
        foutroot = 'changecase_apparentpower_severity'
        N1_flow_severity_database2 = N1_flow_severity_database.reshape(-1,line_total_num)
        save_database(N1_flow_severity_database2, dirout, foutroot)
        foutroot = 'changecase_busvoltage_severity'
        N1_voltage_severity_database2 = N1_voltage_severity_database.reshape(-1,bus_total_num)
        save_database(N1_voltage_severity_database2, dirout, foutroot)
        
        # reshape line flows and bus voltages into 2D data structures
        N1_flows_severity = N1_flow_severity_database.reshape((gen_area_total_num+1) * (line_total_num+1), line_total_num) # +1 to account for N-0 solution
        N1_voltages_severity = N1_voltage_severity_database.reshape((gen_area_total_num+1) * (line_total_num+1), bus_total_num) # ditto
        # prepend an index column to the 2d data structures to denote the idx of each moved generator
        gidx = [ss.PV.idx.v[i] for i in area_gens_indices] #  extract idxs of all generators in the area
        gidx = np.hstack(([0],gidx)) # add 0 for the original case
        N1_index = np.repeat(gidx, line_total_num+1).reshape(-1,1) # expand it to match the 2d solution arrays, then turn into a column-vector
        temp = np.hstack((N1_index,N1_flows_severity)) # prepend the index column to line flows
        save_database(temp, dirout, 'line_flows_severity') # save line flows
        temp = np.hstack((N1_index,N1_voltages_severity)) # prepend the index column to bus voltages 
        save_database(temp, dirout, 'bus_voltages_severity') # save it
    
    
    
    
    
    

    # preparing for exit
    logging.shutdown()
    print('end')
