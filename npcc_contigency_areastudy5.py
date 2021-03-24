# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:39:16 2021

@author: txia4@vols.utk.edu
@author: jzb@achillearesearch.com

v5 (Mar 20-22, 2021) JZB
Corrected  slack generator placement to the actual islanding bus. There
was a one bus offset before. This created a conflict with v0 relative to
the existing islanded PV unit and the cases were not converging. Hantao
figured out the root cause.

Corrected the problem with line loading severity. The line flows were being
divided by the MVA limits (instead of pu limits.) Changed
calculate_line_loading_severity to use just normalized flows as input.

Refactored the code to separate case_utility_functions into their own file.

Revised to remove dependencies on global variables from within the functions.

Added a list of contingencies as an optional input to contingency_analysis and
created two functions to create such lists.

Changed displacement analysis to consider only a subset of flow and voltage
severities based on one sigma difference from the mean value between N-0 displaced
and N-0 base case. Added inputs to check_v_severity and check_flow_severity to
limit the extent of severity calculations.

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
from case_utility_functions import area_generators_indices, \
                                   area_loads_indices, \
                                   area_interface_lines_indices, \
                                   area_import_limits, \
                                   line_idxs_causing_islands, \
                                   generate_contingency_list, \
                                   save_database, \
                                   compute_lineapparentpower

from npcc_powerflow_severity import read_cont_mx1
from review_shunt_compensation import output_continuous_heatmap_page


#%% Retired JZB20210323
##%% find the link of two area | returen uid
# def area_num_detect(ss:andes.system,area1_num, area2_num):
#
#     line_total_num = len(ss.Line.u.v)
#     line_area_num_database = np.zeros((line_total_num,2))
#     for i in range(line_total_num):
#         # print(i)
#         line_area_num_database[i,:] = area_num_for_line(ss,i)
#
#     result = np.where((line_area_num_database == (area1_num, area2_num)).all(axis=1)|(line_area_num_database == (area2_num, area1_num)).all(axis=1))
#     # result1 = np.where((line_area_num_database == (area1_num, area2_num)).all(axis=1))
#     # result2 = np.where((line_area_num_database == (area2_num, area1_num)).all(axis=1))
#     return result
#
# #%% find the area the first bus of one line
# def area_num_for_line(ss:andes.system,line_num=0):  # start from line 0
#
#     bus1_num = ss.Line.bus1.v[line_num]
#     bus2_num = ss.Line.bus2.v[line_num]
#
#     area1_num = ss.Bus.area.v[bus1_num-1]     # uid = bus_num(bus name) - 1
#     area2_num = ss.Bus.area.v[bus2_num-1]
#     area_num = [area1_num,area2_num]
#     return area_num
# #%% sort the line based on the capacity
# def sort_line(line_database,line_capacity_database):  # start from line 0
#
#     line_capacity_database_temp = line_capacity_database[line_database]
#     line_database = np.asarray(line_database).T
#     temp = np.append(line_database,line_capacity_database_temp,axis=1)
#     temp2 = temp[np.lexsort(-temp.T)]
#
#     return temp2
#
# #%% sort the line based on the capacity
# def islanding_list(ss:andes.system):  # start from line 0
#
#     toal_line_num = len(ss.Line.uid)-1
#     islanding_case = []
#     ss.setup()
#     for i in range(toal_line_num):
#         ss.Line.u.v[i] = 0
#         ss.connectivity()
#         if ss.Bus.n_islanded_buses:
#            islanding_case.append(i)
#         ss.Line.u.v[i] = 1
#
#     return islanding_case

#%% contiency analysis
def contingency_analysis(ss:andes.system, contingencies:list): # system model

    line_flows_2D = np.zeros((line_total_num, line_total_num)) # +1 to hold N-0 solution| NO +1 HERE
    bus_voltages_2D = np.zeros((line_total_num, bus_total_num)) # ditto

    for line_con_num in range(line_total_num):
        # stuff np.nan into all contingencies not in the list of contingencies
        if len(contingencies) > 0 and ss.Line.idx.v[line_con_num] not in contingencies:
            line_flows_2D[line_con_num, :] = np.nan
            bus_voltages_2D[line_con_num, :] = np.nan
            continue

        # print('Line contingency = %d' %line_con_num)
        ss.Line.u.v[line_con_num] = 0  # `.v` is the property for values. always use in-place modification `[]`
        ss.connectivity() # look for islands
        # if ss.Bus.island_sets:
        if len(ss.Bus.islands) > 1:
            logging.info('  Contingency on %s creates islands - changing case' %(ss.Line.name.v[line_con_num]))
            print('  Contingency on %s creates islands - changing case' %(ss.Line.name.v[line_con_num]))
            if ss.Slack.bus.v[0] in ss.Bus.islands[0]:
                # add slack bus in islands[1]
                bix = ss.Bus.islands[1][0]
            else:
                #add slack bus in islands[0]
                bix = ss.Bus.islands[0][0]

            newslack_bus_idx = ss.Bus.idx.v[bix]
            logging.info('  Adding slack generator to %s' %(newslack_bus_idx))
            print('  Adding slack generator to %s' %(newslack_bus_idx))

            # ss_temp = andes.load('temp.xlsx', input_path=casedir, setup=False)
            ss_temp = andes.load('temp.xlsx', setup=False)

            ix = [i for i, b in enumerate(ss.PV.bus.v) if b == newslack_bus_idx]
            if len(ix) > 0:
                vref = ss.PV.v0.v[ix[0]]
            else:
                vref = 1.0
            newslack_idx = ss_temp.add('Slack', dict(bus=newslack_bus_idx,
                                                     p0=0,
                                                     v0=vref,
                                                     Vn=230,
                                                     u=1,
                                                     idx=95001,
                                                     name='slack 95001'))
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
                    line_flows_2D[line_con_num,:] = apparent_power
                    bus_voltages_2D[line_con_num,:] = bus_voltage
                else:
                    logging.info('  Load flow did not converge for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                    print('  Load flow did not converge for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                    line_flows_2D[line_con_num, :] = 9999.9 # +1 to account for N-0 solution
                    bus_voltages_2D[line_con_num, :] = 9999.9 # ditto
            except:
                logging.info('  Load flow did not solve for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))
                print('  Load flow did not solve for contingency on %s' %(ss_temp.Line.name.v[line_con_num]))

            # restore the line status
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
def calculate_line_loading_severity(line_flows_norm,
                                    dp_values = np.array([0.8, 1.0, 1.2]),
                                    ks_values = np.array([5, 10, 20])):

    temp = np.ones_like(line_flows_norm)
    line_flows_severity = -temp
    (rows, cols) = line_flows_norm.shape
    for r in range(rows):
        line_flows_severity[r,:]  = calculate_voltage_severity(line_flows_norm[r,:], dp_values, ks_values, vnom=0)

    return line_flows_severity
#%% bus voltage severity
def calculate_bus_voltage_severity(bus_voltage,
                                   dv_values = np.array([0.03, 0.05, 0.07]),
                                   ks_values = np.array([50, 100, 200])  ):

    temp = np.ones_like(bus_voltage)
    bus_voltage_severity = -temp
    (rows, cols) = bus_voltage.shape
    for r in range(rows):
        bus_voltage_severity[r,:]  = calculate_voltage_severity(bus_voltage[r,:], dv_values, ks_values, vnom=1)

    return bus_voltage_severity
#%% bus voltage severity
def displace_generator(ss:andes.system, # system
                       pq_ix, # index of a local substitute pq generator
                       newgen_ix, # index of a new generaotr
                       target_power, # pu value of power of the generator
                       eps = 0.01): # exit precision

    displacement_factor = target_power
    ddf = displacement_factor # delta displacement factor
    p_temp = -ss.PQ.p0.v[pq_ix]
    q_temp = -ss.PQ.q0.v[pq_ix]
    while True:
        ss.PV.p0.v[newgen_ix] = p_temp * displacement_factor
        ss.PQ.p0.v[pq_ix] = -p_temp * (1. - displacement_factor)
        ss.PQ.q0.v[pq_ix] = -q_temp * (1. - displacement_factor)
        ss.PFlow.run() # run power flow
        if ss.PFlow.converged:
            if ddf < eps: break
            ddf = ddf/2
            if displacement_factor + ddf > target_power: break
            displacement_factor += ddf
        else:
            ddf = ddf/2
            displacement_factor -= ddf
            if ddf < 1e-3:
                raise RuntimeError("Failed to displace the generator")

    # update bus voltage magnitudes and angles
    ss.Bus.v0.v[:] = ss.Bus.v.v
    ss.Bus.a0.v[:] = ss.Bus.a.v
    ss.PQ.p0.v[pq_ix] = -p_temp
    ss.PQ.q0.v[pq_ix] = -q_temp
    logging.info('  transferred %f pu power' %(displacement_factor))
    print('  transferred %f pu power' %(displacement_factor))

    return displacement_factor

    # Retired JZB20210323
    # factor_list = np.linspace(0, target_power, 10)

    # displacement_factor_z1 = factor_list[0]
    # for displacement_factor in factor_list:
    #     # if displacement_factor>target_power:
    #     #     break
    #     ppower_change = ss.PV.p0.v[gen_ix] * displacement_factor
    #     qpower_change = ss.PV.q0.v[gen_ix] * displacement_factor
    #     ss.PQ.p0.v[pq_ix] = -(ss.PV.p0.v[gen_ix] - ppower_change) # PQ are loads
    #     ss.PQ.q0.v[pq_ix] = -(ss.PV.q0.v[gen_ix] - qpower_change)
    #     ss.PV.p0.v[newgen_ix] = ppower_change
    #     ss.PV.q0.v[newgen_ix] = qpower_change

    #     # ss.PFlow.init() # helps with the initial guess
    #     ss.PFlow.run() # run power flow

    #     # if it did not converge give up, but return the last value that did converge
    #     if not ss.PFlow.converged:
    #         displacement_factor = displacement_factor_z1
    #         break

    #     #update the initial guess
    #     ss.Bus.v0.v[:] = ss.Bus.v.v
    #     ss.Bus.a0.v[:] = ss.Bus.a.v
    #     displacement_factor_z1 = displacement_factor

    # logging.info('Displaced %f pu power from generator %s' %(displacement_factor, ss.PV.name.v[gen_ix]))
    # print('Displaced %f pu power from generator %s' %(displacement_factor, ss.PV.name.v[gen_ix]))

    # return displacement_factor

#%% check_v_severity
def check_v_severity(bus_voltage_severity, ind_threshold = 9999.9, dv_filt_ix=None, eps=1e-3): # acc_threshold = 99999.99
    check_result = True   # the default value is True means we don't need to worry about it
    if dv_filt_ix is not None:
        max_v_sev = np.nanmax(bus_voltage_severity[:, dv_filt_ix])
    else:
        max_v_sev = np.nanmax(bus_voltage_severity)
    logging.info('  max voltage severity = %g' %max_v_sev)
    print('  max voltage severity = %g' %max_v_sev)

    if max_v_sev-ind_threshold > eps:
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
def check_flow_severity(line_flow_severity, ind_threshold = 9999.9, ds_filt_ix = None, eps=1e-3): #,acc_threshold = 9999.99
    check_result = True   # the default value is True means we don't need to worry about it
    if ds_filt_ix is not None:
        max_s_sev = np.nanmax(line_flow_severity[:, ds_filt_ix])
    else:
        max_s_sev = np.nanmax(line_flow_severity)
    logging.info('  max flow severity = %g ' %max_s_sev)
    print('  max flow severity = %g ' %max_s_sev)
    if max_s_sev-ind_threshold > eps:
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
    # npMx_limit = dfMx_limit.to_numpy()
    # npMx_limit = npMx_limit
    # line_flow_limit = (npMx_limit/100).T
    lines_flow_limits = dfMx_limit['new_limit'].values

    # Read in the case file
    casedir = 'cases/'
    casefile = 'caseNPCC_wAreas.xlsx'
    outdir = 'results/'
    ss = andes.load(casefile, input_path=casedir, setup=False)
    # ss.PFlow.config.max_iter = 50

    # Retired JZB20210320
    # calculate the area input limit as the thermal capability of interface lines
    # connection_line = area_num_detect(ss,1,2)
    # temp2 = sort_line(connection_line, npMx_limit)
    # area_import_limit = sum(temp2[:,1])
    # print('Area import limit: %f MVA' %(area_import_limit))

    #  calculate area import limit by considering N-1 on interface lines
    study_area = 2
    if study_area not in ss.Area.idx.v:
        raise RuntimeError("Specified study area %d is not in the system. Exiting..." %study_area)
    n0_MVA_lim, n1_MVA_lim = area_import_limits(ss, study_area, lines_flow_limits)
    logging.info('Area %d import limits:' %(study_area))
    print('Area %d import limits:' %(study_area))
    logging.info('  N-0: %g' %(n0_MVA_lim))
    print('  N-0: %g' %(n0_MVA_lim))
    logging.info('  N-1: %g MVA' %(n1_MVA_lim))
    print('  N-1: %g MVA' %(n1_MVA_lim))

    # find the slack bus idx
    # slack_bus = ss.Bus.idx.v.index(ss.Slack.bus.v[0])  # uid of slack bus
    if len(ss.Slack.bus.v) > 1:
        raise RuntimeError("There is more than one slack bus in the system. Exiting...")
    slack_bus_idx = ss.Slack.bus.v[0]

    if True:
        # This selects all generators in the specified area (no need for manual lookup)
        area_gens_indices = area_generators_indices(ss, study_area)
        # print(area_gens_indices)
        area_load_indices = area_loads_indices(ss, study_area)
        # print(area_load_indices)

        #add a generator to target bus
        target_bus_idx = 39 # use the bus number as shown on the system map
        if target_bus_idx not in ss.Bus.idx.v:
            raise RuntimeError("Specified target bus %d is not in the system. Exiting..." %target_bus_idx)

        newgen_idx = ss.add('PV', dict(bus=target_bus_idx, # we specify a bus idx, not index
                                       p0=0, v0=1, # initial values for load flow
                                       Vn=230,
                                       u=0, # starting with an added generator disabled
                                       idx=98001) # 98 prefix is used to designate displaced units, and
                           )                      # 001 is the index of a displaced unit.
        newgen_ix = ss.PV.idx.v.index(newgen_idx)

        # Add substitute PQ models to all generators (PV models) in the area.
        # The PQ models are used to gradually move the generator to a target bus.
        # Retired JZB20210323
        # Total_PQ = len(ss.PQ.u.v)
        added_PQ_idxs = []
        for i in area_gens_indices:
            temp = ss.add('PQ', dict(bus=ss.PV.bus.v[i], p0=0, q0=0, Vn=230, u=0))
            added_PQ_idxs.append(temp)

        # check the result
        print(ss.PV.bus.v)
        print(ss.PQ.bus.v)
        print(ss.PQ.u.v)

        ss.setup()      # setup system
        ss.PFlow.run()  # run power flow

        # Retired JZB20210320
        # Collecting info about the area's total generation, load, and imports
        # PVP = ss.PV.p.v
        # PVQ = ss.PV.q.v
        # PQP = ss.PQ.Ppf.v
        # PQQ = ss.PQ.Qpf.v
        # LineP = ss.Line.a1.e

        # Total_gen = np.sum(PVP[area_gens_indices])
        # Total_load = np.sum(PQP[area_load_indices])
        # Total_import = np.sum(LineP[connection_line])

        print('Total system load = %g MW' %(ss.PQ.Ppf.v.sum()*ss.config.mva))
        print('Total output from system generators = %g MW' %(ss.PV.p.v.sum()*ss.config.mva))
        print('Slack generator contribution = %g MW' %(ss.Slack.p.v.sum()*ss.config.mva))
        print('Total system losses = %g MW' %((ss.PV.p.v.sum() +
                                               ss.Slack.p.v.sum() -
                                               ss.PQ.Ppf.v.sum())*ss.config.mva))

        il_ixs, il_signs = area_interface_lines_indices(ss, study_area)
        print('Total import to area %d = %g MW' %(study_area, np.sum(ss.Line.a1.e[il_ixs] * il_signs)))

        #save the data of N-0 starting case
        basecase_vmags = ss.Bus.v.v
        basecase_angs = ss.Bus.a.v

    if True: # the generation displacement "inner loop" "

        # Setting up contingencies of interest
        # Option 1: run all line contingencies
        # contingency_list = []
        # # Option 2: determine which line contingencies cause islands and exclude
        # temp1 = line_idxs_causing_islands(ss)
        # temp2 = set(ss.Line.idx.v) - set(temp1)
        # contingency_list = list(temp2)
        # Option 3: filter by area and exclude island-causing contingencies
        contingency_list = generate_contingency_list(ss, study_area)

        # Initialize numpy data structures to hold results
        line_total_num = len(ss.Line)  # 234-1 # 3 means fake line
        bus_total_num = len(ss.Bus) # 140
        gen_area_total_num = len(area_gens_indices) # select_bus_list.shape[0]
        N1_flows = np.zeros((gen_area_total_num+1, line_total_num+1, line_total_num)) # first +1 to hold the original case;second +1 to hold N-0 solution
        N1_voltages = np.zeros((gen_area_total_num+1, line_total_num+1, bus_total_num)) # ditto
        N1_flow_severities = -np.ones((gen_area_total_num+1, line_total_num+1, line_total_num)) # first +1 to hold the original case;second +1 to hold N-0 solution
        N1_voltage_severities = -np.ones((gen_area_total_num+1, line_total_num+1, bus_total_num)) # ditto
        Target_pu_power = np.ones_like(area_gens_indices, dtype=float)

        # Perform  N-1 contigency analysis of the base case (without displacement)
        ss.PFlow.init() # helps with the initial guess
        ss.PFlow.run() # run power flow
        apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
        bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
        apparent_power_pu = apparent_power/(lines_flow_limits/ss.config.mva)
        flow_sev_mx = calculate_line_loading_severity(apparent_power_pu)
        v_sev_mx  = calculate_bus_voltage_severity(bus_voltage)
        N1_flows[0,0,:] = apparent_power_pu # saving N-0 solution
        N1_voltages[0,0,:] = bus_voltage # ditto
        N1_flow_severities[0,0,:] = flow_sev_mx
        N1_voltage_severities[0,0,:] = v_sev_mx

        # Save the case to file before staritng the contingency analysis
        write(ss, 'temp.xlsx', overwrite = True)

        line_flows_2D, bus_voltages_2D = contingency_analysis(ss, contingency_list)
        line_flows_2D_pu = line_flows_2D/(lines_flow_limits/ss.config.mva)
        # here is the severity for N-1 contingency for original case
        flow_sev_mx = calculate_line_loading_severity(line_flows_2D_pu)
        v_sev_mx  = calculate_bus_voltage_severity(bus_voltages_2D)

        N1_flows[0,1:,:] = line_flows_2D_pu
        N1_voltages[0,1:,:] = bus_voltages_2D
        N1_flow_severities[0,1:,:] = flow_sev_mx
        N1_voltage_severities[0,1:,:] = v_sev_mx

        # Extract the individual max severity for flows and voltages
        # Define severity limits
        basecase_flow_sev_lim = np.nanmax(flow_sev_mx)
        basecase_v_sev_lim = np.nanmax(v_sev_mx)

        logging.info('Basecase max flow severity = %g' %basecase_flow_sev_lim)
        print('Basecase max flow severity = %g' %basecase_flow_sev_lim)
        logging.info('Basecase max bus voltage severity = %g' %basecase_v_sev_lim)
        print('Basecase max bus voltage severity = %g' %basecase_v_sev_lim)

    # Begin the displacment analysis
    if True:
        # Enable the target generaotr
        ss.PV.alter('u', newgen_idx, 1) # enable the target generator (39)

        # Loop over generators
        for ig, gen_ix in enumerate(area_gens_indices):

            # Enable the local PQ unit to gradually shift the output to a target generator
            pq_ix = ss.PQ.uid[added_PQ_idxs[ig]] # find the index of the matching unit
            ss.PQ.alter('u', ss.PQ.idx.v[pq_ix], 1) # enable it
            # Set the p0, q0 output of the replacement unit to match actual p, q from the generator to be displaced
            p_keep = ss.PV.p.v[gen_ix]
            q_keep = ss.PV.q.v[gen_ix]
            ss.PQ.p0.v[pq_ix] = -p_keep
            ss.PQ.q0.v[pq_ix] = -q_keep
            # Disable the generator to be displaced
            ss.PV.alter('u', ss.PV.idx.v[gen_ix], 0)

            logging.info('Displacing unit %s' %(ss.PV.name.v[gen_ix]))
            print('Displacing unit %s' %(ss.PV.name.v[gen_ix]))

            # displace_generator will move up to the specified Target_pu_power from PQ @ pq_ix to PV @ newgen_ix
            Target_pu_power[ig] = displace_generator(ss, pq_ix, newgen_ix, Target_pu_power[ig])

            displaceable_power = Target_pu_power[ig]
            ddp = displaceable_power # ddp = delta_displaceable_power; always > 0

            # save the N-0 case
            apparent_power = compute_lineapparentpower(ss).reshape((1,line_total_num))
            bus_voltage = ss.Bus.v.v.reshape((1,bus_total_num))
            apparent_power_pu = apparent_power/(lines_flow_limits/ss.config.mva)
            flow_sev_mx = calculate_line_loading_severity(apparent_power_pu)
            v_sev_mx  = calculate_bus_voltage_severity(bus_voltage)
            N1_flows[ig+1,0,:] = apparent_power_pu # saving N-0 solution
            N1_voltages[ig+1,0,:] = bus_voltage # ditto
            N1_flow_severities[ig+1,0,:] = flow_sev_mx
            N1_voltage_severities[ig+1,0,:] = v_sev_mx

            write(ss, 'temp.xlsx', overwrite = True) # save the N-0 case with a displaced generator

            # set filters to lines and buses that have actually changed due to displacement
			# ds_d2b means: "delta apparent power, displacement to base". dv_d2b is analogous for voltage
            ds_d2b = N1_flows[ig+1,0,:] - N1_flows[0,0,:]
            dv_d2b = N1_voltages[ig+1,0,:] - N1_voltages[0,0,:]
            ds_alpha = 1.0
            dv_alpha = 1.0
            # ds_filt_ix are the values of column indices that are to be considered in
			# calculate_line_loading_severity. dv_filt_ix is the same for voltage.
			# These values are tuples, so they are used as ds_filt_ix[0]
            ds_filt_ix = np.nonzero(np.abs(ds_d2b - ds_d2b.mean()) > ds_alpha * ds_d2b.std())
            dv_filt_ix = np.nonzero(np.abs(dv_d2b - dv_d2b.mean()) > dv_alpha * dv_d2b.std())
            temp = [ss.Line.name.v[i] for i in ds_filt_ix[0]]
            logging.info('Monitoring lines:')
            logging.info(temp)
            print('Monitoring lines:', temp)

            temp = [ss.Bus.idx.v[i] for i in dv_filt_ix[0]]
            logging.info('Monitoring buses:')
            logging.info(temp)
            print('Monitoring buses:', temp)

            # Display severities on monitored lines and buses.
            logging.info('Basecase N-1 flow severity = %g' %np.nanmax(N1_flow_severities[0,:,ds_filt_ix[0]]))
            print('Basecase N-1 flow severity = %g' %np.nanmax(N1_flow_severities[0,:,ds_filt_ix[0]]))
            logging.info('Basecase N-1 voltage severity = %g' %np.nanmax(N1_voltage_severities[0,:,dv_filt_ix[0]]))
            print('Basecase N-1 voltage severity = %g' %np.nanmax(N1_voltage_severities[0,:,dv_filt_ix[0]]))
            logging.info('Displaced N-0 flow severity = %g' %np.nanmax(flow_sev_mx[0,ds_filt_ix[0]]))
            print('Displaced N-0 flow severity = %g' %np.nanmax(flow_sev_mx[0,ds_filt_ix[0]]))
            logging.info('Displaced N-0 voltage severity = %g' %np.nanmax(v_sev_mx[0,dv_filt_ix[0]]))
            print('Displaced N-0 voltage severity = %g' %np.nanmax(v_sev_mx[0,dv_filt_ix[0]]))

            # Set severity limits here
            # allowing 1.5 line loading
            flow_sev_lim = 5*(1-0.8) + 10*(1.2-1) + 20*(1.5-1.2) # basecase_flow_sev_lim
            # allowing 0.92pu for voltage
            v_sev_lim = (0.95-0.92)*200 + (0.95-0.93)*100 + (0.97-0.95)*50 # basecase_v_sev_lim
            logging.info('Flow severity limit = %g' %flow_sev_lim)
            print('Flow severity limit = %g' %flow_sev_lim)
            logging.info('Voltage severity limit = %g' %v_sev_lim)
            print('Voltage severity limit = %g' %v_sev_lim)

            eps = 0.125  # the threshold of interval halving
            eps2 = 0.06  # the threshold of power reduction
            # N-1 contigency
            while True:
                line_flows_2D, bus_voltages_2D = contingency_analysis(ss, contingency_list)
                line_flows_2D_pu = line_flows_2D/(lines_flow_limits/ss.config.mva)

                flow_sev_mx = calculate_line_loading_severity(line_flows_2D_pu)
                v_sev_mx  = calculate_bus_voltage_severity(bus_voltages_2D)
                # let the power change to be zero and prepare for the next change of slect bus
                check_flow_result = check_flow_severity(flow_sev_mx, flow_sev_lim, ds_filt_ix[0])
                check_v_result = check_v_severity(v_sev_mx, v_sev_lim, dv_filt_ix[0])
                #logging.info('  voltage severity = %g' %(check_flow_result))  # sum of the entire matrix
                #print('Displacing unit %s' %(ss.PV.name.v[ig]))

                if (check_flow_result == False) or (check_v_result == False):
                    if (displaceable_power < eps2):
                        print('  reached the reduction limit, exiting')
                        logging.debug('  reached the reduction limit, exiting')
                        break
                    ddp = ddp/2 # adjust delta displaceable power
                    #can’t exit onon ddp < eps, because we just failed
                    displaceable_power -= ddp
                    print('  reducing power by %g' %(ddp))
                    logging.debug('  reducing power by %g' %(ddp))

                else:
                    ddp = ddp/2 # adjust delta displaceable power
                    if ddp<eps:
                        print('  severity within limits and precision reached, exiting')
                        logging.debug('  severity within limits and precision reached, exiting')
                        break
                    if (displaceable_power >= Target_pu_power[ig]):
                        print('  severity within limits at target power, exiting')
                        logging.debug('  severity within limits at target power, exiting')
                        break
                    displaceable_power += ddp
                    print('  increasing power by %g' %(ddp))
                    logging.debug('  increasing power by %g' %(ddp))

                # Interval halving continues, prepare for the next round
                ss.PQ.p0.v[pq_ix] = -p_keep
                ss.PQ.q0.v[pq_ix] = -q_keep
                ss.PV.p0.v[newgen_ix] = 0
                ss.PV.q0.v[newgen_ix] = 0
                Target_pu_power[ig] = displace_generator(ss, pq_ix, newgen_ix, displaceable_power)
                write(ss, 'temp.xlsx', overwrite = True) # save the N-0 case with displaced generator

            # Save the data after the interval halving
            N1_flows[ig+1,1:,:] = line_flows_2D_pu # +1 to account for N-0 solution
            N1_voltages[ig+1,1:,:] = bus_voltages_2D # ditto
            N1_flow_severities[ig+1,1:,:] = flow_sev_mx
            N1_voltage_severities[ig+1,1:,:] = v_sev_mx

            # Revert the changes to begin analysis of the next generator
            ss.PV.p0.v[newgen_ix] = 0
            ss.PV.q0.v[newgen_ix] = 0
            ss.PV.alter('u', ss.PV.idx.v[gen_ix], 1) # enable PV unit that was being displaced
            ss.PQ.p0.v[pq_ix] = 0 # reset its PQ equivalent
            ss.PQ.q0.v[pq_ix] = 0
            ss.PQ.alter('u', ss.PQ.idx.v[pq_ix], 0)
            # Restore the basecase voltage magnitudes and angles
            ss.Bus.v0.v[:] = basecase_vmags
            ss.Bus.a0.v[:] = basecase_angs

    if True:
        dirout = 'output/' # output directory
        # Retired JZB202103424
        # foutroot = 'changecase_apparentpower'
        # N1_flows2 = N1_flows.reshape(-1,line_total_num)
        # save_database(N1_flows2, dirout, foutroot)
        # foutroot = 'changecase_busvoltage'
        # N1_voltages2 = N1_voltages.reshape(-1,bus_total_num)
        # save_database(N1_voltages2, dirout, foutroot)

        # reshape line flows and bus voltages into 2D data structures
        N1_flows = N1_flows.reshape((gen_area_total_num+1) * (line_total_num+1), line_total_num) # +1 to account for N-0 solution
        N1_voltages = N1_voltages.reshape((gen_area_total_num+1) * (line_total_num+1), bus_total_num) # ditto
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
        # Retired JZB202103424
        # foutroot = 'changecase_apparentpower_severity'
        # N1_flow_severities2 = N1_flow_severities.reshape(-1,line_total_num)
        # save_database(N1_flow_severities2, dirout, foutroot)
        # foutroot = 'changecase_busvoltage_severity'
        # N1_voltage_severities2 = N1_voltage_severities.reshape(-1,bus_total_num)
        # save_database(N1_voltage_severities2, dirout, foutroot)

        # Reshape line flows and bus voltages into 2D data structures
        dirout = 'output/' # output directory
        N1_flows_severity = N1_flow_severities.reshape((gen_area_total_num+1) * (line_total_num+1), line_total_num) # +1 to account for N-0 solution
        N1_voltages_severity = N1_voltage_severities.reshape((gen_area_total_num+1) * (line_total_num+1), bus_total_num) # ditto
        # prepend an index column to the 2d data structures to denote the idx of each moved generator
        gidx = [ss.PV.idx.v[i] for i in area_gens_indices] #  extract idxs of all generators in the area
        gidx = np.hstack(([0],gidx)) # add 0 for the original case
        N1_index = np.repeat(gidx, line_total_num+1).reshape(-1,1) # expand it to match the 2d solution arrays, then turn into a column-vector
        temp = np.hstack((N1_index,N1_flows_severity)) # prepend the index column to line flows
        save_database(temp, dirout, 'line_flow_severities') # save line flows
        temp = np.hstack((N1_index,N1_voltages_severity)) # prepend the index column to bus voltages
        save_database(temp, dirout, 'bus_voltage_severities') # save it

        save_database(Target_pu_power, dirout, 'displaced_pu_powers')

    # preparing for exit
    logging.shutdown()
    print('end')
