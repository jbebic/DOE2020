# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 08:36:42 2021
@author: jzb@achillearesearch.com

vX.Y JZBYYYYMMDD
As we make changes the latest revisions are kept on top.

v0.1 JZB20210320
Utility functions for loading, saving, and topology processing of an ANDES system.

"""

import logging
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # plotting

# from datetime import datetime # time stamps
import os # operating system interface

import andes
# from andes.core.var import BaseVar, Algeb, ExtAlgeb
# from andes.io.xlsx import write

#%% 
def load_case(dirin:str, fnamein:str, setup=False):
    # ss = andes.load(os.path.join(dirin, fnamein))
    ss = andes.load(fnamein, input_path=dirin, setup=False)
    logging.info("Loaded '%s'" %fnamein)
    logging.info("The case has %d areas" %len(ss.Area))
    for anum, aname, buses in zip(ss.Area.idx.v, ss.Area.name.v, ss.Area.Bus.v):
        logging.info("  Area %d: '%s', with %d buses" %(anum, aname, len(buses)))
    return ss

#%% Read the indexed result file
def load_displacement_analysis_results(dirin:str, fname:str):
    df1 = pd.read_csv(os.path.join(dirin, fname),
                      header = None)
    return df1

#%% save_results
def save_database(input_database, dirout:str, foutroot:str, cnames=None):
    logging.debug('saving ANDES data')
    # create a dataframe from a 2D input array
    df = pd.DataFrame(input_database)
    # prepare the filename
    fout = os.path.join(dirout,foutroot + '.csv')
    # Save with column names if cnames argument was specified, or "headless" otherwise
    if cnames is None:    
        df.to_csv(fout, index=False, header=False)
    else:
        df.to_csv(fout, index=False, header=cnames)
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


#%% retired
# def area_index(ss:andes.System, area_idx):
#     ix = [i for i, idx in enumerate(ss.Area.idx.v) if idx == area_idx][0]
#     return ix

#%%
def area_buses_indices(ss:andes.System, area_idx):
    bix = [i for i, area in enumerate(ss.Bus.area.v) if area == area_idx]
    return bix

#%% 
def area_interface_lines_indices(ss:andes.System, area_idx):
    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    # find all area's interface lines as those lines with either bus1 or bus2 in the area, but not both bus1 and bus2 in the area.
    lix = [i for i, buses in enumerate(zip(ss.Line.bus1.v, ss.Line.bus2.v)) \
                              if (buses[0] in bidx or buses[1] in bidx) and not \
                                 (buses[0] in bidx and buses[1] in bidx)]
    signs = [-1 if ss.Line.bus1.v[i] in bidx else 1 for i in lix]
    return lix, signs

#%%
def area_generators_indices(ss:andes.System, area_idx):
    # find idxs of all buses in the area
    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    # find all area's interface lines as those lines with either bus1 or bus2 in the area, but not both bus1 and bus2 in the area.
    gix = [i for i, bus in enumerate(ss.PV.bus.v) if bus in bidx]
    return gix

#%% retired
# def pq_gens (ss:andes.System, gens):
#     p = [ss.PV.p.v[i] for i in gens]
#     q = [ss.PV.q.v[i] for i in gens]
#     return p, q

#%% 
def area_loads_indices(ss:andes.System, area_idx):
    # find idxs of all buses in the area
    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    # find all area's interface lines as those lines with either bus1 or bus2 in the area, but not both bus1 and bus2 in the area.
    pqix = [i for i, bus in enumerate(ss.PQ.bus.v) if bus in bidx]
    return pqix

#%%
def area_import_limits(ss:andes.System, area_idx, lines_flow_limits):
    il_ixs, _ = area_interface_lines_indices(ss, area_idx) # interface lines indices
    il_flow_lims = lines_flow_limits[il_ixs] # interface lines flow limits
    # N-0
    n0_MVA_lim = il_flow_lims.sum()
    n1_MVA_arr = np.array([n0_MVA_lim-mva for mva in il_flow_lims])
    n1_MVA_lim = n1_MVA_arr.min()
    return n0_MVA_lim, n1_MVA_lim

#%%
def line_idxs_causing_islands(ss:andes.System):
    logging.debug('Screening for line contingencies that result in islands')
    print('Screening for line contingencies that result in islands')
    lidxs = [] # contingencies causing islands
    for lix in range(len(ss.Line)):
        ss.Line.u.v[lix] = 0
        ss.connectivity(info=False)
        if len(ss.Bus.islands) > 1:
            lidx = ss.Line.idx.v[lix]
            bus1 = ss.Line.bus1.v[lix]
            bus2 = ss.Line.bus2.v[lix]
            lidxs.append(lidx)
            logging.debug('  Contingency on %s from %s to %s causes an island' %(lidx, bus1, bus2))
            print('  Contingency on %s from %s to %s causes an island' %(lidx, bus1, bus2))
        ss.Line.u.v[lix] = 1
    return lidxs

def generate_contingency_list(ss:andes.System, area_idx):
    # determine which line contingencies cause islands and exclude 
    temp1 = line_idxs_causing_islands(ss)
    temp2 = set(ss.Line.idx.v) - set(temp1)
    contingency_list = list(temp2)

    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    lix = [i for i, buses in enumerate(zip(ss.Line.bus1.v, ss.Line.bus2.v)) \
                              if (buses[0] in bidx or buses[1] in bidx)]
    lidx = [ss.Line.idx.v[i] for i in lix if ss.Line.idx.v[i] in contingency_list]
    return lidx
    