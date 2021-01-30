# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:42:12 2021
@author: jzb@achillearesearch.com

vX.Y JZBYYYYMMDD
As we make changes the latest revisions are kept on top.

v0.1 JZB20210127
Importing the npcc case, identifying the interface line to an area, and 
reconfiguring the case to replace each interface line with two line segments, 
each terminating into a PV source. For illustration, let an interface line be 
connected between buses "s" and "r". The reconfiguration includes the following 
steps:
    1) Selecting a location along the line. Let it be represented by a factor 
       k, measured as a normalized distance from the sending end. 
    2) Adding two new buses to the case, let them be labeled "1" and "2".
    3) Adding two new line segments: one from bus "s" to bus "1", and another 
       from bus "2" to bus "r". The parameters of the line segents are derived 
       from the parameters of the original line. Series rectance xl is divided 
       into two parts: k*xl and (1-k)*xl, allocated to the segments "s" to "1", 
       and "2" to "r", respectively. Analogous arithemtic is used to split rl, 
       bl and gl.
    4) Adding two new generators, one at bus "1" and another at bus "2".
       The generators must be constrained by active power. Let P1 be the active 
       power delivered to bus "1" from the s-to-1, line segment and P2 be 
       the active power injected into 2-to-r line segment from bus "2". 
       Assuming a lossless power flow controller, a constraint P1=P2 must be 
       enforced in all operating points. 
       This is being done here by solving the power flow of the system and the 
       compensated lines separately-but-synchronously. Such setup allows 
       accurate enforcement of equipment constraints for two-converter 
       compensators (such as a UPFC.)
    5) Adding a third option with a line segment from bus "1" to bus "r". 
       This will be used to solve the load flow on an uncompensated line.
      
This file deals with the reconfiguration of an input case to add new elements 
per steps 1-5 above, and illustrates the separate-but-synchronous solution of 
the compensated lines and the system. 

"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb
from andes.io.xlsx import write

# from vectorized_severity import calculate_voltage_severity

#%% 
def load_case(dirin:str, fnamein:str, setup=False):
    # ss = andes.load(os.path.join(dirin, fnamein))
    ss = andes.load(fnamein, input_path=dirin, setup=False)
    logging.info("Loaded '%s'" %fnamein)
    logging.info("The case has %d areas" %len(ss.Area))
    for anum, aname, buses in zip(ss.Area.idx.v, ss.Area.name.v, ss.Area.Bus.v):
        logging.info("  Area %d: '%s', with %d buses" %(anum, aname, len(buses)))
    return ss

#%%
def area_index(ss:andes.System, area_idx):
    ix = [i for i, idx in enumerate(ss.Area.idx.v) if idx == area_idx][0]
    return ix

#%%
def area_buses_indices(ss:andes.System, area_idx):
    bix = [i for i, area in enumerate(ss.Bus.area.v) if area == area_idx]
    return bix

#%% 
def area_interface_lines_indices(ss:andes.System, area_idx):
    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    # find all area's interface lines as those lines with either bus1 or bus2 in the area, but not both bus1 and bus2 in the area.
    lidx = [i for i, buses in enumerate(zip(ss.Line.bus1.v, ss.Line.bus2.v)) \
                              if (buses[0] in bidx or buses[1] in bidx) and not (buses[0] in bidx and buses[1] in bidx)]
    return lidx

def area_generators_indices(ss:andes.System, area_idx):
    # find idxs of all buses in the area
    bidx = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    # find all area's interface lines as those lines with either bus1 or bus2 in the area, but not both bus1 and bus2 in the area.
    gix = [i for i, bus in enumerate(ss.PV.bus.v) if bus in bidx]
    return gix

def pq_gens (ss:andes.System, gens):
    p = [ss.PV.p.v[i] for i in gens]
    q = [ss.PV.q.v[i] for i in gens]
    return p, q

#%%
def add_gpfcs(ss:andes.System, interface_lines, gpfc_locations, area_idx, import_direction=True):
    abidxs = [ss.Bus.idx.v[i] for i in area_buses_indices(ss, area_idx)]
    for n, (line, k) in enumerate(zip(interface_lines, gpfc_locations)):
        # logging.info("Adding flow control to: '%s'" %ss.Line.name.v[line])
        b1idx = ss.Line.bus1.v[line]
        b2idx = ss.Line.bus2.v[line]
        # reconciling the controller's point of instalation
        if import_direction:
            if b2idx in abidxs:
                bus_s_idx = b1idx
                bus_r_idx = b2idx
            else:
                bus_s_idx = b2idx
                bus_r_idx = b1idx
        else: # export
            if b1idx in abidxs:
                bus_s_idx = b1idx
                bus_r_idx = b2idx
            else:
                bus_s_idx = b2idx
                bus_r_idx = b1idx
        logging.info("Adding a GPFC to '%s', at %g distance from bus %d" %(ss.Line.name.v[line], k, bus_s_idx))
        # Retrieve line variables
        xsr = ss.Line.x.v[line]
        rsr = ss.Line.r.v[line]
        bsr = ss.Line.b1.v[line] + ss.Line.b2.v[line]
        gsr = ss.Line.g1.v[line] + ss.Line.g2.v[line]
        Vns = ss.Line.Vn1.v[line]
        Vnr = ss.Line.Vn2.v[line]

        if Vns != Vnr:
            raise ValueError("FACTS can only be added on lines with the same voltage ratings at both ends")
        
        # Define idxs for added buses. The convention is 99<t><nn>, where
        # 99 designates the added buses
        # <t> bus type 0 for a common middle bus, 1 and 2 for buses facing the sending and receiving part of the line
        # <nn> added controller
        bus0 = 99000+0+n
        bus1 = 99000+100+n
        bus2 = 99000+200+n
        ss.add('Bus', dict(idx=bus0, Vn=Vns, vmax=1.06, vmin=0.94, v0=1.0, area=2, zone=1))
        ss.add('Bus', dict(idx=bus1, Vn=Vns, vmax=1.06, vmin=0.94, v0=1.0, area=2, zone=1))
        ss.add('Bus', dict(idx=bus2, Vn=Vnr, vmax=1.06, vmin=0.94, v0=1.0, area=2, zone=1))

        ss.add('Line', dict(bus1 = bus_s_idx, 
                            bus2 = bus0,
                            name = 'FACTS %02d %s-%s' %(n+1, bus_s_idx, bus0),
                            x = k*xsr, 
                            r = k*rsr, 
                            b1 = k*bsr/2, b2=k*bsr/2, 
                            g1 = k*gsr/2, g2=k*gsr/2,
                            Vn1 = Vns, Vn2 = Vns))
        ss.add('Line', dict(bus1 = bus0, 
                            bus2 = bus_r_idx, 
                            name = 'FACTS %02d %s-%s' %(n+1, bus0, bus_r_idx),
                            x = (1-k)*xsr, 
                            r = (1-k)*rsr, 
                            b1 = (1-k)*bsr/2, b2=(1-k)*bsr/2, 
                            g1 = (1-k)*gsr/2, g2=(1-k)*gsr/2,
                            Vn1 = Vnr, Vn2 = Vnr))

        # Add line segments "s1" and "2r"
        # Add generators (PV elements) to buses "1" and "2"
    return

#%% Code testing
if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    ss = load_case('cases/', 'caseNPCC_wAreas.xlsx')
    interface_lines = area_interface_lines_indices(ss, 2)
    gens_in_a2 = area_generators_indices(ss, 2)
    gens_in_a1 = area_generators_indices(ss, 1)
    locations = len(interface_lines)*[0.6]
    # add_gpfcs(ss, interface_lines, locations, 2)
    # write(ss, 'jovan.xlsx')
    
    # preparing for exit
    logging.shutdown()
