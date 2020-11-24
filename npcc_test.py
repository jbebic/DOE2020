# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:27:17 2020
@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com

v1.0 JZB20200903
Preliminary analysis of NPCC cases from ANDES including visualization
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb

#%%
def save_results(ss:andes.system, fout):
    fout.write('Jovan was here\n')
    pass

#%% 
def run_generator_contingencies(ss:andes.system):
    pass

if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/SBIR.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    if True:        
        
        dirin = 'cases/'
        fnamein = 'npcc.raw'; fnamedyr = 'npcc_full.dyr'
        # fnamecoord = 'npcc_globalpos.csv'
        
        dirout = 'output/' # output directory
        fout = open(os.path.join(dirin,fnamein.replace('.raw', '.csv')), 'w')
        
        # output file names are based on the input file name + a suffix
        # suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        
        ss = andes.run(fnamein, addfile=fnamedyr, input_path=dirin, output_path=dirout) # fails on dyr file path
        # ss = andes.run(fnamein, addfile=os.path.join(dirin,fnamedyr), input_path=dirin, output_path=dirout) # workaround
        for uid in ss.PV.cache.df.index:
            gen = ss.PV.cache.df.loc[uid]
            ss.PV.u.v[uid] = 0
            ss.PFlow.run()
            save_results(ss, fout)
            ss.PV.u.v[uid] = 1

        fout.close()

    if False:        
        
        dirin = '../andes/andes/cases/kundur/' # specifies an arbitrary path to input file
        fnamein = 'kundur_full.xlsx' # specifies the system data file
        dirout = 'output/' # specifies the output path
        
        ss = andes.run(os.path.join(dirin,fnamein)) # creates a .txt file
        ss.files.npz = os.path.join(dirout, ss.files.npz) # changes the location of the .npz file
        ss.files.npy = os.path.join(dirout, ss.files.npy) # ditto .npy file
        ss.files.lst = os.path.join(dirout, ss.files.lst) # ditto .lst file
        # ss.files.txt = os.path.join(dirout, ss.files.txt) # changing the path here is too late --  .txt file was created in andes.run above
        
        ss.Toggler.alter('u', 1, 0)
        
        ss.TDS.config.tf = 1 # set final time to 1 sec.
        ss.TDS.run() # run time domain simulation
        
        ss.TGOV1.paux0.v[0] = 0.05 # set paux
        ss.TDS.config.tf = 2 # set final time to 2 sec.
        ss.TDS.run() # run time domain simulation
        
        ss.TGOV1.paux0.v[0] = 0. # reset paux
        ss.TDS.config.tf = 10 # set final time to 10 sec.
        ss.TDS.run() # run time domain simulation

        # ss.TDS.plt.file_name = dirout + ss.TDS.plt.file_name # somehow this is already done by earlier filename interventions
        ss.TDS.plotter.plot(ss.TGOV1.paux, savefig=True, save_format='pdf')
        ss.TDS.plotter.plot(ss.TGOV1.pout, savefig=True, save_format='pdf')
        ss.TDS.plotter.plot(ss.GENROU.omega, savefig=True, save_format='pdf')
        
        # It is possible to extract variables as numpy arrays
        # the results can be retrieved by TDS.plotter.get_values using the correct column index
        # the columns are time, states, algebraic variables
        print('Number of states in the case is %d' %ss.TDS.plotter.dae.n)
        
        xidx = (0,) # time is always at column zero
        yidx = ss.GENROU.omega # transfer the desired symbolic name to yidx
        if isinstance(yidx, BaseVar):
                if yidx.n == 0:
                    raise ValueError('Case contains no data' )
                offs = 1
                if isinstance(yidx, (Algeb, ExtAlgeb)):
                    offs += ss.TDS.plotter.dae.n
                yidx = yidx.a + offs

        yidx = np.take(yidx, [0, 2]) # use this to downselect indices of desired variables within the same type
        # check headers to confirm that the right index was used
        print('Header  of the time aray is %s' %ss.TDS.plotter.get_header(xidx))
        print('Headers of the selected variables are %s' %ss.TDS.plotter.get_header(yidx))
        t  = ss.TDS.plotter.get_values(xidx) # fetches the time array
        ys = ss.TDS.plotter.get_values(yidx) # fetches the matrix of values
        
        # another way is to get variables from the output file
        results = np.load(ss.files.npz) 
        print('shape of results is: (%d, %d)' %results['data'].shape)
        
        # Move the .txt file to the dirout folder
        fname_txt = fnamein.replace('.xlsx', '_out.txt')
        os.replace(fname_txt, os.path.join(dirout, fname_txt))
        
    # preparing for exit
    logging.shutdown()
