# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 08:41:52 2020
@copyright: Achillea Research, Inc.
@author: jzb@achillearesearch.com

v1.0 JZB20201125
Importing the npcc case in matpower format, solving the loadflow, and saving 
the results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting

from datetime import datetime # time stamps
import os # operating system interface

import andes
from andes.core.var import BaseVar, Algeb, ExtAlgeb

def save_results(ss:andes.system, fout):
    
    return

if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)

    if False: # Loading the PSS/E file
        
        dirin = 'cases/'
        fnamein = 'npcc.raw'
        
        dirout = 'output/' # output directory
        # ANDES output file names are derived from the input file name with an added suffix
        # The suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
        
        # adding one more output file to save csv results
        fout = open(os.path.join(dirout,fnamein.replace('.raw', '.csv')), 'w')
        save_results(ss, fout)
        fout.close()

    if True: # loading the matpower file
        dirin = 'cases/'
        fnamein = 'caseNPCC.m'
        
        dirout = 'output/' # output directory
        # ANDES output file names are derived from the input file name with an added suffix
        # The suffix can be one or more of : _out.txt, _out.lst, _out.npz, _out_N.pdf
        ss = andes.run(fnamein, input_path=dirin, output_path=dirout)
        
        # adding one more output file to save csv results
        fout = open(os.path.join(dirout,fnamein.replace('.m', '.csv')), 'w')
        save_results(ss, fout)
        fout.close()

    # preparing for exit
    logging.shutdown()
