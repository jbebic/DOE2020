# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:27:16 2021

@author: txia4@vols.utk.edu
"""

import logging
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt # plotting
import numpy.matlib 
import matplotlib.backends.backend_pdf as dpdf # pdf output

from datetime import datetime # time stamps
import os # operating system interface

# import andes
# from andes.core.var import BaseVar, Algeb, ExtAlgeb

from npcc_voltages_severity import calculate_voltage_severity


def plot_voltage_transfer(pltPdf, plot_x, plot_y, plot_x2, plot_y2, title='', xlabel='Line contigency number',ylabel='Line-loading severity'):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4.8), dpi=300) # , sharex=True
    plt.suptitle(title)
    temp = datetime.now()
    temp = temp.strftime("%m/%d/%Y, %H:%M:%S")
    ax0.annotate(temp, (0.98,0.02), xycoords="figure fraction", horizontalalignment="right")
    
    # ax0.bar(plot_x, plot_y, label='Pseverity')
    ax0.plot(plot_x, plot_y, label='Pseverity')
    ax0.plot(plot_x2, plot_y2, label='Pseverity')


    ax0.grid(True, which='both', axis='both')
    #ax0.set_ylim([0,0.3]) # when range not specified, the graph autoscales 
    ax0.set_ylabel(ylabel)
    #ax0.set_xlabel('Voltage [pu]')
    ax0.set_xlabel(xlabel)
    
    pltPdf.savefig() # saves figure to the pdf file scpecified by pltPdf
    plt.title(title)
    plt.close() # Closes fig to clean up memory
    return

if __name__ == "__main__":

    # If changing basicConfig, make sure to close the dedicated console; it will not take otherwise
    logging.basicConfig(filename='logs/DOE2020.log', filemode='w', 
                        format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    
    # this supposedly shows where the file is, but it does not work for me
    # print(logging.getLoggerClass().root.handlers[0].baseFilename)
       
    if True: # loading the contingency results and test voltage severtiy
          
            
        dirplots = 'plots/' # must create this relative directory path before running
        fnameplot = 'Severity transfer function.pdf' # file name to save the plot
        pltPdf = dpdf.PdfPages(os.path.join(dirplots,fnameplot)) # opens a pdf file
        
        dv_values = np.array([0.05, 0.08])
        ks_values = np.array([10, 15])
        plot_x = np.linspace(0.8, 1.2, 100)
        plot_y = calculate_voltage_severity(plot_x, dv_values, ks_values, vnom=1.0)
        
        dv_values = np.array([0.03, 0.05, 0.08])
        ks_values = np.array([5, 10, 15])
        plot_x2 = np.linspace(0.8, 1.2, 100)
        plot_y2 = calculate_voltage_severity(plot_x, dv_values, ks_values, vnom=1.0)
        
        title_temp = 'Bus-voltage severity transfer function' 
        plot_voltage_transfer(pltPdf, plot_x, plot_y, plot_x2, plot_y2, title_temp,xlabel='Bus voltage [p.u.]',ylabel='Bus-voltage severity') # places a plot page into the pdf file      

        dv_values = np.array([1.0, 1.25, 1.725])
        ks_values = np.array([10, 15, 20])      
        plot_x = np.linspace(-1.2, 1.2, 100)
        plot_y = calculate_voltage_severity(plot_x, dv_values, ks_values, vnom=0)
        
        dv_values = np.array([0.8, 1.0, 1.25, 1.725])
        ks_values = np.array([5, 10, 15, 20])      
        plot_x2 = np.linspace(-1.2, 1.2, 100)
        plot_y2 = calculate_voltage_severity(plot_x, dv_values, ks_values, vnom=0)
        
        title_temp = 'Line-loading severity transfer function' 
        plot_voltage_transfer(pltPdf, plot_x, plot_y, plot_x2, plot_y2, title_temp,xlabel='Line-loading [p.u.]',ylabel='Line-loading severity') # places a plot page into the pdf file                      


          
        pltPdf.close() # closes a pdf file
    
      
    logging.shutdown()
    print('end')
        
        
        