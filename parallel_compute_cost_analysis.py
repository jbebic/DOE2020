# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:22:19 2021

@author: jzb@achillearesearch.com

"""

import numpy as np
import matplotlib.pyplot as plt

def cost_ratio(N, eff, Np=100, uselogscale=True):
    '''
    This function calculates cost ratio of parallel relative to sequential execution of 
    an embarassingly parallel workload.
    Assuming:
        ts = time required to set up for execution
        tw = time required to execute a unit of work
        N  = total required number of units of work
        n  = number of units of work assigned to a parallel process
    the total time to execute the job as a function of n is:
        T(n) = (ts + n*tw)*N/n
    N/n must be an integer. (This is not a limiting factor if N is a large number.) 
    
    Setting n=N gives the time of sequential execution:
        Ts = ts+N*tW
    
    Assuming that the cost of compute is directly proportional to the execution time, 
    a cost to execute the job in parallel can be expressed as a ratio to the cost to 
    execute the same job sequentially. (Sequential execution has the minimum cost 
    because the setup is incurred only once.)
    
    The cost ratio then is:
              (ts + n*tw)*N/n
        cr = -----------------
               ts + N*tw
    
    Rearranging the terms yields:
        cr = eff * (1 + ts/(n*tw)) = eff * (1 + ts/(N*tw)/(n/N)), 
    where: 
                 N*tw           
        eff = ----------- = 1/(1 + ts/(N*tw))
               ts + N*tw

    The same parameters can be used to determine the time ratio
              ts + n*tw
        tr = -----------
              ts + N*tw

    Rearranging the terms yields
        tr = n/N * cr
        
    Parameters
    ----------
    N : int
        Total number of units of work
    eff : float
        efficiency of a sequential process, used to extract ts/tw
    Np : int, optional
        number of points for the returned curve. The default is 100.
    uselogscale : boolean, optional
        return values on log or linear scale. The default is True, selecting a log scale.

    Returns
    -------
    nn = n/N
    cr = cost ratio
    tr = time ratio

    '''
    if uselogscale:
        nn = np.logspace(np.log10(1./N), np.log10(1.), Np)
    else:
        nn = np.linspace(1./N, 1., Np)
    
    ts_over_tw = N*(1/eff-1)
    
    cr = eff*(1 + ts_over_tw/N/nn)
    tr = nn*cr
   
    return nn, cr, tr, ts_over_tw

if __name__ == '__main__':

    N = 250
    eff = 0.98
    nn, cr1, tr1, tsotw = cost_ratio(N, eff, Np=20) # uselogscale=False
    
    fig, ax1 = plt.subplots(ncols=1)

    fig.suptitle('N=%d, eff=%g, ts/tw=%g' %(N, eff, tsotw))
    ax1.plot(nn*N, cr1, '-o')
    ax1.set_xlim(0,0.2*N)
    ax1.set_ylim(1, 3)
    ax1.set_ylabel('cost ratio')
    ax1.set_xlabel('units of work assigned to each process')
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiates a second axes that shares the same x-axis
    
    ax2.set_ylabel('execution time ratio')
    ax2.plot(nn*N, tr1, '-^')
    ax2.set_ylim(0,0.25)
    
    fig.show()
    