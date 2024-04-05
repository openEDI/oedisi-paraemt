## An example of dynamic simulation
## Frequency response after a generator trip event (3018).
## change the file address (.raw, .dyr, and .out) before you run!!!pip 
## Please let me know if you have questions! Hongyu Li (hli90@utk.edu)

import os
import sys

sys_path_PSSE = r"C:\Program Files (x86)\PTI\PSSE34\PSSPY37"  #or where else you find the psspy.pyc
ierr=sys.path.append(sys_path_PSSE)
os.environ['PATH'] += ';' + sys_path_PSSE

os_path_PSSE = r"C:\Program Files (x86)\PTI\PSSE34\PSSPY37"  # or where else you find the psse.exe
ierr=sys.path.append(os_path_PSSE)
os.environ['PATH'] += ';' + os_path_PSSE

import psse34
import psspy
from psspy import _i, _f

ierr=psspy.psseinit(0) ## initial PSS/E, like open PSS/E.

## read raw (power flow) file
psspy.read(0,r"""C:\Users\mxiong3\Desktop\20231213_ParaEMT\models\2area_psse\Benchmark_twoarea_v33.raw""")

## modify raw file illustration
#psspy.machine_chng_2(3432,r"""S""",[_i,_i,_i,_i,_i,1],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # 1 is for S 
#psspy.machine_chng_2(3432,r"""S""",[_i,_i,_i,_i,_i,3],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # 3 is for DP
## DP have 0 limits on reactive power 

# Pgen Qgen Qmax Qmin Pmax Pmin Mbase
psspy.machine_chng_2(1,r"""G""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
psspy.mbidmac(1,r"""G""",r"""S""")
psspy.machine_chng_2(1,r"""S""",[_i,_i,_i,_i,_i,1],[_f,_f,_f,_f,_f,_f, _f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# psspy.machine_chng_2(3432,r"""NP""",[_i,_i,_i,_i,_i,1],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.mbidmac(3432,r"""NP""",r"""S""")
# psspy.machine_chng_2(3432,r"""S""",[_i,_i,_i,_i,_i,1],[0.00,_f,_f,_f,_f,_f,  1000.00,    _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # need to give some active power to S

# psspy.machine_chng_2(3831,r"""NN""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.mbidmac(3831,r"""NN""",r"""S""")
# psspy.machine_chng_2(3831,r"""S""",[_i,_i,_i,_i,_i,1],[_f,_f,_f,_f,_f,_f,  4000.00,   _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# psspy.machine_chng_2(2130,r"""G""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(2130,r"""S""",[_i,_i,_i,_i,_i,_i],[ 470.00, 61.188,    86,-86,  700.00,0,   1000,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# psspy.machine_chng_2(2233,r"""DG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(2233,r"""EG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  
# psspy.machine_chng_2(2233,r"""TG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
# psspy.machine_chng_2(2233,r"""S""",[1,_i,_i,_i,_i,_i],[ 760.00,   502.2820, 600,_f,  1000.00, _f,   1500,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # in service

# psspy.machine_chng_2(2438,r"""EG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(2438,r"""RG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  
# psspy.machine_chng_2(2438,r"""SG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
# psspy.machine_chng_2(2438,r"""WG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(2438,r"""S""",[_i,_i,_i,_i,_i,_i],[  2245.50,   106.6250,_f,_f,  3088.70, _f,   4146,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(2438,r"""SW""",[_i,_i,_i,_i,_i,_i],[ 2245.50,    106.6250,_f,_f,  3049.50, _f,  4210,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# psspy.machine_chng_2(2630,r"""G""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(2630,r"""S""",[1,_i,_i,_i,_i,_i],[ 2269.00,   184.5390, 1346.0000,_f,  3749.6500, _f,  3947.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # in service

# psspy.machine_chng_2(2634,r"""C""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(2634,r"""S""",[1,_i,_i,_i,_i,_i],[ 1537.00,    425.5210, 950.0000,_f, 1537.1000, _f,   2618.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # in service

# psspy.machine_chng_2(3133,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3133,r"""S""", [1,_i,_i,_i,_i,_i],[ 1.00,   2.00, 2.00,_f, 58.90, _f,    10.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # in service
# psspy.machine_chng_2(3133,r"""SC""",[_i,_i,_i,_i,_i,_i],[ _f,   227.053, 500.00, -500.00,   _f, _f,  500.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3135,r"""MG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3135,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3135,r"""S""", [1,_i,_i,_i,_i,_i],[ 220.00,   87.939, 239.00, -239.00,  550.00, _f,  1000.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3234,r"""DG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3234,r"""MG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3234,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
#psspy.machine_chng_2(3234,r"""S""", [1,_i,_i,_i,_i,_i],[  2965.00,    350.6010, 500.00, -500.00,  4000.00, _f, 4000.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
# better to move to both renewables
# psspy.machine_chng_2(3234,r"""S""", [1,_i,_i,_i,_i,_i],[  1990.50,    235.3695, 500.00, -500.00,  3000.00, _f, 3000.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
# psspy.machine_chng_2(3234,r"""NW""",[_i,_i,_i,_i,_i,_i],[ 1990.50,    235.3695, 500.00, -500.00,  3000.00, _f, 3000.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3333,r"""CG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3333,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3333,r"""S""", [1,_i,_i,_i,_i,_i],[ 726.00,   455.02, 600.00, -600.00, 1000.00, 0,   2000.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3433,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3433,r"""S""",[_i,_i,_i,_i,_i,_i],[ 1105.00,    48.8680,_f,_f,  _f, _f, 2426.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
# # move 100 MW to 3432 S 
#psspy.machine_chng_2(3433,r"""S""",[_i,_i,_i,_i,_i,_i],[ 1005.00,    48.8680,_f,_f,  _f, _f, 2426.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3631,r"""NB""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3631,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3631,r"""S""", [1,_i,_i,_i,_i,_i],[ 97.00,    -8.8220, 94.0000, -97.0000, 200.00, _f,  200.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

## 3835 Test 1
# psspy.machine_chng_2(3835,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3835,r"""S""",[_i,_i,_i,_i,_i,_i],[ 359.00,  139.7360,_f,_f,  _f, _f,   1333.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# 3835 Test 2
# psspy.machine_chng_2(3835,r"""NG""",[_i,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.mbidmac(3835,r"""NG""",r"""NW""")
# psspy.machine_chng_2(3835,r"""NW""",[_i,_i,_i,_i,_i,1],[_f,_f,_f,_f,_f,_f,  1000.00 ,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# psspy.machine_chng_2(3836,r"""DG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3836,r"""S""", [1,_i,_i,_i,_i,_i],[ 679,   -0.8410, 476.00, -476.00,  1422.15, _f,  1497.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3931,r"""NB""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3931,r"""NH""",[_i,_i,_i,_i,_i,_i],[ 1447.00,    315.7500, _f,_f,  2731.25, _f, _f,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  

# psspy.machine_chng_2(3933,r"""CG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(3933,r"""NB""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3933,r"""NG""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
# psspy.machine_chng_2(3933,r"""S""", [_i,_i,_i,_i,_i,_i],[ 14.9930,   511.125, _f, _f,  _f, _f,  _f,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 

# psspy.machine_chng_2(8034,r"""G""",[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  # out service
# psspy.machine_chng_2(8034,r"""S""",[1,_i,_i,_i,_i,_i],[ 2756.00,   607.8860, 1280.0000, -1280.00,  3566.30, 0,  3754.00,  _f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 

## Save it to a new file
psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,r"""C:\Users\mxiong3\Desktop\20231213_ParaEMT\models\2area_psse\Benchmark_twoarea_v33_Modify.raw""")

## solve power flow
psspy.fdns([0,0,0,1,1,0,99,0])

## Convert generators
psspy.cong(0)
## Convert generators
## constant current and impedance
psspy.conl(0,1,1,[0,0],[ 100.0,0.0,0.0, 100.0])
psspy.conl(0,1,2,[0,0],[ 100.0,0.0,0.0, 100.0])
psspy.conl(0,1,3,[0,0],[ 100.0,0.0,0.0, 100.0])

## constant impedance
# psspy.conl(0,1,1,[0,0],[0.0, 100.0,0.0, 100.0])
# psspy.conl(0,1,2,[0,0],[0.0, 100.0,0.0, 100.0])
# psspy.conl(0,1,3,[0,0],[0.0, 100.0,0.0, 100.0])

## read dyr (dynamic model location, type, and parameters, etc) file
#psspy.dyre_new([1,1,1,1],r"""C:\Users\mxiong3\Desktop\240busPSSE\WECC240_v04_DPV_RE20_v33_6302_xfmr_DPbuscode_PFadjusted_2619_2600G.dyr""","","","")
psspy.dyre_new([1,1,1,1],r"""C:\Users\mxiong3\Desktop\240busPSSE\WECC240_v04_DPV_RE100.dyr""","","","")

## delete the default save channels. If not do so, in batch simulation, psspy.chsb function keeping adding the same channels with no considerion if the adding channels exists or not. 
psspy.delete_all_plot_channels()

## Add channel, select the varable you want to save
#psspy.chsb(0,1,[-1,-1,-1,1,12,0])
psspy.chsb(0,1,[-1,-1,-1,1,1,0])
psspy.chsb(0,1,[-1,-1,-1,1,2,0])
psspy.chsb(0,1,[-1,-1,-1,1,3,0])
psspy.chsb(0,1,[-1,-1,-1,1,4,0])
psspy.chsb(0,1,[-1,-1,-1,1,5,0])
psspy.chsb(0,1,[-1,-1,-1,1,6,0])
psspy.chsb(0,1,[-1,-1,-1,1,7,0])
psspy.chsb(0,1,[-1,-1,-1,1,13,0])
psspy.chsb(0,1,[-1,-1,-1,1,16,0])

## set output file address
psspy.strt_2([0,0],r"""C:\Users\mxiong3\Desktop\240busPSSE\test100RE.out""")

## prefault simulation
psspy.run(0, 5.0,10,1,0)

## add distubance
# psspy.dist_machine_trip(1032,r"""C""")
psspy.dist_machine_trip(1431,r"""N""")
#psspy.dist_machine_trip(1431,r"""S""")
#psspy.dist_machine_trip(1131,r"""C""")
#psspy.dist_machine_trip(1331,r"""H""")
#psspy.dist_machine_trip(2130,r"""E""")
#psspy.dist_machine_trip(1131,r"""C""")

## Line grounded
#psspy.dist_3phase_bus_fault(7,0,1, 230.0,[ 0, -0.2E+10])  # bus 7 grounded

## during fault simulation
#psspy.run(0, 1+5/60,10,1,0)

## clear fault
#psspy.dist_clear_fault(1)

## postfault simulation
psspy.run(0, 20.0,10,1,0)