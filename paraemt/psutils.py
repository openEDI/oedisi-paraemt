
import json
import numpy as np
import os
import time
import pandas as pd

import xlrd
# xlrd.xlsx.ensure_elementtree_imported(False, None)
# xlrd.xlsx.Element_has_iter = True

from paraemt.dydata import DyData
from paraemt.emtsimu import EmtSimu
from paraemt.initialize import Initialize
from paraemt.pfdata import PFData

# from preprocessscript import get_json_pkl

import pickle
import scipy.sparse as sp
import scipy.sparse.linalg as la
from copy import deepcopy

class storage:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_json_pkl(filename):

    f = open(filename,)
    data = json.load(f)

    for x in data.keys():
        if type(data[x]) != list:
            pass
        else:
            try:
                if "j" in data[x][0]:
                    tmp = [np.complex(y) for y in data[x]]
                    data[x] = np.array(tmp)
                else:
                    tmp = data[x]
                    data[x] = np.array(tmp)
            except Exception as e:
                tmp = data[x]
                data[x] = np.array(tmp)

    pfd = storage(**data)

    return (pfd)


def load_pfd(filename):

    return PFData.load_from_json(get_json_pkl(filename))


def initialize_emt(workingfolder,
                   systemN, EMT_N, N_row, N_col,
                   ts, Tlen, kts, stepk,
                   save_rate,
                   network_mode, loadmodel_option, record4cosim):

    # print("Running simulation for system {:d} in layout {:d} x {:d}".format(systemN, N_row, N_col))
#     info_str = """
# ---- Simulation Info ----
# Running:
#   System: {:d}
#   Layout: {:d} x {:d}
#   Length: {:g} (s)
#   Step:   {:g} (s)
# -------------------------
# """.format(systemN, N_row, N_col, Tlen, ts)
#     print(info_str)

    if systemN == 1:
        pfd_name = 'pfd_4_' + str(N_row) + '_' + str(N_col)
    elif systemN == 2:
        pfd_name = 'pfd_9_' + str(N_row) + '_' + str(N_col)
    elif systemN == 3:
        pfd_name = 'pfd_39_' + str(N_row) + '_' + str(N_col)
    elif systemN == 4:
        pfd_name = 'pfd_179_' + str(N_row) + '_' + str(N_col)
    elif systemN == 5:
        pfd_name = 'pfd_240_' + str(N_row) + '_' + str(N_col)
    elif systemN == 6:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col)
    elif systemN == 7:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col) + '_red'
    elif systemN == 8:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col) + '_ibrepri'
    elif systemN == 9:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col) + '_red'
    elif systemN == 10:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col)
    elif systemN == 11:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col) + '_mod'
    elif systemN == 12:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col) + '_mod_X1'
    elif systemN == 13:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col) + '_mod_2ibr'
    elif systemN == 14:
        pfd_name = 'pfd_240_' + str(N_row) + '_' + str(N_col)
    elif systemN == 15:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col)
    elif systemN == 16:
        pfd_name = 'pfd_2area_' + str(N_row) + '_' + str(N_col)
    elif systemN == 17:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col) + '_mod_tripSG'
    elif systemN == 18 or systemN == 19 or systemN == 20:
        pfd_name = 'pfd_3bus_' + str(N_row) + '_' + str(N_col) + '_newcase' + str(systemN-17)
    elif systemN == 21:
        pfd_name = 'pfd_2gen_' + str(N_row) + '_' + str(N_col)
    elif systemN == 22:
        pfd_name = 'pfd_240_mod_' + str(N_row) + '_' + str(N_col)
    elif systemN == 23:
        pfd_name = 'pfd_240_' + str(N_row) + '_' + str(N_col) + '_old'
    else:
        pfd_name = 'pfd_{}_{}_{}'.format(systemN, N_row, N_col)

    filename = os.path.join(workingfolder, 'cases', pfd_name + '.json')
    pfd = load_pfd(filename)

    ## added Min
    # dfsgbase = pd.DataFrame(np.transpose(pfd.gen_MVA_base))
    # dfsgbase.to_csv("emt_gen_mvabase.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.ibr_MVA_base))
    # ibrbase.to_csv("emt_ibr_mvabase.csv")

    # ibrbase = pd.DataFrame(np.transpose(pfd.bus_name))
    # ibrbase.to_csv("busNAME.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.bus_num))
    # ibrbase.to_csv("busNUMBER.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.gen_id))
    # ibrbase.to_csv("genID.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.gen_bus))
    # ibrbase.to_csv("genBUS.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.ibr_id))
    # ibrbase.to_csv("ibrID.csv")
    # ibrbase = pd.DataFrame(np.transpose(pfd.ibr_bus))
    # ibrbase.to_csv("ibrBUS.csv")

    # # TODO: to be removed after re-creating a json file with two DC IBRs having Fixed mode
    # if systemN == 5:
    #     idx = 34
    #     ibr_bus = np.copy(pfd.ibr_bus)
    #     ibr_bus = np.insert(ibr_bus, idx, pfd.gen_bus[28])
    #     ibr_bus = np.insert(ibr_bus, idx+4, pfd.gen_bus[29])
    #     pfd.ibr_bus = np.copy(ibr_bus)

    #     ibr_id = np.copy(pfd.ibr_id)
    #     ibr_id = np.insert(ibr_id, idx, pfd.gen_id[28])
    #     ibr_id = np.insert(ibr_id, idx+4, pfd.gen_id[29])
    #     pfd.ibr_id = np.copy(ibr_id)

    #     ibr_MW = np.copy(pfd.ibr_MW)
    #     ibr_MW = np.insert(ibr_MW, idx, pfd.gen_MW[28])
    #     ibr_MW = np.insert(ibr_MW, idx+4, pfd.gen_MW[29])
    #     pfd.ibr_MW = np.copy(ibr_MW)

    #     ibr_Mvar = np.copy(pfd.ibr_Mvar)
    #     ibr_Mvar = np.insert(ibr_Mvar, idx, pfd.gen_Mvar[28])
    #     ibr_Mvar = np.insert(ibr_Mvar, idx+4, pfd.gen_Mvar[29])
    #     pfd.ibr_Mvar = np.copy(ibr_Mvar)

    #     ibr_MVA_base = np.copy(pfd.ibr_MVA_base)
    #     ibr_MVA_base = np.insert(ibr_MVA_base, idx, pfd.gen_MVA_base[28])
    #     ibr_MVA_base = np.insert(ibr_MVA_base, idx+4, pfd.gen_MVA_base[29])
    #     pfd.ibr_MVA_base = np.copy(ibr_MVA_base)

    #     pfd.gen_id = np.delete(pfd.gen_id, [28,29])
    #     pfd.gen_bus = np.delete(pfd.gen_bus, [28,29])
    #     pfd.gen_MVA_base = np.delete(pfd.gen_MVA_base, [28,29])
    #     pfd.gen_S = np.delete(pfd.gen_S, [28,29])
    #     pfd.gen_mod = np.delete(pfd.gen_mod, [28,29])
    #     pfd.gen_MW = np.delete(pfd.gen_MW, [28,29])
    #     pfd.gen_Mvar = np.delete(pfd.gen_Mvar, [28,29])
    # # end of TODO

    
    # load dynamic data in a certain format
    if systemN == 1:
        fdyd = os.path.join('2gen_psse', '2gen_GAST.xlsx')
    elif systemN == 2:
        fdyd = os.path.join('9bus_psse', '9bus.xlsx')
    elif systemN == 3:
        # fdyd = os.path.join('39bus_psse', '39bus.xlsx')
        fdyd = os.path.join('39bus_psse', '39bus_epri_ibr.xlsx')
    elif systemN == 4:
        fdyd = os.path.join('179bus_psse', '179bus.xlsx')
    elif systemN == 5 or systemN == 22:
        fdyd = os.path.join('240bus_psse','240bus.xlsx')
    elif systemN == 6:
        # fdyd = os.path.join('2area_psse','OneGov','TGOV1','Benchmark_twoarea.xlsx')
        fdyd = os.path.join('2area_psse','Benchmark_twoarea_test.xlsx')
        # fdyd = os.path.join('2area_psse','Benchmark_twoarea_G1IBR.xlsx')
        # fdyd = os.path.join('2area_psse','Benchmark_twoarea_G1_SGwithIBR.xlsx')
        # fdyd = os.path.join('2area_psse','Benchmark_twoarea_All_modify.xlsx')
    elif systemN == 7:
        fdyd = os.path.join('2area_psse', 'OneGov', 'TGOV1', 'Benchmark_twoarea.xlsx')
    elif systemN == 8:
        fdyd = os.path.join('2area_psse', 'OneGov', 'TGOV1', 'Benchmark_twoarea_ibrepri.xlsx')
    elif systemN == 9:
        fdyd = os.path.join('2area_psse', 'OneGov', 'TGOV1', 'Benchmark_twoarea.xlsx')
    elif systemN == 10 or systemN == 11 or systemN == 12:
        fdyd = os.path.join('ibr_psse', 'Benchmark_ibrepri.xlsx')
    elif systemN == 13:
        fdyd = os.path.join('ibr_psse', 'Benchmark_ibrepri_2.xlsx')
    elif systemN == 14:
        fdyd = os.path.join('240bus_psse', '240bus_ibrepri.xlsx')
    elif systemN == 15:
        fdyd = os.path.join('2area_psse','OneGov','TGOV1','Benchmark_twoarea.xlsx')
    elif systemN == 16:
        fdyd = os.path.join('2area_psse','OneGov','TGOV1','Benchmark_twoarea.xlsx')
    elif systemN == 17:
        fdyd = os.path.join('ibr_psse', 'Benchmark_ibrepri_tripSG.xlsx')
    elif systemN == 18 or systemN == 20:
        fdyd = os.path.join('ibr_psse', 'Benchmark_ibrepri_newcase.xlsx')
    elif systemN == 19:
        fdyd = os.path.join('ibr_psse', 'Benchmark_ibrepri_newcase2.xlsx')
    elif systemN == 21:
        fdyd = os.path.join('2gen_matlab', '2gen.xlsx')
    elif systemN == 23:
        fdyd = os.path.join('240bus_psse', '240bus_old.xlsx')
        # fdyd = os.path.join('240bus_psse', '240bus_old_revised.xlsx')
        # fdyd = os.path.join('240bus_psse', '240bus_37epri_ibr.xlsx')
    else:
        raise RuntimeError("Unrecognized system number: {:d}".format(systemN))
    ## End if

    file_dydata = os.path.join(workingfolder, 'models', fdyd)
    dyd0 = DyData()
    dyd0.getdata(file_dydata, pfd, N_row*N_col)

    if N_row*N_col > 1:
        dyd = dyd0.spreaddyd(pfd, dyd0, N_row*N_col)
    else:
        dyd = dyd0

    dyd.ToEquiCirData(pfd, dyd)


    # load network partitioning data
    if systemN == 6 or systemN == 7 or systemN == 15:
        fnwp = os.path.join('2area_psse','nwpart.xlsx')
    elif systemN == 14 or systemN == 5 or systemN == 23:
        fnwp = os.path.join('240bus_psse','nwpart.xlsx')
    else:
        fnwp = []
        # raise RuntimeError("Unrecognized system number: {:d}".format(systemN))

    if len(fnwp)>0:
        file_nwpdata = os.path.join(workingfolder, 'models', fnwp)
        nwp_spreadsheet = xlrd.open_workbook(file_nwpdata)
        nwp_data = nwp_spreadsheet.sheet_by_index(0)
        emtzones_n = nwp_data.ncols - 1
        emt_zones = []
        for i in range(emtzones_n):
            bdrybus_n = int(nwp_data.cell_value(1, i + 1))
            bdrybus = []
            for j in range(bdrybus_n):
                bdrybus = np.append(bdrybus, nwp_data.cell_value(2 + j, i + 1))
            zbuses_n = int(nwp_data.cell_value(bdrybus_n + 2, i + 1))
            zbuses = bdrybus.copy()
            for j in range(zbuses_n):
                zbuses = np.append(zbuses, int(nwp_data.cell_value(bdrybus_n + 3 + j, i + 1)))
            emtzonei = {'buses':zbuses, 'bus_b':bdrybus}
            emt_zones = np.append(emt_zones, emtzonei) 
    else:
        emt_zones = []
        pass 



    # extract the EMT area
    if len(emt_zones)>0:
        if EMT_N>len(emt_zones):
            print('EMT N should not exceed num of EMT zones')

    if len(emt_zones)>0 and record4cosim == False:
        del_gen_idx, del_ibr_idx, emt_buses = pfd.extract_emt_area([emt_zones[EMT_N - 1]])
        dyd.extract_emt_area(del_gen_idx, del_ibr_idx, emt_buses, pfd)


    if record4cosim == False:
        for i in range(3):    # TODO: Added for delet load at boundary bus  01112024 Min
            j=0
            pfd.load_MW = np.delete(pfd.load_MW, j)
            pfd.load_Mvar = np.delete(pfd.load_Mvar, j)
            pfd.load_Z = np.delete(pfd.load_Z, j)
            pfd.load_I = np.delete(pfd.load_I, j)
            pfd.load_P = np.delete(pfd.load_P, j)
            pfd.load_bus = np.delete(pfd.load_bus, j)
            pfd.load_id = np.delete(pfd.load_id, j)
        
   # No need
    # pfd.load_P[0:3]=0
    # pfd.load_MW[0:3]=0
    # pfd.load_Mvar[0:3]=0
    # initialize three-phase system
        
    ti_0 = time.time()

    ini = Initialize(pfd, dyd)

    ti_1 = time.time()

    ini.InitNet(pfd, dyd, ts, loadmodel_option)

    ti_2 = time.time()

    ini.InitMac(pfd, dyd)

    ti_3 = time.time()

    ini.InitExc(pfd, dyd)

    ti_4 = time.time()

    ini.InitGov(pfd, dyd)

    ti_5 = time.time()

    ini.InitPss(pfd, dyd)  # IEEEST added

    ti_6 = time.time()

    ini.InitREGCA(pfd, dyd)

    ti_7 = time.time()

    ini.InitREECB(pfd, dyd)

    ti_8 = time.time()

    ini.InitREPCA(pfd, dyd)

    ini.InitPLL(pfd, dyd)

    ini.InitIbrepri(pfd, dyd)

    ti_9 = time.time()

    ini.InitBusMea(pfd)

    ti_10 = time.time()

    ini.InitLoad(pfd)

    ti_11 = time.time()

    ini.CheckMacEq(pfd, dyd)

    ti_12 = time.time()

    ini.MergeIbrG(pfd, dyd, ts, [])

    ini.MergeMacG(pfd, dyd, ts, [])

    ini.InitIhis()

    ini.CalcGnGinv(network_mode)

    ti_13 = time.time()

    # elapsed = ti_13 - ti_0
    # tini_str = """
    # Construct:   {:10.2e} {:8.2%}
    # InitNet:     {:10.2e} {:8.2%}
    # InitMac:     {:10.2e} {:8.2%}
    # InitExc:     {:10.2e} {:8.2%}
    # InitGov:     {:10.2e} {:8.2%}
    # InitPss:     {:10.2e} {:8.2%}
    # InitRegca:   {:10.2e} {:8.2%}
    # InitReecb:   {:10.2e} {:8.2%}
    # InitRepca:   {:10.2e} {:8.2%}
    # InitBusMea:  {:10.2e} {:8.2%}
    # InitLoad:    {:10.2e} {:8.2%}
    # CheckMacEq:  {:10.2e} {:8.2%}
    # MergeMacG:   {:10.2e} {:8.2%}
    # Init:        {:10.2e}
    # """.format(
    #     ti_1 - ti_0, (ti_1 - ti_0)/elapsed,
    #     ti_2 - ti_1, (ti_2 - ti_1)/elapsed,
    #     ti_3 - ti_2, (ti_3 - ti_2)/elapsed,
    #     ti_4 - ti_3, (ti_4 - ti_3)/elapsed,
    #     ti_5 - ti_4, (ti_5 - ti_4)/elapsed,
    #     ti_6 - ti_5, (ti_6 - ti_5)/elapsed,
    #     ti_7 - ti_6, (ti_7 - ti_6)/elapsed,
    #     ti_8 - ti_7, (ti_8 - ti_7)/elapsed,
    #     ti_9 - ti_8, (ti_9 - ti_8)/elapsed,
    #     ti_10 - ti_9, (ti_10 - ti_9)/elapsed,
    #     ti_11 - ti_10, (ti_11 - ti_10)/elapsed,
    #     ti_12 - ti_11, (ti_12 - ti_11)/elapsed,
    #     ti_13 - ti_12, (ti_13 - ti_12)/elapsed,
    #     elapsed
    # )

    # print(tini_str)

    # emt.preprocess(ini, pfd, dyd)

    return (pfd, ini, dyd, emt_zones)


def initialize_from_snp(input_snp, netMod):

    with open(input_snp, 'rb') as f:
        pfd, dyd, ini, emt = pickle.load(f)
        emt.t = [0.0]
        emt.save_idx = 0
        # emt.v = {}
        # emt.v[0] = x
        ini.Init_net_G0 = sp.coo_matrix((ini.Init_net_G0_data, (ini.Init_net_G0_rows, ini.Init_net_G0_cols)),
                                        shape=(ini.Init_net_N, ini.Init_net_N)
                                        ).tolil()

        if netMod == 'inv':
            ini.Init_net_G0_inv = la.inv(ini.Init_net_G0.tocsc())
        elif netMod == 'lu':
            ini.Init_net_G0_lu = la.splu(ini.Init_net_G0.tocsc())
        elif netMod == 'bbd':
            pass
        else:
            raise ValueError('Unrecognized mode: {}'.format(netMod))
        ini.admittance_mode = netMod

        emt.ibr_epri = [MODELINSTANCE() for jj in range(dyd.ibr_epri_n)]
        ini.InitIbrepri(pfd, dyd, emt)

    ## End with

    return (pfd, ini, dyd, emt)


def initialize_fault(emt, mode = 'inv'):

    pfd = emt.pfd
    ini = emt.ini
    dyd = emt.dyd

    rt = np.real(pfd.line_RX[emt.fault_line_idx])
    Lt = np.imag(pfd.line_RX[emt.fault_line_idx])
    Ct = pfd.line_chg[emt.fault_line_idx]

    Yft = np.zeros((9, 9), dtype=complex)

    # get base
    nbus = len(pfd.bus_num)
    Init_net_ZbaseA = np.zeros(nbus)
    for i in range(nbus):
        Vbase_temp = pfd.bus_basekV[i] / np.sqrt(3.0)
        Init_net_ZbaseA[i] = Vbase_temp * Vbase_temp * 3 / pfd.basemva

    busi = np.where(pfd.bus_num==pfd.line_from[emt.fault_line_idx])[0][0]

    if emt.fault_type == 1:
        r1g = emt.fault_r[0] / Init_net_ZbaseA[busi]
        r2g = np.Inf
        r3g = np.Inf
        r12 = np.Inf
        r23 = np.Inf
        r13 = np.Inf
    elif emt.fault_type == 2:
        r1g = np.Inf
        r2g = emt.fault_r[1] / Init_net_ZbaseA[busi]
        r3g = np.Inf
        r12 = np.Inf
        r23 = np.Inf
        r13 = np.Inf
    elif emt.fault_type == 3:
        r1g = np.Inf
        r2g = np.Inf
        r3g = emt.fault_r[2] / Init_net_ZbaseA[busi]
        r12 = np.Inf
        r23 = np.Inf
        r13 = np.Inf
    elif emt.fault_type == 4:
        r1g = emt.fault_r[0] / Init_net_ZbaseA[busi]
        r2g = emt.fault_r[1] / Init_net_ZbaseA[busi]
        r3g = np.Inf
        r12 = emt.fault_r[3] / Init_net_ZbaseA[busi]
        r23 = np.Inf
        r13 = np.Inf
    elif emt.fault_type == 5:
        r1g = np.Inf
        r2g = emt.fault_r[1] / Init_net_ZbaseA[busi]
        r3g = emt.fault_r[2] / Init_net_ZbaseA[busi]
        r12 = np.Inf
        r23 = emt.fault_r[4] / Init_net_ZbaseA[busi]
        r13 = np.Inf
    elif emt.fault_type == 6:
        r1g = emt.fault_r[0] / Init_net_ZbaseA[busi]
        r2g = np.Inf
        r3g = emt.fault_r[2] / Init_net_ZbaseA[busi]
        r12 = np.Inf
        r23 = np.Inf
        r13 = emt.fault_r[5] / Init_net_ZbaseA[busi]
    elif emt.fault_type == 7:
        r1g = np.Inf
        r2g = np.Inf
        r3g = np.Inf
        r12 = emt.fault_r[3] / Init_net_ZbaseA[busi]
        r23 = np.Inf
        r13 = np.Inf
    elif emt.fault_type == 8:
        r1g = np.Inf
        r2g = np.Inf
        r3g = np.Inf
        r12 = np.Inf
        r23 = emt.fault_r[4] / Init_net_ZbaseA[busi]
        r13 = np.Inf
    elif emt.fault_type == 9:
        r1g = np.Inf
        r2g = np.Inf
        r3g = np.Inf
        r12 = np.Inf
        r23 = np.Inf
        r13 = emt.fault_r[5] / Init_net_ZbaseA[busi]
    elif emt.fault_type == 10:
        r1g = np.Inf
        r2g = np.Inf
        r3g = np.Inf
        r12 = emt.fault_r[3] / Init_net_ZbaseA[busi]
        r23 = emt.fault_r[4] / Init_net_ZbaseA[busi]
        r13 = emt.fault_r[5] / Init_net_ZbaseA[busi]
    elif emt.fault_type == 11:
        r1g = emt.fault_r[0] / Init_net_ZbaseA[busi]
        r2g = emt.fault_r[1] / Init_net_ZbaseA[busi]
        r3g = emt.fault_r[2] / Init_net_ZbaseA[busi]
        r12 = emt.fault_r[3] / Init_net_ZbaseA[busi]
        r23 = emt.fault_r[4] / Init_net_ZbaseA[busi]
        r13 = emt.fault_r[5] / Init_net_ZbaseA[busi]
    else:
        assert(False)

    # yii abc  From bus
    Yft[0,0] = 1/((rt + Lt*1j)*emt.fault_dist) + Ct/2 * 1j
    Yft[1,1] = 1/((rt + Lt*1j)*emt.fault_dist) + Ct/2 * 1j
    Yft[2,2] = 1/((rt + Lt*1j)*emt.fault_dist) + Ct/2 * 1j

    # yjj abc  To bus
    Yft[3,3] = 1/((rt + Lt*1j)*(1-emt.fault_dist)) + Ct/2 * 1j
    Yft[4,4] = 1/((rt + Lt*1j)*(1-emt.fault_dist)) + Ct/2 * 1j
    Yft[5,5] = 1/((rt + Lt*1j)*(1-emt.fault_dist)) + Ct/2 * 1j

    # ykk abc Grounding point
    Yft[6,6] = 1/((rt + Lt*1j)*emt.fault_dist) + 1/((rt + Lt*1j)*(1-emt.fault_dist)) + 1/r1g + 1/r12 + 1/r13
    Yft[7,7] = 1/((rt + Lt*1j)*emt.fault_dist) + 1/((rt + Lt*1j)*(1-emt.fault_dist)) + 1/r2g + 1/r12 + 1/r23
    Yft[8,8] = 1/((rt + Lt*1j)*emt.fault_dist) + 1/((rt + Lt*1j)*(1-emt.fault_dist)) + 1/r3g + 1/r13 + 1/r23

    #
    Yft[6,7] = - 1/r12
    Yft[7,6] = - 1/r12
    Yft[6,8] = - 1/r13
    Yft[8,6] = - 1/r13
    Yft[7,8] = - 1/r23
    Yft[8,7] = - 1/r23

    # yik&yki abc
    Yft[0,6] = - 1/((rt + Lt*1j)*emt.fault_dist)
    Yft[6,0] = - 1/((rt + Lt*1j)*emt.fault_dist)
    Yft[1,7] = - 1/((rt + Lt*1j)*emt.fault_dist)
    Yft[7,1] = - 1/((rt + Lt*1j)*emt.fault_dist)
    Yft[2,8] = - 1/((rt + Lt*1j)*emt.fault_dist)
    Yft[8,2] = - 1/((rt + Lt*1j)*emt.fault_dist)

    # yjk&ykj abc
    Yft[3,6] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))
    Yft[6,3] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))
    Yft[4,7] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))
    Yft[7,4] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))
    Yft[5,8] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))
    Yft[8,5] = - 1/((rt + Lt*1j)*(1-emt.fault_dist))

    range1 = [0,1,2,3,4,5]
    range2 = [6,7,8]

    # Kron network reduction, also used in phasor ODE simulation
    Yftr = Yft[np.ix_(range1,range1)] - np.matmul(np.matmul(Yft[np.ix_(range1,range2)], np.linalg.inv(Yft[np.ix_(range2,range2)])), Yft[np.ix_(range2,range1)])

    # create G and New matrices
    pfd_ft = deepcopy(pfd)
    # pfd_ft.line_P = np.delete(pfd_ft.line_P, emt.fault_line_idx)   # Delete and make a copy for line tripping
    # pfd_ft.line_Q = np.delete(pfd_ft.line_Q, emt.fault_line_idx)
    pfd_ft.line_RX = np.delete(pfd_ft.line_RX, emt.fault_line_idx)
    pfd_ft.line_chg = np.delete(pfd_ft.line_chg, emt.fault_line_idx)
    pfd_ft.line_from = np.delete(pfd_ft.line_from, emt.fault_line_idx)
    pfd_ft.line_id = np.delete(pfd_ft.line_id, emt.fault_line_idx)
    pfd_ft.line_to = np.delete(pfd_ft.line_to, emt.fault_line_idx)


    ini_ft = Initialize(pfd_ft, dyd)
    ini_ft.InitNet(pfd_ft, dyd, emt.ts, emt.loadmodel_option)
    ini_ft.Init_net_G0 = sp.coo_matrix((ini_ft.Init_net_G0_data,
                                        (ini_ft.Init_net_G0_rows, ini_ft.Init_net_G0_cols)),
                                         shape=(ini_ft.Init_net_N, ini_ft.Init_net_N)
                                         ).tolil()

    ini_pf = deepcopy(ini_ft)

    Frombus = np.where(pfd_ft.bus_num == pfd.line_from[emt.fault_line_idx])[0][0]
    Tobus = np.where(pfd_ft.bus_num == pfd.line_to[emt.fault_line_idx])[0][0]

    leng = len(pfd_ft.bus_num)
    PF_Ln_add = []

    for i in range(6):
        for j in range(6):
            if i<=2:
                if j<=2:
                    From_idx = Frombus + i*leng  # aaa.., bbb..., ccc... of all From_bus
                    To_idx = Frombus + j*leng
                else:
                    From_idx = Frombus + i*leng
                    To_idx = Tobus + (j-3)*leng
            else:
                if j<=2:
                    From_idx = Tobus + (i-3)*leng
                    To_idx = Frombus + j*leng
                else:
                    From_idx = Tobus + (i-3)*leng
                    To_idx = Tobus + (j-3)*leng

            if i==j:
                gi0 = np.sum(Yftr[i])
                zi0 = 1/gi0
                temp_line = []
                if np.imag(zi0)>0:
                    if np.abs(np.real(zi0))>1e10:
                        temp_line = [From_idx, -1, np.inf, np.imag(zi0), 0]
                    else:
                        temp_line = [From_idx, -1, np.real(zi0), np.imag(zi0), 0]
                else:
                    if np.abs(1/np.real(gi0))>1e10:       
                        temp_line = [From_idx, -1, np.inf, 0, np.imag(gi0)]    
                    else:
                        temp_line = [From_idx, -1, 1/np.real(gi0), 0, np.imag(gi0)]  # R//C
            else:
                temp_line = []  
                if np.abs(Yftr[i,j])>1e-5:  # 
                    temp_line = [From_idx, To_idx, np.real(-1/Yftr[i,j]),np.imag(-1/Yftr[i,j]), 0]
            ## End if

            if j>=i:
                PF_Ln_add.append(temp_line)

    j = 0
    indx_ft = []  # 1 by 1 compare the first two of Init_net_coe0, to find differences
    for i in range(len(ini.Init_net_coe0)):
        # print(np.real([ini.Init_net_coe0[i,0],ini.Init_net_coe0[i,1]]))
        # print(np.real([ini_ft.Init_net_coe0[j,0],ini_ft.Init_net_coe0[j,1]]))
        if np.abs(ini.Init_net_coe0[i,0]-ini_ft.Init_net_coe0[j,0]) + np.abs(ini.Init_net_coe0[i,1]-ini_ft.Init_net_coe0[j,1])<1e-5:  # From and To bus
            pass
        else:
            ini_ft.Init_net_coe0 = np.insert(ini_ft.Init_net_coe0, j, np.zeros((1,9)), 0 )  # Give 0 values first
            indx_ft.append(j)
        j += 1


    # print(ini.Init_net_coe0)

    # print('\n\n\n')
    # print(ini_ft.Init_net_coe0)

    # ft_idx = ini_ft.resize_G0_data(len(PF_Ln_add))

    n = np.sum(np.array(PF_Ln_add)[:,1] == -1)  # self parameter, to ground
    m = len(PF_Ln_add) - n
    ft_idx = ini_ft.resize_G0_data(n + 4*m)  # expand to 66 with 0 for non-existing elements

    for i in range(len(PF_Ln_add)):
        Frombus = PF_Ln_add[i][0]
        Tobus = PF_Ln_add[i][1]

        R = PF_Ln_add[i][2]

        L = PF_Ln_add[i][3] / pfd.ws
        Rp = 20/3*2*L/emt.ts

        C = PF_Ln_add[i][4] / pfd.ws
        if np.abs(C)<1e-10:
            Rs = np.inf
        else:
            Rs = 0.15*emt.ts/2/C
        if L==0:
            Req = R
            icf = -1
            Gv1 = 1/R
        else:
            Req = (1+R*(emt.ts/2/L + 1/Rp)) / (emt.ts/2/L + 1/Rp)
            icf = (1-R*(emt.ts/2/L - 1/Rp)) / (1+R*(emt.ts/2/L + 1/Rp))
            Gv1 = (emt.ts/2/L - 1/Rp) / (1+R*(emt.ts/2/L + 1/Rp))

        if Tobus == -1:
            if np.abs(C)<1e-10:
                ini_ft.addtoG0(ft_idx, Frombus, Frombus, 1/R)  # ft_idx should not add, if this does not exist, it will be created
                temp = [Frombus, Tobus, Req, icf, Gv1, R, L, C, 0]
            else:
                ini_ft.addtoG0(ft_idx, Frombus, Frombus, 1/R + 1/(Rs+emt.ts/2/C))
                temp = [Frombus, Tobus, Rs+emt.ts/2/C, -(emt.ts/2/C - Rs)/(emt.ts/2/C + Rs), -1/(emt.ts/2/C + Rs), R, L, C, 0]
            ft_idx += 1
        else:
            ini_ft.addtoG0(ft_idx, Frombus, Frombus, 1/Req)
            ini_ft.addtoG0(ft_idx + 1, Tobus, Tobus, 1/Req)
            ini_ft.addtoG0(ft_idx + 2, Frombus, Tobus, -1/Req)
            ini_ft.addtoG0(ft_idx + 3, Tobus, Frombus, -1/Req)
            ft_idx += 4

            temp = [Frombus, Tobus, Req, icf, Gv1, R, L, C, 0]

        # print(ini_ft.Init_net_coe0)

        foundit = 0
        for j in indx_ft:
            if np.abs(ini.Init_net_coe0[j,0] - Frombus) + np.abs(ini.Init_net_coe0[j,1] - Tobus) < 1e-5:
                ini_ft.Init_net_coe0[j] = temp
                foundit = 1
                break
        if foundit == 0:
            np.append(ini_ft.Init_net_coe0, temp)

        # print(ini_ft.Init_net_coe0)

    # during-fault    # TODO: does this consider EPRI IBR model?
    N1 = len(pfd.bus_num)
    N2 = len(pfd.bus_num) * 2
    ft_idx = ini_ft.resize_G0_data(9 * len(pfd.bus_num))
    for i in range(len(pfd.gen_bus)):
        genbus_idx = int(np.where(pfd.bus_num == pfd.gen_bus[i])[0])
        ini_ft.addtoG0(ft_idx, genbus_idx, genbus_idx, ini.Init_mac_Gequiv[i,0,0])
        ini_ft.addtoG0(ft_idx + 1, genbus_idx, genbus_idx + N1, ini.Init_mac_Gequiv[i,0,1])
        ini_ft.addtoG0(ft_idx + 2, genbus_idx, genbus_idx + N2, ini.Init_mac_Gequiv[i,0,2])
        ini_ft.addtoG0(ft_idx + 3, genbus_idx + N1, genbus_idx, ini.Init_mac_Gequiv[i,1,0])
        ini_ft.addtoG0(ft_idx + 4, genbus_idx + N1, genbus_idx + N1, ini.Init_mac_Gequiv[i,1,1])
        ini_ft.addtoG0(ft_idx + 5, genbus_idx + N1, genbus_idx + N2, ini.Init_mac_Gequiv[i,1,2])
        ini_ft.addtoG0(ft_idx + 6, genbus_idx + N2, genbus_idx, ini.Init_mac_Gequiv[i,2,0])
        ini_ft.addtoG0(ft_idx + 7, genbus_idx + N2, genbus_idx + N1, ini.Init_mac_Gequiv[i,2,1])
        ini_ft.addtoG0(ft_idx + 8, genbus_idx + N2, genbus_idx + N2, ini.Init_mac_Gequiv[i,2,2])
        ft_idx += 9

    ini_ft.Init_net_G0 = sp.coo_matrix((ini_ft.Init_net_G0_data,
                                        (ini_ft.Init_net_G0_rows, ini_ft.Init_net_G0_cols)
                                        ),
                                       shape=(ini_ft.Init_net_N, ini_ft.Init_net_N)
                                       ).tolil()

    # post-fault        #If this line is tripped
    N1 = len(pfd.bus_num)
    N2 = len(pfd.bus_num) * 2
    pf_idx = ini_pf.resize_G0_data(9 * len(pfd.bus_num))
    for i in range(len(pfd.gen_bus)):
        genbus_idx = int(np.where(pfd.bus_num == pfd.gen_bus[i])[0])
        ini_pf.addtoG0(pf_idx, genbus_idx, genbus_idx, ini.Init_mac_Gequiv[i,0,0])
        ini_pf.addtoG0(pf_idx + 1, genbus_idx, genbus_idx + N1, ini.Init_mac_Gequiv[i,0,1])
        ini_pf.addtoG0(pf_idx + 2, genbus_idx, genbus_idx + N2, ini.Init_mac_Gequiv[i,0,2])
        ini_pf.addtoG0(pf_idx + 3, genbus_idx + N1, genbus_idx, ini.Init_mac_Gequiv[i,1,0])
        ini_pf.addtoG0(pf_idx + 4, genbus_idx + N1, genbus_idx + N1, ini.Init_mac_Gequiv[i,1,1])
        ini_pf.addtoG0(pf_idx + 5, genbus_idx + N1, genbus_idx + N2, ini.Init_mac_Gequiv[i,1,2])
        ini_pf.addtoG0(pf_idx + 6, genbus_idx + N2, genbus_idx, ini.Init_mac_Gequiv[i,2,0])
        ini_pf.addtoG0(pf_idx + 7, genbus_idx + N2, genbus_idx + N1, ini.Init_mac_Gequiv[i,2,1])
        ini_pf.addtoG0(pf_idx + 8, genbus_idx + N2, genbus_idx + N2, ini.Init_mac_Gequiv[i,2,2])
        pf_idx += 9
    ini_pf.Init_net_G0 = sp.coo_matrix((ini_pf.Init_net_G0_data, (ini_pf.Init_net_G0_rows, ini_pf.Init_net_G0_cols)),
                                         shape=(ini_pf.Init_net_N, ini_pf.Init_net_N)
                                         ).tolil()

    ini.Init_net_G1 = deepcopy(ini_ft.Init_net_G0)
    ini.Init_net_coe1 = deepcopy(ini_ft.Init_net_coe0)
    if mode == 'inv':
        ini.Init_net_G1_inv = la.inv(ini.Init_net_G1.tocsc())
    elif mode == 'lu':
        ini.Init_net_G1_lu = la.splu(ini.Init_net_G1.tocsc())
    elif mode == 'bbd':
        pass
    else:
        raise ValueError('Unrecognized mode: {}'.format(mode))


    if emt.fault_tripline == 0:  # if temporary grounding fault, then use original G
        ini.Init_net_G2 = deepcopy(ini.Init_net_G0)
        # ini.Init_net_G2_cols = deepcopy(ini.Init_net_G0_cols)
        # ini.Init_net_G2_data = deepcopy(ini.Init_net_G0_data)
        # ini.Init_net_G2_rows = deepcopy(ini.Init_net_G0_rows)
        ini.Init_net_coe2 = deepcopy(ini.Init_net_coe0)
    else:   # if trip a line, use post fault G
        ini.Init_net_G2 = deepcopy(ini_pf.Init_net_G0)
        # ini.Init_net_G2_cols = deepcopy(ini_pf.Init_net_G0_cols)
        # ini.Init_net_G2_data = deepcopy(ini_pf.Init_net_G0_data)
        # ini.Init_net_G2_rows = deepcopy(ini_pf.Init_net_G0_rows)
        ini.Init_net_coe2 = deepcopy(ini_pf.Init_net_coe0)

    if mode == 'inv':
        ini.Init_net_G2_inv = la.inv(ini.Init_net_G2.tocsc())
    elif mode == 'lu':
        ini.Init_net_G2_lu = la.splu(ini.Init_net_G2.tocsc())
    elif mode == 'bbd':
        pass
    else:
        raise ValueError('Unrecognized mode: {}'.format(mode))

    return


def modify_system(EMT_N, pfd, ini, record4cosim, emt_zones, Gd, Go):
    if record4cosim == False and len(emt_zones) >0:
        bus_rec = []
        nbus = len(pfd.bus_num)

        for i in range(len(emt_zones[EMT_N-1]['bus_b'])):
            busb = int(emt_zones[EMT_N-1]['bus_b'][i])
            bus_rec = np.append(bus_rec, busb)
        
        for i in range(len(bus_rec)): 
            idx = np.where(pfd.bus_num==bus_rec[i])[0][0]
            ini.addtoG0(       idx,        idx, Gd)
            ini.addtoG0(  idx+nbus,   idx+nbus, Gd)
            ini.addtoG0(idx+2*nbus, idx+2*nbus, Gd)
            ini.addtoG0(       idx,   idx+nbus, Go)
            ini.addtoG0(  idx+nbus,        idx, Go)
            ini.addtoG0(       idx, idx+2*nbus, Go)
            ini.addtoG0(idx+2*nbus,        idx, Go)
            ini.addtoG0(idx+2*nbus,   idx+nbus, Go)
            ini.addtoG0(  idx+nbus, idx+2*nbus, Go)

        ini.Init_net_G0 = sp.coo_matrix((ini.Init_net_G0_data,
                                            (ini.Init_net_G0_rows, ini.Init_net_G0_cols)),
                                            shape=(ini.Init_net_N, ini.Init_net_N)
                                            ).tolil()

        if ini.admittance_mode == 'inv':
            ini.Init_net_G0_inv = la.inv(ini.Init_net_G0)
        elif ini.admittance_mode == 'lu':
            ini.Init_net_G0_lu = la.splu(ini.Init_net_G0.tocsc())
        elif ini.admittance_mode == 'bbd':
            pass
        else:
            raise ValueError('Unrecognized mode: {}'.format(mode))

    ## End if

    return
