import pickle
import numpy as np
import time
import os
from collections import deque

from paraemt.states import States

from paraemt.states_ibr import States_ibr

from paraemt.lib_numba import *

# import helics as h

from ctypes import *


from paraemt.Python_C_DLL_wrapper import wrap_function, MODELINSTANCE, MODELINFO

shared_lib_path = os.getcwd() + "\\models\\ibrepri.dll"
# EPRI's IBR model
add_lib = CDLL(shared_lib_path)  # Min
Model_GetInfo = wrap_function(add_lib, "Model_GetInfo", POINTER(MODELINFO), None)
Model_Outputs = wrap_function(add_lib, "Model_Outputs", c_int, [POINTER(MODELINSTANCE)])

info = Model_GetInfo()
num_in_ports, num_out_ports, num_param = (
    info.contents.cNumInputPorts,
    info.contents.cNumOutputPorts,
    info.contents.cNumParameters,
)
num_int_states, num_float_states, num_double_states = (
    info.contents.cNumIntStates,
    info.contents.cNumFloatStates,
    info.contents.cNumDoubleStates,
)

import pandas as pd

alpha = np.exp(1j * 2 * np.pi / 3)
Ainv = (
    np.asarray([[1, 1, 1], [1, alpha * alpha, alpha], [1, alpha, alpha * alpha]]) / 3.0
)

#### To enable use of EPRI IBR Model set EPRI_IBR environment variable to 1 ####
if "EPRI_IBR" in os.environ.keys():
    EPRI_IBR = int(os.environ["EPRI_IBR"])
else:
    EPRI_IBR = False

if EPRI_IBR:
    from paraemt.Python_C_DLL_wrapper import *

    Model_GetInfo = wrap_function(add_lib, "Model_GetInfo", POINTER(MODELINFO), None)
    Model_Outputs = wrap_function(
        add_lib, "Model_Outputs", c_int, [POINTER(MODELINSTANCE)]
    )

    info = Model_GetInfo()
    num_in_ports = info.contents.cNumInputPorts
    num_out_ports = info.contents.cNumOutputPorts
    num_param = info.contents.cNumParameters
    num_int_states = info.contents.cNumIntStates
    num_float_states = info.contents.cNumFloatStates
    num_double_states = info.contents.cNumDoubleStates


# EMT sim
class EmtSimu:
    @staticmethod
    def initialize_from_snp(input_snp, netMod):
        with open(input_snp, "rb") as f:
            pfd, dyd, ini, emt = pickle.load(f)

            emt.t = [0.0]
            emt.save_idx = 0
            # emt.v = {}
            # emt.v[0] = x

            # ini.Init_net_G0 = sp.coo_matrix((ini.Init_net_G0_data,
            #                                  (ini.Init_net_G0_rows, ini.Init_net_G0_cols)),
            #                                 shape=(ini.Init_net_N, ini.Init_net_N)
            #                             ).tolil()

            # if netMod == 'inv':
            #     ini.Init_net_G0_inv = la.inv(ini.Init_net_G0.tocsc())
            # elif netMod == 'lu':
            #     ini.Init_net_G0_lu = la.splu(ini.Init_net_G0.tocsc())
            # elif netMod == 'bbd':
            #     pass
            # else:
            #     raise ValueError('Unrecognized mode: {}'.format(netMod))
            # ini.admittance_mode = netMod

            ini.CalcGnGinv(netMod)

            # emt.ibr_epri = [MODELINSTANCE() for jj in range(dyd.ibr_epri_n)]
            if EPRI_IBR:
                ini.InitIbrepri(pfd, dyd)
            # emt._init_ibr_epri()

            emt.pfd = pfd
            emt.ini = ini
            emt.dyd = dyd

        ## End with

        return emt

    def __init__(self, ngen, nibr, nbus, nload, nibrepri, save_rate):
        # three-phase synchronous machine model, unit in Ohm
        self.EMT_N = 1  # TODO: fix this hack and delete it
        self.ts = 50e-6  # second
        self.Tlen = 0.1  # second
        self.Nlen = np.asarray([])

        self.ngen = ngen
        self.nibr = nibr
        self.nibrepri = nibrepri
        self.nbus = nbus
        self.nload = nload

        self.save_idx = 0
        self.save_rate = save_rate

        self.t = {}
        self.x = {}
        self.x_pv_1 = []
        self.x_pred = {}
        self.x_ibr = {}

        self.x_ibr_pv_1 = []
        self.x_ibr_epri = {}
        self.x_ibr_epri_pv_1 = []
        self.x_load = {}
        self.x_load_pv_1 = []
        self.x_bus = {}
        self.x_bus_pv_1 = []

        self.v = {}
        self.i = {}
        self.i_branch = {}
        self.Ginv = []
        self.Glu = []
        self.net_coe = []

        self.ibr_epri = []

        self.xp = States(
            ngen
        )  # seems not necessary, try later and see if they can be deleted
        self.xp_ibr = States_ibr(nibr)
        self.Igs = np.zeros(3 * (nbus + nibrepri))
        self.Isg = np.zeros(3 * ngen)
        self.Igi = np.zeros(3 * (nbus + nibrepri))
        self.Igi_epri = np.zeros(3 * (nbus + nibrepri))
        self.Il = np.zeros(3 * (nbus + nibrepri))  # to change to Igl and Iload
        self.Ild = np.zeros(3 * nload)
        self.Iibr = np.zeros(3 * nibr)
        self.Iibr_epri = np.zeros(3 * nibrepri)
        self.brch_Ihis = np.asarray([])
        self.brch_Ipre = np.asarray([])
        self.node_Ihis = np.asarray([])

        # self.I_RHS = np.zeros(3*(nbus+nibrepri))
        self.Vsol = None  # np.zeros(3*nbus)
        # self.Vsol_1 = np.zeros(3*nbus)

        self.compute_phasor = 1
        self.fft_vabc = deque()
        self.fft_T = 1
        self.fft_N = 0
        self.fft_vma = np.zeros(2 * 3 * (nbus + nibrepri))
        self.fft_vpn0 = np.zeros(2 * 3 * (nbus + nibrepri))

        self.fft_iabc = deque()
        self.fft_ima = np.zeros(6)
        self.fft_ipn0 = np.zeros(6)

        # phasor signals on the GridPack-ParaEMT interface
        self.Sinj = 0.0
        self.Pinj = 0.0
        self.Qinj = 0.0

        # self.Vm_itfc = 1.0
        # self.Va_itfc = 0.0

        self.theta = np.zeros(ngen)
        self.ed_mod = np.zeros(ngen)
        self.eq_mod = np.zeros(ngen)

        self.t_release_f = 0.1
        self.loadmodel_option = 1  # 1-const rlc, 2-const z

        # step change
        self.t_sc = 1000  # the time when the step change occurs
        self.i_gen_sc = 1  # which gen, index in pfd.gen_bus
        self.flag_exc_gov = 1  # 0 - exc, 1 - gov
        self.dsp = -0.2  # increment
        self.flag_sc = 1  # 1 - step change to be implemented, 0 - step change completed

        # gen trip
        self.t_gentrip = 1000  # the time when the gentrip occurs
        self.i_gentrip = 1  # which gen, index in pfd.gen_bus
        self.flag_gentrip = 1  # 1 - gentrip to be implemented, 0 - gentrip completed
        self.flag_reinit = 1  # 1 - re-init to be implemented, 0 - re-init completed

        # fault
        self.fault_t = 1000
        self.fault_line_idx = 0
        self.fault_tlen = 0
        self.fault_dist = 0
        self.fault_type = 11
        self.fault_r = [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf]
        self.fault_tripline = (
            0  # 0 - no line tripping, 1 - line tripped upon fault clearance
        )
        # Bus fault
        self.busfault_t = 1
        self.fault_bus_idx = 2
        self.busfault_tlen = 5/60 # 5 cycles
        self.busfault_type = 1 # Check psutils for fault types
        self.busfault_r = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
        self.add_line_num=0
        self.fault_line_idx = 0
        self.fault_tripline = 0
        self.bus_del_ind=[]  #bus delete index

        # ref at last time step (for calculating dref term)
        self.vref = np.zeros(ngen)
        self.vref_1 = np.zeros(ngen)
        self.gref = np.zeros(ngen)

        # playback
        self.data = []
        self.playback_enable = 0
        self.playback_t_chn = 1
        self.playback_sig_chn = 2
        self.playback_tn = 0

        # mac as I source
        self.flag_Isrc = 0

        # simulate an EMT area
        self.buses = []
        self.branches = []

        self.rank = 0
        self.gen_range = np.array([0, ngen]).reshape(2, 1)
        self.gen_counts = np.array([ngen])
        self.ibr_range = np.array([0, nibr]).reshape(2, 1)
        self.ibr_counts = np.array([nibr])
        self.ebr_range = np.array([0, nibrepri]).reshape(2, 1)
        self.ebr_counts = np.array([nibrepri])
        self.load_range = np.array([0, nload]).reshape(2, 1)
        self.load_counts = np.array([nload])
        self.bus_range = np.array([0, nbus]).reshape(2, 1)
        self.bus_counts = np.array([nbus])

        self.use_helics = False

        # #### Temporary timing attributes ####
        # ## PredictX ##
        # self.logic = 0.0
        # self.px_numba = 0.0
        # self.predx_total = 0.0
        # # ## UpdateX ##
        # # self.ux_numba = 0.0
        # # self.ux_gather = 0.0
        # # self.ux_total = 0.0
        # # ## UpdateXibr ##
        # # self.uxi_numba = 0.0
        # # self.uxi_gather = 0.0
        # # self.uxi_total = 0.0

        return

    def init_ibr_epri(self):
        dyd = self.dyd
        ini = self.ini
        pfd = self.pfd

        self.ibr_epri = [MODELINSTANCE() for jj in range(self.nibrepri)]

        for i in range(self.nibrepri):
            # ibri = MODELINSTANCE()
            # self.ibr_epri.append(ibri)
            ibri = self.ibr_epri[i]

            ibrbus = dyd.ibr_epri_bus[i]
            ibrid = dyd.ibr_epri_id[i]
            # if len(ibrid)==1:
            #     ibrid = ibrid + ' '

            bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]

            # ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0][0]
            ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0]
            ibrid_idx = np.where(pfd.ibr_id[ibrbus_idx] == ibrid)[0][0]
            ibrbus_idx = ibrbus_idx[ibrid_idx]

            # Set initial inputs, outputs, parameters, and states
            (kVbase, IBR_MVA_base, fbase, Vdcbase) = (
                pfd.bus_basekV[bus_idx],
                pfd.ibr_MVA_base[ibrbus_idx],
                60.0,
                dyd.ibr_epri_Vdcbase[i],
            )

            kAbase = IBR_MVA_base / kVbase / np.sqrt(3.0)

            (Imax, KiI, KiP, KiPLL, KiQ, KpI, KpP, KpPLL, KpQ, Pqflag, Vdip, Vup) = (
                dyd.ibr_epri_Imax[i],
                dyd.ibr_epri_KiI[i],
                dyd.ibr_epri_KiP[i],
                dyd.ibr_epri_KiPLL[i],
                dyd.ibr_epri_KiQ[i],
                dyd.ibr_epri_KpI[i],
                dyd.ibr_epri_KpP[i],
                dyd.ibr_epri_KpPLL[i],
                dyd.ibr_epri_KpQ[i],
                dyd.ibr_epri_Pqflag[i],
                dyd.ibr_epri_Vdip[i],
                dyd.ibr_epri_Vup[i],
            )

            (Cfilt, Lchoke, Rchoke, Rdamp) = (
                dyd.ibr_epri_Cfilt[i],
                dyd.ibr_epri_Lchoke[i],
                dyd.ibr_epri_Rchoke[i],
                dyd.ibr_epri_Rdamp[i],
            )

            # IBR time step
            ts_ibr = self.ts

            Inputs = [
                kVbase * np.sqrt(2.0 / 3.0) * ini.Init_ibr_epri_Va[i],
                kVbase * np.sqrt(2.0 / 3.0) * ini.Init_ibr_epri_Vb[i],
                kVbase * np.sqrt(2.0 / 3.0) * ini.Init_ibr_epri_Vc[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_Ia[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_Ib[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_Ic[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_IaL1[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_IbL1[i],
                kAbase * np.sqrt(2.0) * ini.Init_ibr_epri_IcL1[i],
                ini.Init_ibr_epri_Pref[i],
                ini.Init_ibr_epri_Qref[i],
                ini.Init_ibr_epri_Vd[i],
            ]  # with PQ ref initialized at P3
            Outputs = [
                ini.Init_ibr_epri_Ea[i],
                ini.Init_ibr_epri_Eb[i],
                ini.Init_ibr_epri_Ec[i],
                ini.Init_ibr_epri_Idref[i],
                ini.Init_ibr_epri_Idref[i],
                ini.Init_ibr_epri_Iqref[i],
                ini.Init_ibr_epri_Iqref[i],
                ini.Init_ibr_epri_Vd[i],
                ini.Init_ibr_epri_Vq[i],
                60.0,
                ini.Init_ibr_epri_Pref[i],
                ini.Init_ibr_epri_Qref[i],
            ]  # Updated by DLL

            Parameters = [
                kVbase,
                IBR_MVA_base,
                Vdcbase,
                KpI,
                KiI,
                KpPLL,
                KiPLL,
                KpP,
                KiP,
                KpQ,
                KiQ,
                Imax,
                Pqflag,
                Vdip,
                Vup,
                Rchoke,
                Lchoke,
                Cfilt,
                Rdamp,
                ts_ibr,
            ]
            IntSt = []
            FloatSt = []
            DoubleSt = [
                0.0,
                0.0,
                ini.Init_ibr_epri_thetaPLL[i],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ini.Init_ibr_epri_Id[i],
                0.0,
                ini.Init_ibr_epri_Iq[i],
            ]  # Updated by DLL

            # Define object of model. One object per instance of the model used in simulation
            # ibri = emt.ibr_epri[i]
            ibri.cExternalInputs = (c_double * num_in_ports)(*Inputs)
            ibri.cExternalOutputs = (c_double * num_out_ports)(*Outputs)
            ibri.cParameters = (c_double * num_param)(*Parameters)
            # Should be updated at each time step
            ibri.cTime = 0.0
            ibri.cIntStates = (c_int * num_int_states)(*IntSt)
            ibri.cFloatStates = (c_float * num_float_states)(*FloatSt)
            ibri.cDoubleStates = (c_double * num_double_states)(*DoubleSt)

            # Call function to check parameter values. Not critical for initial test run of model use
            # print ("Check Parameters")
            Model_CheckParameters = wrap_function(
                add_lib, "Model_CheckParameters", c_int, [POINTER(MODELINSTANCE)]
            )
            return_int = Model_CheckParameters(ibri)
            self._check_epri_return_status(return_int)

            # Call function to initialize states of the model. Needs appropriate code in the DLL
            # print ("Model Initialization")
            Model_Initialize = wrap_function(
                add_lib, "Model_Initialize", c_int, [POINTER(MODELINSTANCE)]
            )
            return_int = Model_Initialize(ibri)
            self._check_epri_return_status(return_int)

        ## End for

        return

    def _check_epri_return_status(self, return_int):
        if return_int == 0:
            pass
        else:
            raise RuntimeError("Issue with EPRI IBR model!")
        ## End if

        return

    def Record4CoSim(self, record4cosim, brch_Ipre, Vsol, t):
        if record4cosim == False and len(self.emt_zones) > 0:
            # TODO: # load recorded signals
            data_csv = (
                "record_i_int.csv"  # Min Deleted 01302024, for FULL EMT simulation
            )
            with open(data_csv, "rb") as csvfile:
                self.data = pd.read_csv(csvfile, sep=",", header=None).to_numpy()
                print("Playback current(s) injected to boundary bus(es).\n")

            current_rec = []
            voltage_rec = []
            bus_rec = []

        if record4cosim == True and len(self.emt_zones) > 0:
            current_rec = [t]
            voltage_rec = [t]
            bus_rec = []
            nbus = len(self.pfd.bus_num)
            for i in range(len(self.emt_zones)):
                for j in range(len(self.emt_zones[i]["bus_b"])):
                    busb = int(self.emt_zones[i]["bus_b"][j])
                    bus_rec.append(busb)

                    # current: find branch and add current
                    # find all branches
                    InjBranch = np.array([])
                    bus_emt = self.emt_zones[i]["buses"]
                    # line
                    idx1 = np.where(self.pfd.line_from == busb)[0]
                    for k in range(len(idx1)):
                        bust = self.pfd.line_to[idx1[k]]
                        idx2 = np.where(bus_emt == bust)[0]
                        if len(idx2) != 0:  # in emt zone
                            # check if identified branch is already in InjBranch
                            if len(InjBranch) > 0:
                                flag = 0
                                for ii in range(len(InjBranch)):
                                    if any(
                                        [
                                            all(InjBranch[ii] == [busb, bust]),
                                            all(InjBranch[ii] == [bust, busb]),
                                        ]
                                    ):
                                        flag = 1
                                if flag == 0:
                                    InjBranch = np.append(
                                        InjBranch,
                                        [
                                            np.where(self.pfd.bus_num == busb)[0][0],
                                            np.where(self.pfd.bus_num == bust)[0][0],
                                            1,
                                        ],
                                    )  # third entry: 1-line, 0-xfmr
                            else:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        1,
                                    ],
                                )

                    idx1 = np.where(self.pfd.line_to == busb)[0]
                    for k in range(len(idx1)):
                        bust = self.pfd.line_from[idx1[k]]
                        idx2 = np.where(bus_emt == bust)[0]
                        if len(idx2) != 0:  # in emt zone
                            # check if identified branch is already in InjBranch
                            if len(InjBranch) > 0:
                                flag = 0
                                for ii in range(len(InjBranch)):
                                    if any(
                                        [
                                            all(InjBranch[ii] == [busb, bust]),
                                            all(InjBranch[ii] == [bust, busb]),
                                        ]
                                    ):
                                        flag = 1
                                if flag == 0:
                                    InjBranch = np.append(
                                        InjBranch,
                                        [
                                            np.where(self.pfd.bus_num == busb)[0][0],
                                            np.where(self.pfd.bus_num == bust)[0][0],
                                            1,
                                        ],
                                    )
                            else:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        1,
                                    ],
                                )
                    # xfmr
                    idx1 = np.where(self.pfd.xfmr_from == busb)[0]
                    for k in range(len(idx1)):
                        bust = self.pfd.xfmr_to[idx1[k]]
                        idx2 = np.where(bus_emt == bust)[0]
                        if len(idx2) != 0:  # in emt zone
                            # check if identified branch is already in InjBranch
                            if len(InjBranch) > 0:
                                flag = 0
                                for ii in range(len(InjBranch)):
                                    if any(
                                        [
                                            all(InjBranch[ii] == [busb, bust]),
                                            all(InjBranch[ii] == [bust, busb]),
                                        ]
                                    ):
                                        flag = 1
                                if flag == 0:
                                    InjBranch = np.append(
                                        InjBranch,
                                        [
                                            np.where(self.pfd.bus_num == busb)[0][0],
                                            np.where(self.pfd.bus_num == bust)[0][0],
                                            0,
                                        ],
                                    )
                            else:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        0,
                                    ],
                                )

                    idx1 = np.where(self.pfd.xfmr_to == busb)[0]
                    for k in range(len(idx1)):
                        bust = self.pfd.xfmr_from[idx1[k]]
                        idx2 = np.where(bus_emt == bust)[0]
                        if len(idx2) != 0:  # in emt zone
                            # check if identified branch is already in InjBranch
                            if len(InjBranch) > 0:
                                flag = 0
                                for ii in range(len(InjBranch)):
                                    if any(
                                        [
                                            all(InjBranch[ii] == [busb, bust]),
                                            all(InjBranch[ii] == [bust, busb]),
                                        ]
                                    ):
                                        flag = 1
                                if flag == 0:
                                    InjBranch = np.append(
                                        InjBranch,
                                        [
                                            np.where(self.pfd.bus_num == busb)[0][0],
                                            np.where(self.pfd.bus_num == bust)[0][0],
                                            0,
                                        ],
                                    )
                            else:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        0,
                                    ],
                                )

                    lenIB = int(len(InjBranch) / 3)
                    # print([self.pfd.bus_num[int(InjBranch[:][0])], self.pfd.bus_num[int(InjBranch[:][1])]])
                    ia = 0
                    ib = 0
                    ic = 0
                    for k in range(lenIB):
                        # print([self.pfd.bus_num[int(InjBranch[3*k])],self.pfd.bus_num[int(InjBranch[3*k+1])]])
                        idx1 = np.where(
                            self.ini.Init_net_coe0[:, 0] == InjBranch[3 * k]
                        )[0]
                        idx2 = np.where(
                            self.ini.Init_net_coe0[idx1, 1] == InjBranch[3 * k + 1]
                        )[0]
                        len2 = len(idx2)
                        if len2 > 0:
                            for ii in range(len2):
                                idx = idx1[idx2[ii]]
                                # print([idx,-brch_Ipre[idx  ] - brch_Ipre[idx+3]*InjBranch[3*k+2]])
                                ia = (
                                    ia
                                    - brch_Ipre[idx]
                                    - brch_Ipre[idx + 3] * InjBranch[3 * k + 2]
                                )
                                ib = (
                                    ib
                                    - brch_Ipre[idx + 1]
                                    - brch_Ipre[idx + 4] * InjBranch[3 * k + 2]
                                )
                                ic = (
                                    ic
                                    - brch_Ipre[idx + 2]
                                    - brch_Ipre[idx + 5] * InjBranch[3 * k + 2]
                                )

                        idx1 = np.where(
                            self.ini.Init_net_coe0[:, 1] == InjBranch[3 * k]
                        )[0]
                        idx2 = np.where(
                            self.ini.Init_net_coe0[idx1, 0] == InjBranch[3 * k + 1]
                        )[0]
                        len2 = len(idx2)
                        if len2 > 0:
                            for ii in range(len2):
                                idx = idx1[idx2[ii]]
                                # print(-brch_Ipre[idx  ] + brch_Ipre[idx+6]*InjBranch[3*k+2])
                                # print(brch_Ipre[idx  ] - brch_Ipre[idx+6]*InjBranch[3*k+2])
                                ia = (
                                    ia
                                    - brch_Ipre[idx]
                                    + brch_Ipre[idx + 6] * InjBranch[3 * k + 2]
                                )
                                ib = (
                                    ib
                                    - brch_Ipre[idx + 1]
                                    + brch_Ipre[idx + 7] * InjBranch[3 * k + 2]
                                )
                                ic = (
                                    ic
                                    - brch_Ipre[idx + 2]
                                    + brch_Ipre[idx + 8] * InjBranch[3 * k + 2]
                                )

                        # idx1 = np.where(self.ini.Init_net_coe0[:,0] == InjBranch[2*k])[0]
                        # idx2 = np.where(self.ini.Init_net_coe0[idx1,1] == InjBranch[2*k+1])[0]
                        # len2 = len(idx2)
                        # if len2>0:
                        #     for ii in range(len2):
                        #         idx = idx1[idx2[ii]]
                        #         print(-brch_Ipre[idx  ] - brch_Ipre[idx+3])
                        #         ia = ia - brch_Ipre[idx  ] - brch_Ipre[idx+3]
                        #         ib = ib - brch_Ipre[idx+1] - brch_Ipre[idx+4]
                        #         ic = ic - brch_Ipre[idx+2] - brch_Ipre[idx+5]

                        # idx1 = np.where(self.ini.Init_net_coe0[:,1] == InjBranch[2*k])[0]
                        # idx2 = np.where(self.ini.Init_net_coe0[idx1,0] == InjBranch[2*k+1])[0]
                        # len2 = len(idx2)
                        # if len2>0:
                        #     for ii in range(len2):
                        #         idx = idx1[idx2[ii]]
                        #         print(brch_Ipre[idx  ] - brch_Ipre[idx+6])
                        #         ia = ia + brch_Ipre[idx  ] - brch_Ipre[idx+6]
                        #         ib = ib + brch_Ipre[idx+1] - brch_Ipre[idx+7]
                        #         ic = ic + brch_Ipre[idx+2] - brch_Ipre[idx+8]
                    current_temp = [ia, ib, ic]
                    current_rec = current_rec + current_temp

                    # voltage: find bus and add volt
                    bus_idx = np.where(self.pfd.bus_num == busb)[0][0]
                    voltage_temp = [
                        Vsol[bus_idx],
                        Vsol[bus_idx + nbus],
                        Vsol[bus_idx + 2 * nbus],
                    ]

                    voltage_rec = voltage_rec + voltage_temp
        if len(self.emt_zones) == 0:
            current_rec = []
            voltage_rec = []
            bus_rec = []

        return bus_rec, current_rec, voltage_rec

    def preprocess(self, ini, pfd, dyd):
        self.t = [0.0]

        nbus = self.nbus
        nebr = self.nibrepri

        ini.CombineX(pfd, dyd)
        self.x[0] = ini.Init_x.copy()
        self.x_ibr[0] = ini.Init_x_ibr.copy()
        self.x_ibr_epri[0] = ini.Init_x_ibr_epri.copy()
        self.x_bus[0] = ini.Init_x_bus.copy()
        self.x_load[0] = ini.Init_x_load.copy()
        self.x_pv_1 = ini.Init_x.copy()
        self.x_ibr_pv_1 = ini.Init_x_ibr.copy()
        self.x_ibr_epri_pv_1 = ini.Init_x_ibr_epri.copy()
        self.x_bus_pv_1 = ini.Init_x_bus.copy()
        self.x_load_pv_1 = ini.Init_x_load.copy()

        self.Vsol = np.real(ini.Init_net_Vt)
        self.Vsol_1 = np.real(ini.Init_net_Vt)
        self.v[0] = self.Vsol.copy()

        self.i[0] = np.real(ini.Init_net_It)

        if self.compute_phasor == 1:
            self.fft_vabc.append(np.real(ini.Init_net_Vt))
            if self.fft_T == 1:
                self.fft_N = int(1 / (pfd.ws / 2 / np.pi) / self.ts)
            ## End if

            nstep = nbus + nebr
            self.fft_vma[0 : 3 * nstep] = abs(ini.Init_net_Vt)
            self.fft_vma[3 * nstep : 6 * nstep] = np.angle(ini.Init_net_Vt)
            self.fft_vpn0[0:nstep] = abs(ini.Init_net_Vt[0:nstep])
            self.fft_vpn0[nstep : 3 * nstep] = np.zeros(2 * nstep)
            self.fft_vpn0[3 * nstep : 4 * nstep] = np.angle(ini.Init_net_Vt)[0:nstep]
            self.fft_vpn0[4 * nstep : 6 * nstep] = np.zeros(2 * nstep)

            if len(self.emt_zones) > 0:
                self.fft_iabc.append(
                    np.zeros(3 * len(self.emt_zones[self.EMT_N - 1]["bus_b"]))
                )
            self.fft_ima = np.zeros(
                6 * nbus
            )  # set to zeros for simplicity, should be initialized based on steady state condition. same for the following
            self.fft_ipn0 = np.zeros(6 * nbus)

        ## End if

        self.x_pred = {0: self.x[0], 1: self.x[0], 2: self.x[0]}
        # self.x_pred = {0: self.x[0], 1: self.x[0]}

        self.brch_Ihis = ini.Init_brch_Ihis.copy()
        self.brch_Ipre = ini.Init_brch_Ipre.copy()
        self.node_Ihis = ini.Init_node_Ihis.copy()
        term=self.brch_Ipre.copy()
        term2=9*len(pfd.line_from)
        self.i_branch[0]=np.concatenate((term[0:term2:9], term[1:term2:9], term[2:term2:9]))  

        self.brch_range = np.array([0, len(self.brch_Ihis)]).reshape(2, 1)
        self.brch_counts = np.array([self.brch_range.size])

        self.vref = ini.Init_mac_vref.copy()
        self.vref_1 = ini.Init_mac_vref.copy()
        self.gref = ini.Init_mac_gref.copy()

        ginfo = """
        ---- Grid Info ----
        Generators: {:d}
        WECC IBRs:  {:d}
        EPRI IBRs:  {:d}
        Buses:      {:d}
        Branches:   {:d}
        """.format(
            self.ngen,
            self.nibr,
            self.nibrepri,
            self.nbus,
            len(pfd.line_id) + len(pfd.xfmr_id),
        )
        print(ginfo)

        return

    def helics_setup(self):
        helics = {
            "deltat": 0.00001,
            "subscription_topic": [
                "gridpack_i_9_10_real",
                "gridpack_i_9_10_imag",
                "gridpack_v_9_10_real",
                "gridpack_v_9_10_imag",
            ],
            "publication_topic": ["emt_Pinj_9_10", "emt_Qinj_9_10"],
            "endpoints": [],
            "publication_interval": 0.00001,
            "endpoint_interval": 0.00001,
        }
        self.currenttime = 0.0
        name_helics = "EMT"
        self.deltat = helics["deltat"]
        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetCoreName(fedinfo, name_helics)
        h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
        fedinitstring = "--federates=1  --brokerport={} --broker_address={} ".format(
            32000, "127.0.0.1"
        )
        # fedinitstring = "--federates=1  --broker_address={} ".format( broker_ip)
        # fedinitstring = "--federates=1  --broker_address='0:0' "
        h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, self.deltat
        )
        cfed = h.helicsCreateValueFederate(name_helics, fedinfo)
        helics = {
            "deltat": 0.00001,
            "subscription_topic": [
                "gridpack_i_9_10_real",
                "gridpack_i_9_10_imag",
                "gridpack_v_9_10_real",
                "gridpack_v_9_10_imag",
            ],
            "publication_topic": ["emt_Pinj_9_10", "emt_Qinj_9_10"],
            "endpoints": [],
            "publication_interval": 0.00001,
            "endpoint_interval": 0.00001,
        }
        self.pub = {}
        for x in helics["publication_topic"]:
            self.pub[x] = h.helicsFederateRegisterGlobalTypePublication(
                cfed, x, "double", ""
            )
        self.sub = {}
        for x in helics["subscription_topic"]:
            self.sub[x] = h.helicsFederateRegisterSubscription(cfed, x, "double")
        ends = {}
        for x in helics["endpoints"]:
            ends[x] = h.helicsFederateRegisterGlobalEndpoint(fed=cfed, name=x)
        h.helicsFederateEnterExecutingMode(cfed)

        Gd = 100
        Go = 0.001

        self.Gitfc = np.asarray([[Gd, Go, Go], [Go, Gd, Go], [Go, Go, Gd]])
        self.Ritfc = np.linalg.inv(self.Gitfc)

        self.cfed = cfed
        self.use_helics = True

        return

    def predictX(self,pfd,dyd,ts):
        # t1 = time.time()

        xlen = len(self.x)
        x_pv_1 = self.x_pv_1
        if xlen == 1:
            x_pv_2 = x_pv_1
            x_pv_3 = x_pv_1
        else:
            x_pv_2 = self.x_pred[1]
            if xlen == 2:
                x_pv_3 = x_pv_2
            else:
                x_pv_3 = self.x_pred[0]

        # t2 = time.time()

        numba_predictX(
            self.gen_range[:, self.rank],
            ## Altered Arguments ##
            # were returned directly
            self.xp.pd_w,
            self.xp.pd_id,
            self.xp.pd_iq,
            self.xp.pd_EFD,
            self.xp.pd_u_d,
            self.xp.pd_u_q,
            self.xp.pd_dt,
            # were in point_one_tuple
            self.xp.pv_dt_1,
            self.xp.pv_w_1,
            self.xp.pv_id_1,
            self.xp.pv_iq_1,
            self.xp.pv_ifd_1,
            self.xp.pv_i1d_1,
            self.xp.pv_i1q_1,
            self.xp.pv_i2q_1,
            self.xp.pv_v1_1,
            self.xp.pv_EFD_1,
            self.xp.pv_ed_1,
            self.xp.pv_eq_1,
            self.xp.pv_psyd_1,
            self.xp.pv_psyq_1,
            self.xp.pv_te_1,
            self.xp.pv_u_d_1,
            self.xp.pv_u_q_1,
            self.xp.pv_i_d_1,
            self.xp.pv_i_q_1,
            self.xp.pv_p1_1,
            self.xp.pv_p2_1,
            self.xp.pv_p3_1,
            self.xp.pv_pm_1,
            ## Constant Arguments ##
            x_pv_1,
            x_pv_2,
            x_pv_3,
            pfd.ws,
            dyd.gen_genrou_odr,
            dyd.exc_sexs_xi_st,
            dyd.exc_sexs_odr,
            ts,
            xlen,
        )

        # t3 = time.time()

        # self.logic += t2 - t1
        # self.px_numba += t3 - t2
        # self.predx_total += t3 - t1

        return

    def updateIg(self, pfd, dyd, ini):
        numba_updateIg(
            # Indices for looping
            self.gen_range[:, self.rank],
            # Altered Arguments
            self.Igs,
            self.Isg,
            self.x_pv_1,
            self.ed_mod,
            self.eq_mod,
            self.theta,
            self.xp.pv_his_d_1,
            self.xp.pv_his_fd_1,
            self.xp.pv_his_1d_1,
            self.xp.pv_his_q_1,
            self.xp.pv_his_1q_1,
            self.xp.pv_his_2q_1,
            self.xp.pv_his_red_d_1,
            self.xp.pv_his_red_q_1,
            # Constant Arguments
            # pfd
            pfd.gen_bus,
            pfd.bus_num,
            # dyd
            dyd.base_Is,
            dyd.ec_Rfd,
            dyd.ec_Lad,
            dyd.gen_genrou_odr,
            # ini
            ini.Init_mac_alpha,
            ini.Init_mac_Rd,
            ini.Init_mac_Rq,
            ini.Init_mac_Rd2,
            ini.Init_mac_Rq2,
            ini.Init_mac_Rd_coe,
            ini.Init_mac_Rq_coe,
            ini.Init_mac_Rav,
            ini.Init_net_IbaseA,
            # self.xp
            self.xp.pv_i_d_1,
            self.xp.pv_u_d_1,
            self.xp.pv_EFD_1,
            self.xp.pv_i_q_1,
            self.xp.pv_u_q_1,
            self.xp.pd_EFD,
            self.xp.pd_u_d,
            self.xp.pd_u_q,
            self.xp.pd_id,
            self.xp.pd_iq,
            self.xp.pd_dt,
            self.flag_gentrip,
            self.i_gentrip,
        )

        return

    def updateIibr(self, pfd, dyd, ini):
        numba_updateIibr(
            self.ibr_range[:, self.rank],
            ## Begin "Returned" Arrays ##
            self.Igi,
            self.Iibr,
            ## End "Returned" Arrays ##
            # pfd
            pfd.ibr_bus,
            pfd.bus_num,
            # dyd
            dyd.ibr_Ibase,
            # ini
            ini.Init_net_IbaseA,
            dyd.ibr_wecc_odr,
            # other
            self.Vsol,
            self.x_ibr_pv_1,
            self.ts,
            self.x_bus_pv_1,
            dyd.bus_odr,
        )

        return

    def updateIibr_epri(self, pfd, dyd, ini, tn):
        if self.nibrepri > 0:
            N1 = len(pfd.bus_num)
            N3 = N1 * 3

            Nibr = self.nibrepri
            Nbch = len(ini.Init_net_coe0) - 6 * Nibr

            # self.Iibr_epri[:] = 0.0
            # self.Igi_epri[:] = 0.0

            ebr_range = self.ebr_range[:, self.rank]

            for i in range(ebr_range[0], ebr_range[1]):
                ibri = self.ibr_epri[i]

                ibri.cTime = tn * self.ts

                ibrbus = dyd.ibr_epri_bus[i]
                ibrid = dyd.ibr_epri_id[i]
                # if len(ibrid)==1:
                #     ibrid = ibrid + ' '

                ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0]
                ibrid_idx = np.where(pfd.ibr_id[ibrbus_idx] == ibrid)[0][0]
                ibrbus_idx = ibrbus_idx[ibrid_idx]

                bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]
                kVbase = pfd.bus_basekV[bus_idx]
                IBR_MVA_base = pfd.ibr_MVA_base[ibrbus_idx]
                kAbase = IBR_MVA_base / (kVbase * np.sqrt(3.0))

                coe_idx = Nbch + 6 * i

                # print("i = ", i)
                # print("ibrbus = ", ibrbus)
                # print("ibrid = ", ibrid)
                # print("ibrbus_idx = ", ibrbus_idx)
                # print("ibrid_idx = ", ibrid_idx)
                # print("bus_idx = ", bus_idx)
                # print("coe_idx = ", coe_idx)

                # Should be updated by the network solution based on model output from
                # previous time step
                # Va, Vb, Vc
                coef1 = kVbase * np.sqrt(2.0 / 3.0)
                Vabc_1 = self.Vsol[N3 + i : N3 + i + 2 * Nibr + 1 : Nibr]
                ibri.cExternalInputs[0] = coef1 * Vabc_1[0]
                ibri.cExternalInputs[1] = coef1 * Vabc_1[1]
                ibri.cExternalInputs[2] = coef1 * Vabc_1[2]

                # print("Vabc_1 =", coef1 * Vabc_1)

                # Ia, Ib, Ic
                coef2 = (
                    kAbase * np.sqrt(2.0) * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]
                )
                ibri.cExternalInputs[3] = -coef2 * self.brch_Ipre[coe_idx]
                ibri.cExternalInputs[4] = -coef2 * self.brch_Ipre[coe_idx + 1]
                ibri.cExternalInputs[5] = -coef2 * self.brch_Ipre[coe_idx + 2]

                # print("brch_Ipre =", -coef2 * self.brch_Ipre[coe_idx:coe_idx+3])

                # IaL1, IbL1, IcL1
                IabcL1_1 = (
                    self.brch_Ipre[coe_idx + 3 : coe_idx + 6]
                    - self.brch_Ipre[coe_idx : coe_idx + 3]
                )
                ibri.cExternalInputs[6] = coef2 * IabcL1_1[0]
                ibri.cExternalInputs[7] = coef2 * IabcL1_1[1]
                ibri.cExternalInputs[8] = coef2 * IabcL1_1[2]

                # print("IabcL1_1 =", coef2 * IabcL1_1)

                # ibri.cExternalInputs[6] = coef2 * (self.brch_Ipre[coe_idx+3] - self.brch_Ipre[coe_idx])
                # ibri.cExternalInputs[7] = coef2 * (self.brch_Ipre[coe_idx+4] - self.brch_Ipre[coe_idx+1])
                # ibri.cExternalInputs[8] = coef2 * (self.brch_Ipre[coe_idx+5] - self.brch_Ipre[coe_idx+2])

                # IaL1_1 = -self.brch_Ipre[coe_idx]+self.brch_Ipre[coe_idx+3]
                # IbL1_1 = -self.brch_Ipre[coe_idx+1]+self.brch_Ipre[coe_idx+4]
                # IcL1_1 = -self.brch_Ipre[coe_idx+2]+self.brch_Ipre[coe_idx+5]
                # Va_1 = self.Vsol[N3 + i]
                # Vb_1 = self.Vsol[N3 + i + Nibr]
                # Vc_1 = self.Vsol[N3 + i + 2*Nibr]
                # Ea_1 = ibri.cExternalOutputs[0]/(kVbase*np.sqrt(2.0/3.0))
                # Eb_1 = ibri.cExternalOutputs[1]/(kVbase*np.sqrt(2.0/3.0))
                # Ec_1 = ibri.cExternalOutputs[2]/(kVbase*np.sqrt(2.0/3.0))

                Eabc_1 = np.array(
                    [
                        ibri.cExternalOutputs[0],
                        ibri.cExternalOutputs[1],
                        ibri.cExternalOutputs[2],
                    ]
                )
                Eabc_1 /= coef1

                # print(ibri.cExternalInputs[0:12])
                # print(ibri.cExternalOutputs[0:12])
                # Call main function from the DLL to update outputs based on updated inputs
                ierr = Model_Outputs(
                    ibri
                )  # Return:    Integer status 0 (normal), 1 if messages are written, 2 for errors.  See IEEE_Cigre_DLLInterface_types.h
                # print(ibri.cExternalInputs[0:12])
                # print(ibri.cExternalOutputs[0:12])
                # print('\n')
                self._check_epri_return_status(ierr)

                # Ea = ibri.cExternalOutputs[0]/(kVbase*np.sqrt(2.0/3.0))
                # Eb = ibri.cExternalOutputs[1]/(kVbase*np.sqrt(2.0/3.0))
                # Ec = ibri.cExternalOutputs[2]/(kVbase*np.sqrt(2.0/3.0))
                Eabc = np.array(
                    [
                        ibri.cExternalOutputs[0],
                        ibri.cExternalOutputs[1],
                        ibri.cExternalOutputs[2],
                    ]
                )
                Eabc /= coef1

                # update Iibr
                # Iibr_a = Ea/ini.Init_ibr_epri_Req[i] + (Ea_1-Va_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IaL1_1
                # Iibr_b = Eb/ini.Init_ibr_epri_Req[i] + (Eb_1-Vb_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IbL1_1
                # Iibr_c = Ec/ini.Init_ibr_epri_Req[i] + (Ec_1-Vc_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IcL1_1
                Iibr_abc = (
                    Eabc / ini.Init_ibr_epri_Req[i]
                    + (Eabc_1 - Vabc_1) * ini.Init_ibr_epri_Gv1[i]
                    + ini.Init_ibr_epri_icf[i] * IabcL1_1
                )

                # ===================================
                # considering interfacing resistance
                # if flag_itfc_R:
                #     Rii = np.dot(Ritfc,np.asarray([Iibr_a,Iibr_b,Iibr_c]))
                #     v_int = Rii + np.asarray([Va_1,Vb_1,Vc_1])
                #     iinj = np.dot(Gitfc ,  v_int )

                #     Iibr_a_itfc = iinj[0] - Iibr_a
                #     Iibr_b_itfc = iinj[1] - Iibr_b
                #     Iibr_c_itfc = iinj[2] - Iibr_c

                #     iinj = np.dot(Gitfc ,  np.asarray([Va_1,Vb_1,Vc_1]) )
                #     Iibr_a_itfc = iinj[0]
                #     Iibr_b_itfc = iinj[1]
                #     Iibr_c_itfc = iinj[2]

                # else:
                #     Iibr_a_itfc = 0
                #     Iibr_b_itfc = 0
                #     Iibr_c_itfc = 0
                # Iibr_a_itfc = 0
                # Iibr_b_itfc = 0
                # Iibr_c_itfc = 0
                Iibr_abc_itfc = np.zeros(3)
                # ===================================

                # self.Iibr_epri[3*i] = self.Iibr_epri[3*i] + Iibr_a + Iibr_a_itfc
                # self.Iibr_epri[3*i+1] = self.Iibr_epri[3*i+1] + Iibr_b + Iibr_b_itfc
                # self.Iibr_epri[3*i+2] = self.Iibr_epri[3*i+2] + Iibr_c + Iibr_c_itfc

                # self.Igi_epri[3*N1 + i] = self.Igi_epri[3*N1 + i] + Iibr_a + Iibr_a_itfc
                # self.Igi_epri[3*N1 + dyd.ibr_epri_n+i] = self.Igi_epri[3*N1 + dyd.ibr_epri_n+i] + Iibr_b + Iibr_b_itfc
                # self.Igi_epri[3*N1 + 2*dyd.ibr_epri_n+i] = self.Igi_epri[3*N1 + 2*dyd.ibr_epri_n+i] + Iibr_c + Iibr_c_itfc

                # self.Iibr_epri[3*i] = Iibr_a + Iibr_a_itfc
                # self.Iibr_epri[3*i+1] = Iibr_b + Iibr_b_itfc
                # self.Iibr_epri[3*i+2] = Iibr_c + Iibr_c_itfc
                self.Iibr_epri[3 * i : 3 * i + 3] = Iibr_abc + Iibr_abc_itfc

                # self.Igi_epri[N3 + i] = Iibr_a + Iibr_a_itfc
                # self.Igi_epri[N3 + Nibr + i] = Iibr_b + Iibr_b_itfc
                # self.Igi_epri[N3 + 2*Nibr + i] = Iibr_c + Iibr_c_itfc
                self.Igi_epri[N3 + i : N3 + 2 * Nibr + i + 1 : Nibr] = (
                    Iibr_abc + Iibr_abc_itfc
                )

                idx = i * dyd.ibr_epri_odr
                self.x_ibr_epri_pv_1[0 + idx] = ibri.cExternalOutputs[0]
                self.x_ibr_epri_pv_1[1 + idx] = ibri.cExternalOutputs[1]
                self.x_ibr_epri_pv_1[2 + idx] = ibri.cExternalOutputs[2]
                self.x_ibr_epri_pv_1[3 + idx] = ibri.cExternalOutputs[3]
                self.x_ibr_epri_pv_1[4 + idx] = ibri.cExternalOutputs[4]
                self.x_ibr_epri_pv_1[5 + idx] = ibri.cExternalOutputs[5]
                self.x_ibr_epri_pv_1[6 + idx] = ibri.cExternalOutputs[6]
                self.x_ibr_epri_pv_1[7 + idx] = ibri.cExternalOutputs[7]
                self.x_ibr_epri_pv_1[8 + idx] = ibri.cExternalOutputs[8]
                self.x_ibr_epri_pv_1[9 + idx] = ibri.cExternalOutputs[9]
                self.x_ibr_epri_pv_1[10 + idx] = ibri.cExternalOutputs[10]
                self.x_ibr_epri_pv_1[11 + idx] = ibri.cExternalOutputs[11]
                self.x_ibr_epri_pv_1[12 + idx] = np.sqrt(
                    ibri.cExternalOutputs[7] * ibri.cExternalOutputs[7]
                    + ibri.cExternalOutputs[8] * ibri.cExternalOutputs[8]
                )

            ## End for

        ## End if

        return

    def updateIl(self, pfd, dyd, tn):
        if self.loadmodel_option == 1:
            return

        numba_updateIl(
            self.load_range[:, self.rank],
            ## Returned Arrays ##
            self.Il,
            self.Ild,
            ## Constant Arrays ##
            self.x_load_pv_1,
            self.x_bus_pv_1,
            pfd.bus_num,
            pfd.load_bus,
            pfd.ws,
            dyd.bus_odr,
            dyd.load_odr,
            tn,
            self.ts,
            self.t_release_f,
        )

        # NOTE: Ild does *NOT* need to be synced across ranks!

        return

    def helics_receive(self, tn, record4cosim):
        if record4cosim == False and len(self.emt_zones) > 0:
            if self.playback_enable == False:  # use helics
                # # TODO: the code below is hard-coded for systemN = 9
                # ia_9_10 = 0
                # ib_9_10 = 0
                # ic_9_10 = 0

                # # i_9_10_phasor, v_10_phasor will be provided by GridPack thru HELICS
                # # -------------------------------------------------------------phasor
                # # signal from GridPack

                # sub = self.sub

                # tmp = {}
                # for key in sub.keys():
                #     x = sub[key]
                #     msg = h.helicsInputGetDouble(x)
                #     if True:
                #         tmp[key] = msg

                # if tmp != {}:
                #     keys_to_check = ["gridpack_i_9_10_real",
                #         "gridpack_i_9_10_imag",
                #         "gridpack_v_9_10_real",
                #         "gridpack_v_9_10_imag"]
                #     recv = {}
                #     if self.currenttime > 0.010:
                #         if(set(list(tmp.keys())).intersection(set(keys_to_check))):
                #             recv["i_9_10_phasor"] = complex(tmp['gridpack_i_9_10_real'],
                #                                             tmp['gridpack_i_9_10_imag'])
                #             recv["v_10_phasor"] = complex(tmp["gridpack_v_9_10_real"],
                #                                         tmp["gridpack_v_9_10_imag"])
                #     else:
                #         recv["i_9_10_phasor"] = complex(-9.713144, 10.52305)
                #         recv["v_10_phasor"] = complex(0.70720997, -0.683408834)

                # if recv != {}:
                #     i_9_10_phasor = recv["i_9_10_phasor"]
                #     v_10_phasor = recv["v_10_phasor"]

                # # self.sync_time_helics(self.deltat)

                # # recv = self.data[self.idx].tolist()

                # # print("RECEIVE ", recv)

                # kk = 0

                # =======================================================
                # temp code to be removed once helic is setup
                Gd = 100
                Go = 0.001

                self.Gitfc = np.asarray([[Gd, Go, Go], [Go, Gd, Go], [Go, Go, Gd]])
                self.Ritfc = np.linalg.inv(self.Gitfc)

                i_9_10_phasor0 = complex(
                    -9.713144, 10.52305
                )  # this default value here is expected at t=0.0
                v_10_phasor0 = complex(
                    0.70720997, -0.683408834
                )  # this default value here is expected at t=0.0
                # end of temp code
                # =======================================================

                self.stepk = (
                    self.stepk + 1
                )  # time advances (self.stepk is initialized to be (kts - 1))

                if self.stepk >= self.kts:
                    # TODO: update phasor here and then reset self.stepk
                    self.iphasor = i_9_10_phasor0 * complex(
                        np.cos(2 * np.pi * 60 * self.ts * tn),
                        np.sin(2 * np.pi * 60 * self.ts * tn),
                    )
                    self.vphasor = v_10_phasor0 * complex(
                        np.cos(2 * np.pi * 60 * self.ts * tn),
                        np.sin(2 * np.pi * 60 * self.ts * tn),
                    )

                    self.stepk = self.stepk - self.kts
                # otherwise, iv phasors below will be hold at their values at previous phasor step

                if self.ts * tn <= 0.1:
                    mag = abs(self.iphasor)
                else:
                    mag = abs(self.iphasor) + 0.05 * abs(self.iphasor) * np.cos(
                        2 * np.pi * 2.3 * self.ts * tn
                    )
                theta = np.angle(self.iphasor)
                ia_9_10_rcv = mag * np.cos(
                    2 * np.pi * 60 * self.ts * self.stepk + theta
                )
                ib_9_10_rcv = mag * np.cos(
                    2 * np.pi * 60 * self.ts * self.stepk + theta - 2 / 3 * np.pi
                )
                ic_9_10_rcv = mag * np.cos(
                    2 * np.pi * 60 * self.ts * self.stepk + theta + 2 / 3 * np.pi
                )

                vmag = abs(self.vphasor)
                vtheta = np.angle(self.vphasor)
                va_10 = vmag * np.cos(2 * np.pi * 60 * self.ts * self.stepk + vtheta)
                vb_10 = vmag * np.cos(
                    2 * np.pi * 60 * self.ts * self.stepk + vtheta - 2 / 3 * np.pi
                )
                vc_10 = vmag * np.cos(
                    2 * np.pi * 60 * self.ts * self.stepk + vtheta + 2 / 3 * np.pi
                )

                Rii = np.dot(
                    self.Ritfc, np.asarray([ia_9_10_rcv, ib_9_10_rcv, ic_9_10_rcv])
                )
                v_int = Rii + np.asarray([va_10, vb_10, vc_10])
                iinj = np.dot(self.Gitfc, v_int)

                # print(iinj[0])
                busb = int(self.emt_zones[0]["bus_b"])
                nbus = len(self.pfd.bus_num)
                idx = np.where(self.pfd.bus_num == busb)[0][0]

                self.node_Ihis[idx] = self.node_Ihis[idx] + iinj[0]
                self.node_Ihis[idx + nbus] = self.node_Ihis[idx + nbus] + iinj[1]
                self.node_Ihis[idx + 2 * nbus] = (
                    self.node_Ihis[idx + 2 * nbus] + iinj[2]
                )

                ## END of HELICS

            else:  # use recorded signals
                # TODO: should be placed outside the time loop to save computations
                # # i_phasor = i_phasor*complex(np.cos(2*np.pi*60*self.ts*tn), np.sin(2*np.pi*60*self.ts*tn))
                # # v_phasor = v_phasor*complex(np.cos(2*np.pi*60*self.ts*tn), np.sin(2*np.pi*60*self.ts*tn))
                # GS2 = {'gridpack_v_1201_real': 1.084087181927998, 'gridpack_v_1301_real': 1.029543802229106,\
                #         'gridpack_v_1401_real': 1.000744756353935, 'gridpack_v_3892_real': 1.036294762823339,\
                #         'gridpack_v_3894_real': 1.036160744516098, 'gridpack_v_3896_real': 1.036739330637525,\
                #         'gridpack_v_1201_imag': -0.125784067271704, 'gridpack_v_1301_imag': -0.179065795984702,\
                #         'gridpack_v_1401_imag': -0.264073725747382, 'gridpack_v_3892_imag': -0.07079285730156,\
                #         'gridpack_v_3894_imag': -0.069813548999076, 'gridpack_v_3896_imag': -0.066968427704823,\
                #         'gridpack_i_1201_real': -4.927815908488634, 'gridpack_i_1301_real': -0.307492501873992,\
                #         'gridpack_i_1401_real': -3.131277579516216, 'gridpack_i_3892_real': -6.290400844742230,\
                #         'gridpack_i_3894_real': -6.334348886056785, 'gridpack_i_3896_real': -6.698736952382744,\
                #         'gridpack_i_1201_imag': -3.324602792419579, 'gridpack_i_1301_imag': -1.527801543739057,\
                #         'gridpack_i_1401_imag': -1.464021523895691, 'gridpack_i_3892_imag': -0.193655857221346,
                #         'gridpack_i_3894_imag': -0.206316080586385, 'gridpack_i_3896_imag': -0.146995598784720}

                # bus_rec_all = []
                # bus_rec_idx = []
                # nn_busb = 0

                # for i in range(len(self.emt_zones)):
                #     for j in range(len(self.emt_zones[i]['bus_b'])):
                #         busb = int(self.emt_zones[i]['bus_b'][j])
                #         bus_rec_all = np.append(bus_rec_all, busb)
                #         if i == self.EMT_N - 1:
                #             bus_rec_idx = np.append(bus_rec_idx, nn_busb)
                #         nn_busb = nn_busb + 1
                # Gd = 100
                # Go = 0.001
                # self.Gitfc = np.asarray([[Gd, Go, Go],[Go, Gd, Go],[Go, Go, Gd]])
                # self.Ritfc = np.linalg.inv(self.Gitfc)

                # nbus = len(self.pfd.bus_num)
                # for i in range(len(bus_rec_idx)):
                #     busb_idx = int(bus_rec_idx[i])
                #     busb = int(self.emt_zones[0]['bus_b'][busb_idx])        # 1 EMT zone, TODO
                #     idx = np.where(self.pfd.bus_num == busb)[0][0]
                #     # iphasor = self.bus_iv[busb]["i_phasor"] # this is the iphasor from gridpack
                #     # vphasor = self.bus_iv[busb]["v_phasor"]# this is the vphasor from gridpack
                #     iphasor = -complex(GS2[f"gridpack_i_{int(busb)}_real"], GS2[f"gridpack_i_{int(busb)}_imag"])
                #     vphasor = complex(GS2[f"gridpack_v_{int(busb)}_real"], GS2[f"gridpack_v_{int(busb)}_imag"])
                #     self.stepk = tn  # Min Added
                #     mag = abs(iphasor)
                #     theta = np.angle(iphasor)
                #     ia_9_10_rcv = mag*np.cos(2*np.pi*60*self.ts*self.stepk + theta)
                #     ib_9_10_rcv = mag*np.cos(2*np.pi*60*self.ts*self.stepk + theta - 2/3*np.pi)
                #     ic_9_10_rcv = mag*np.cos(2*np.pi*60*self.ts*self.stepk + theta + 2/3*np.pi)
                #     vmag = abs(vphasor)
                #     vtheta = np.angle(vphasor)
                #     va_10 = vmag*np.cos(2*np.pi*60*self.ts*self.stepk + vtheta)
                #     vb_10 = vmag*np.cos(2*np.pi*60*self.ts *self.stepk + vtheta - 2/3*np.pi)
                #     vc_10 = vmag*np.cos(2*np.pi*60*self.ts *self.stepk + vtheta + 2/3*np.pi)

                #     Rii = np.dot(self.Ritfc, np.asarray([ia_9_10_rcv, ib_9_10_rcv, ic_9_10_rcv]))
                #     v_int = Rii + np.asarray([va_10, vb_10, vc_10])
                #     iinj = np.dot(self.Gitfc, v_int)
                #     self.node_Ihis[idx] = self.node_Ihis[idx] + iinj[0]
                #     self.node_Ihis[idx+nbus] = self.node_Ihis[idx+nbus] + iinj[1]
                #     self.node_Ihis[idx+2*nbus] = self.node_Ihis[idx+2*nbus] + iinj[2]

                # TODO: Orinigal code
                bus_rec_all = []
                bus_rec_idx = []
                nn_busb = 0
                for i in range(len(self.emt_zones)):
                    for j in range(len(self.emt_zones[i]["bus_b"])):
                        busb = int(self.emt_zones[i]["bus_b"][j])
                        bus_rec_all = np.append(bus_rec_all, busb)
                        if i == self.EMT_N - 1:
                            bus_rec_idx = np.append(bus_rec_idx, nn_busb)
                        nn_busb = nn_busb + 1

                nbus = len(self.pfd.bus_num)
                for i in range(len(bus_rec_idx)):
                    busb_idx = int(bus_rec_idx[i])
                    busb = bus_rec_all[busb_idx]
                    idx = np.where(self.pfd.bus_num == busb)[0][0]

                    if tn * self.ts >= self.data[self.playback_tn, self.playback_t_chn]:
                        ia = self.data[
                            self.playback_tn + 1, self.playback_sig_chn + 3 * busb_idx
                        ]
                        ib = self.data[
                            self.playback_tn + 1,
                            self.playback_sig_chn + 3 * busb_idx + 1,
                        ]
                        ic = self.data[
                            self.playback_tn + 1,
                            self.playback_sig_chn + 3 * busb_idx + 2,
                        ]

                        self.node_Ihis[idx] = self.node_Ihis[idx] + ia
                        self.node_Ihis[idx + nbus] = self.node_Ihis[idx + nbus] + ib
                        self.node_Ihis[idx + 2 * nbus] = (
                            self.node_Ihis[idx + 2 * nbus] + ic
                        )

                self.playback_tn += 1
        return

    def BusMea(self, pfd, dyd, tn):
        x_bus_nx = np.zeros(self.nbus * dyd.bus_odr)

        numba_BusMea(
            self.bus_range[:, self.rank],
            x_bus_nx,
            self.Vsol,
            self.x_bus_pv_1,
            self.nbus,
            self.ts,
            self.t_release_f,
            pfd.ws,
            dyd.bus_odr,
            dyd.vm_te,
            dyd.pll_ke,
            dyd.pll_te,
            tn,
        )

        self.x_bus_pv_1 = x_bus_nx

        return

    def updateX(self, pfd, dyd, ini, tn, playback_voltphasor):
        self.x_pv_1 = numba_updateX(
            # Indices for looping
            self.gen_range[:, self.rank],
            # Altered Arguments
            self.x_pv_1,
            self.xp.nx_ed,
            self.xp.nx_eq,
            self.xp.nx_id,
            self.xp.nx_iq,
            self.xp.nx_ifd,
            self.xp.nx_i1d,
            self.xp.nx_i1q,
            self.xp.nx_i2q,
            self.xp.nx_psyd,
            self.xp.nx_psyq,
            self.xp.nx_psyfd,
            self.xp.nx_psy1q,
            self.xp.nx_psy1d,
            self.xp.nx_psy2q,
            self.xp.nx_te,
            self.xp.nx_w,
            self.xp.nx_EFD,
            self.xp.nx_dt,
            self.xp.nx_v1,
            self.xp.nx_pm,
            # Constant Arguments
            self.xp.pd_dt,
            self.xp.pd_EFD,
            self.xp.pv_his_fd_1,
            self.xp.pv_his_1d_1,
            self.xp.pv_his_1q_1,
            self.xp.pv_his_2q_1,
            self.xp.pv_dt_1,
            self.xp.pv_w_1,
            self.xp.pv_EFD_1,
            pfd.gen_bus,
            pfd.bus_num,
            pfd.ws,
            pfd.basemva,
            pfd.gen_MVA_base,
            dyd.gen_H,
            dyd.gen_D,
            dyd.gen_genrou_n,
            dyd.gen_genrou_odr,
            dyd.gen_genrou_xi_st,
            dyd.ec_Rfd,
            dyd.ec_Lad,
            dyd.ec_Laq,
            dyd.ec_Ll,
            dyd.ec_Lffd,
            dyd.ec_L11q,
            dyd.ec_L11d,
            dyd.ec_Lf1d,
            dyd.ec_L22q,
            dyd.pss_ieeest_A1,
            dyd.pss_ieeest_A2,
            dyd.pss_ieeest_A3,
            dyd.pss_ieeest_A4,
            dyd.pss_ieeest_A5,
            dyd.pss_ieeest_A6,
            dyd.pss_ieeest_T1,
            dyd.pss_ieeest_T2,
            dyd.pss_ieeest_T3,
            dyd.pss_ieeest_T4,
            dyd.pss_ieeest_T5,
            dyd.pss_ieeest_T6,
            dyd.pss_ieeest_KS,
            dyd.pss_ieeest_LSMAX,
            dyd.pss_ieeest_LSMIN,
            dyd.pss_ieeest_VCL,
            dyd.pss_ieeest_VCU,
            dyd.pss_ieeest_idx,
            dyd.pss_ieeest_odr,
            dyd.pss_ieeest_xi_st,
            dyd.exc_sexs_TA,
            dyd.exc_sexs_TB,
            dyd.exc_sexs_K,
            dyd.exc_sexs_TE,
            dyd.exc_sexs_Emin,
            dyd.exc_sexs_Emax,
            dyd.exc_sexs_idx,
            dyd.exc_sexs_n,
            dyd.exc_sexs_odr,
            dyd.exc_sexs_xi_st,
            dyd.gov_type,
            dyd.gov_tgov1_bus,
            dyd.gov_tgov1_id,
            dyd.gov_tgov1_Dt,
            dyd.gov_tgov1_R,
            dyd.gov_tgov1_T1,
            dyd.gov_tgov1_T2,
            dyd.gov_tgov1_T3,
            dyd.gov_tgov1_Vmax,
            dyd.gov_tgov1_Vmin,
            dyd.gov_tgov1_idx,
            dyd.gov_tgov1_n,
            dyd.gov_tgov1_odr,
            dyd.gov_tgov1_xi_st,
            dyd.gov_hygov_bus,
            dyd.gov_hygov_id,
            dyd.gov_hygov_At,
            dyd.gov_hygov_Dturb,
            dyd.gov_hygov_GMAX,
            dyd.gov_hygov_GMIN,
            dyd.gov_hygov_R,
            dyd.gov_hygov_TW,
            dyd.gov_hygov_Tf,
            dyd.gov_hygov_Tg,
            dyd.gov_hygov_Tr,
            dyd.gov_hygov_VELM,
            dyd.gov_hygov_qNL,
            dyd.gov_hygov_r,
            dyd.gov_hygov_idx,
            dyd.gov_hygov_n,
            dyd.gov_hygov_odr,
            dyd.gov_hygov_xi_st,
            dyd.gov_gast_bus,
            dyd.gov_gast_id,
            dyd.gov_gast_R,
            dyd.gov_gast_LdLmt,
            dyd.gov_gast_KT,
            dyd.gov_gast_T1,
            dyd.gov_gast_T2,
            dyd.gov_gast_T3,
            dyd.gov_gast_VMIN,
            dyd.gov_gast_VMAX,
            dyd.gov_gast_Dturb,
            dyd.gov_gast_idx,
            dyd.gov_gast_n,
            dyd.gov_gast_odr,
            dyd.gov_gast_xi_st,
            dyd.bus_odr,
            ini.Init_mac_Rav,
            ini.Init_mac_Rd1,
            ini.Init_mac_Rd1inv,
            ini.Init_mac_Rq1,
            ini.Init_mac_Rq1inv,
            ini.Init_mac_Gequiv,
            ini.tgov1_2gen,
            ini.hygov_2gen,
            ini.gast_2gen,
            self.vref,
            self.gref,
            self.Vsol,
            self.Isg,
            self.ed_mod,
            self.eq_mod,
            self.vref_1,
            self.x_bus_pv_1,
            self.ts,
            self.flag_gentrip,
            self.i_gentrip,
            playback_voltphasor,
            # self.tsat_gen_omg,
            # self.tsat_gen_maci,
            # tn
        )

        return

    def updateXibr(self, pfd, dyd, ini):
        # t0 = time.time()

        numba_updateXibr(
            self.ibr_range[:, self.rank],
            # Altered Arguments
            self.x_ibr_pv_1,
            # Constant Arguments
            pfd.ibr_bus,
            pfd.bus_num,
            pfd.ws,
            pfd.basemva,
            pfd.ibr_MVA_base,
            dyd.ibr_regca_Volim,
            dyd.ibr_regca_Khv,
            dyd.ibr_regca_Lvpnt0,
            dyd.ibr_regca_Lvpnt1,
            dyd.ibr_regca_Tg,
            dyd.ibr_regca_Iqrmax,
            dyd.ibr_regca_Iqrmin,
            dyd.ibr_regca_Tfltr,
            dyd.ibr_regca_Zerox,
            dyd.ibr_regca_Brkpt,
            dyd.ibr_regca_Rrpwr,
            dyd.ibr_reecb_PQFLAG,
            dyd.ibr_reecb_PFFLAG,
            dyd.ibr_reecb_VFLAG,
            dyd.ibr_reecb_QFLAG,
            dyd.ibr_reecb_Imax,
            dyd.ibr_reecb_Vdip,
            dyd.ibr_reecb_Vup,
            dyd.ibr_reecb_Trv,
            dyd.ibr_reecb_dbd1,
            dyd.ibr_reecb_dbd2,
            dyd.ibr_reecb_Kqv,
            dyd.ibr_reecb_Iqll,
            dyd.ibr_reecb_Iqhl,
            dyd.ibr_reecb_Tp,
            dyd.ibr_reecb_Qmin,
            dyd.ibr_reecb_Qmax,
            dyd.ibr_reecb_Kqp,
            dyd.ibr_reecb_Kqi,
            dyd.ibr_reecb_Vmin,
            dyd.ibr_reecb_Vmax,
            dyd.ibr_reecb_Kvp,
            dyd.ibr_reecb_Kvi,
            dyd.ibr_reecb_Tiq,
            dyd.ibr_reecb_dPmin,
            dyd.ibr_reecb_dPmax,
            dyd.ibr_reecb_Pmin,
            dyd.ibr_reecb_Pmax,
            dyd.ibr_reecb_Tpord,
            dyd.ibr_repca_FFlag,
            dyd.ibr_repca_VCFlag,
            dyd.ibr_repca_RefFlag,
            dyd.ibr_repca_fdbd1,
            dyd.ibr_repca_fdbd2,
            dyd.ibr_repca_Ddn,
            dyd.ibr_repca_Dup,
            dyd.ibr_repca_Tp,
            dyd.ibr_repca_femin,
            dyd.ibr_repca_femax,
            dyd.ibr_repca_Kpg,
            dyd.ibr_repca_Kig,
            dyd.ibr_repca_Pmin,
            dyd.ibr_repca_Pmax,
            dyd.ibr_repca_Tg,
            dyd.ibr_repca_Rc,
            dyd.ibr_repca_Xc,
            dyd.ibr_repca_Kc,
            dyd.ibr_repca_Tfltr,
            dyd.ibr_repca_dbd1,
            dyd.ibr_repca_dbd2,
            dyd.ibr_repca_emin,
            dyd.ibr_repca_emax,
            dyd.ibr_repca_Vfrz,
            dyd.ibr_repca_Kp,
            dyd.ibr_repca_Ki,
            dyd.ibr_repca_Qmin,
            dyd.ibr_repca_Qmax,
            dyd.ibr_repca_Tft,
            dyd.ibr_repca_Tfv,
            # dyd.ibr_pll_ke,
            # dyd.ibr_pll_te,
            dyd.ibr_wecc_odr,
            ini.Init_ibr_regca_Qgen0,
            ini.Init_ibr_reecb_pfaref,
            ini.Init_ibr_reecb_Vref0,
            ini.Init_ibr_repca_Pref_out,
            self.Vsol,
            self.x_bus_pv_1,
            dyd.bus_odr,
            # vtm,
            self.Iibr,
            self.ts,
        )

        return

    def updateXl(self, pfd, dyd, tn):
        if self.loadmodel_option == 1:
            return

        # calc load power
        # x_load_nx = np.zeros(len(pfd.load_bus) * dyd.load_odr)
        x_load_nx = self.x_load_pv_1.copy()

        numba_updateXl(
            self.load_range[:, self.rank],
            ## Returned Arrays ##
            x_load_nx,
            ## Constants ##
            pfd.load_bus,
            pfd.bus_num,
            dyd.load_odr,
            dyd.bus_odr,
            self.x_load_pv_1,
            # self.x_bus_pv_1,
            self.Vsol,
            self.Ild,
            tn,
        )

        self.x_load_pv_1 = x_load_nx

        return

    def save(self, tn):
        self.x_pred = {0: self.x_pred[1], 1: self.x_pred[2], 2: self.x_pv_1}

        # self.Vsol_1 = self.Vsol
        self.save_idx += 1

        # Save time
        self.t.append(tn * self.ts)

        # Save bus voltages
        self.v[self.save_idx] = self.Vsol.copy()

        # Save branch crrent
        term=self.brch_Ipre.copy()
        term2=9*len(self.pfd.line_from)
        self.i_branch[self.save_idx]=np.concatenate((term[0:term2:9], term[1:term2:9], term[2:term2:9]))  
        
        # Save generator state
        if self.ngen > 0:
            self.x[self.save_idx] = self.x_pv_1

        # Save IBR state
        if self.nibr > 0:
            self.x_ibr[self.save_idx] = self.x_ibr_pv_1.copy()

        # Save EPRI IBR state
        if self.nibrepri > 0:
            self.x_ibr_epri[self.save_idx] = self.x_ibr_epri_pv_1.copy()

        # Save bus state variables
        if self.nbus > 0:
            self.x_bus[self.save_idx] = self.x_bus_pv_1.copy()

        # Save load state variables
        if self.nload > 0 and self.loadmodel_option == 2:
            self.x_load[self.save_idx] = self.x_load_pv_1.copy()

    def update_phasor(self):
        if self.compute_phasor == 1:
            self.fft_vabc.append(self.Vsol.copy())

            # TODO: adapted indices, but can be moved to outside time loop to reduce computations
            brch_Ipre = self.brch_Ipre
            nbus_b = len(self.emt_zones[self.EMT_N - 1]["bus_b"])
            i_inj = np.zeros(3 * nbus_b)
            for j in range(nbus_b):
                busb = int(self.emt_zones[self.EMT_N - 1]["bus_b"][j])

                # current: find branch and add current
                # find all branches
                InjBranch = np.array([])
                bus_emt = self.emt_zones[self.EMT_N - 1]["buses"]
                # line
                idx1 = np.where(self.pfd.line_from == busb)[0]
                for k in range(len(idx1)):
                    bust = self.pfd.line_to[idx1[k]]
                    idx2 = np.where(bus_emt == bust)[0]
                    if len(idx2) != 0:  # in emt zone
                        # check if identified branch is already in InjBranch
                        if len(InjBranch) > 0:
                            flag = 0
                            for ii in range(len(InjBranch)):
                                if any(
                                    [
                                        all(InjBranch[ii] == [busb, bust]),
                                        all(InjBranch[ii] == [bust, busb]),
                                    ]
                                ):
                                    flag = 1
                            if flag == 0:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        1,
                                    ],
                                )  # third entry: 1-line, 0-xfmr
                        else:
                            InjBranch = np.append(
                                InjBranch,
                                [
                                    np.where(self.pfd.bus_num == busb)[0][0],
                                    np.where(self.pfd.bus_num == bust)[0][0],
                                    1,
                                ],
                            )

                idx1 = np.where(self.pfd.line_to == busb)[0]
                for k in range(len(idx1)):
                    bust = self.pfd.line_from[idx1[k]]
                    idx2 = np.where(bus_emt == bust)[0]
                    if len(idx2) != 0:  # in emt zone
                        # check if identified branch is already in InjBranch
                        if len(InjBranch) > 0:
                            flag = 0
                            for ii in range(len(InjBranch)):
                                if any(
                                    [
                                        all(InjBranch[ii] == [busb, bust]),
                                        all(InjBranch[ii] == [bust, busb]),
                                    ]
                                ):
                                    flag = 1
                            if flag == 0:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        1,
                                    ],
                                )
                        else:
                            InjBranch = np.append(
                                InjBranch,
                                [
                                    np.where(self.pfd.bus_num == busb)[0][0],
                                    np.where(self.pfd.bus_num == bust)[0][0],
                                    1,
                                ],
                            )
                # xfmr
                idx1 = np.where(self.pfd.xfmr_from == busb)[0]
                for k in range(len(idx1)):
                    bust = self.pfd.xfmr_to[idx1[k]]
                    idx2 = np.where(bus_emt == bust)[0]
                    if len(idx2) != 0:  # in emt zone
                        # check if identified branch is already in InjBranch
                        if len(InjBranch) > 0:
                            flag = 0
                            for ii in range(len(InjBranch)):
                                if any(
                                    [
                                        all(InjBranch[ii] == [busb, bust]),
                                        all(InjBranch[ii] == [bust, busb]),
                                    ]
                                ):
                                    flag = 1
                            if flag == 0:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        0,
                                    ],
                                )
                        else:
                            InjBranch = np.append(
                                InjBranch,
                                [
                                    np.where(self.pfd.bus_num == busb)[0][0],
                                    np.where(self.pfd.bus_num == bust)[0][0],
                                    0,
                                ],
                            )

                idx1 = np.where(self.pfd.xfmr_to == busb)[0]
                for k in range(len(idx1)):
                    bust = self.pfd.xfmr_from[idx1[k]]
                    idx2 = np.where(bus_emt == bust)[0]
                    if len(idx2) != 0:  # in emt zone
                        # check if identified branch is already in InjBranch
                        if len(InjBranch) > 0:
                            flag = 0
                            for ii in range(len(InjBranch)):
                                if any(
                                    [
                                        all(InjBranch[ii] == [busb, bust]),
                                        all(InjBranch[ii] == [bust, busb]),
                                    ]
                                ):
                                    flag = 1
                            if flag == 0:
                                InjBranch = np.append(
                                    InjBranch,
                                    [
                                        np.where(self.pfd.bus_num == busb)[0][0],
                                        np.where(self.pfd.bus_num == bust)[0][0],
                                        0,
                                    ],
                                )
                        else:
                            InjBranch = np.append(
                                InjBranch,
                                [
                                    np.where(self.pfd.bus_num == busb)[0][0],
                                    np.where(self.pfd.bus_num == bust)[0][0],
                                    0,
                                ],
                            )

                lenIB = int(len(InjBranch) / 3)
                # print([self.pfd.bus_num[int(InjBranch[:][0])], self.pfd.bus_num[int(InjBranch[:][1])]])

                ####   Deleted   Min 01122024
                # ia = 0
                # ib = 0
                # ic = 0
                # for k in range(lenIB):
                #     # print([self.pfd.bus_num[int(InjBranch[3*k])],self.pfd.bus_num[int(InjBranch[3*k+1])]])
                #     idx1 = np.where(self.ini.Init_net_coe0[:,0] == InjBranch[3*k])[0]
                #     idx2 = np.where(self.ini.Init_net_coe0[idx1,1] == InjBranch[3*k+1])[0]
                #     len2 = len(idx2)
                #     if len2>0:
                #         for ii in range(len2):
                #             idx = idx1[idx2[ii]]
                #             # print([idx,-brch_Ipre[idx  ] - brch_Ipre[idx+3]*InjBranch[3*k+2]])
                #             ia = ia - brch_Ipre[idx  ] - brch_Ipre[idx+3]*InjBranch[3*k+2]
                #             ib = ib - brch_Ipre[idx+1] - brch_Ipre[idx+4]*InjBranch[3*k+2]
                #             ic = ic - brch_Ipre[idx+2] - brch_Ipre[idx+5]*InjBranch[3*k+2]

                #     idx1 = np.where(self.ini.Init_net_coe0[:,1] == InjBranch[3*k])[0]
                #     idx2 = np.where(self.ini.Init_net_coe0[idx1,0] == InjBranch[3*k+1])[0]
                #     len2 = len(idx2)
                #     if len2>0:
                #         for ii in range(len2):
                #             idx = idx1[idx2[ii]]
                #             # print(-brch_Ipre[idx  ] + brch_Ipre[idx+6]*InjBranch[3*k+2])
                #             # print(brch_Ipre[idx  ] - brch_Ipre[idx+6]*InjBranch[3*k+2])
                #             ia = ia - brch_Ipre[idx  ] + brch_Ipre[idx+6]*InjBranch[3*k+2]
                #             ib = ib - brch_Ipre[idx+1] + brch_Ipre[idx+7]*InjBranch[3*k+2]
                #             ic = ic - brch_Ipre[idx+2] + brch_Ipre[idx+8]*InjBranch[3*k+2]
                #########

                ia = 0
                ib = 0
                ic = 0
                for k in range(lenIB):
                    # print([self.pfd.bus_num[int(InjBranch[3*k])],self.pfd.bus_num[int(InjBranch[3*k+1])]])
                    idx1 = np.where(self.ini.Init_net_coe0[:, 0] == InjBranch[3 * k])[0]
                    idx2 = np.where(
                        self.ini.Init_net_coe0[idx1, 1] == InjBranch[3 * k + 1]
                    )[0]
                    len2 = len(idx2)
                    if len2 > 0:
                        for ii in range(len2):
                            idx = idx1[idx2[ii]]
                            # print([idx,-brch_Ipre[idx  ] - brch_Ipre[idx+3]*InjBranch[3*k+2]])
                            if InjBranch[3 * k + 2] == 0:
                                ia = ia - brch_Ipre[idx]
                                ib = ib - brch_Ipre[idx + 1]
                                ic = ic - brch_Ipre[idx + 2]
                            else:
                                ia = (
                                    ia
                                    - brch_Ipre[idx]
                                    - brch_Ipre[idx + 3] * InjBranch[3 * k + 2]
                                )
                                ib = (
                                    ib
                                    - brch_Ipre[idx + 1]
                                    - brch_Ipre[idx + 4] * InjBranch[3 * k + 2]
                                )
                                ic = (
                                    ic
                                    - brch_Ipre[idx + 2]
                                    - brch_Ipre[idx + 5] * InjBranch[3 * k + 2]
                                )

                    idx1 = np.where(self.ini.Init_net_coe0[:, 1] == InjBranch[3 * k])[0]
                    idx2 = np.where(
                        self.ini.Init_net_coe0[idx1, 0] == InjBranch[3 * k + 1]
                    )[0]
                    len2 = len(idx2)
                    if len2 > 0:
                        for ii in range(len2):
                            idx = idx1[idx2[ii]]
                            # print(-brch_Ipre[idx  ] + brch_Ipre[idx+6]*InjBranch[3*k+2])
                            # print(brch_Ipre[idx  ] - brch_Ipre[idx+6]*InjBranch[3*k+2])
                            if InjBranch[3 * k + 2] == 0:
                                ia = ia - brch_Ipre[idx]
                                ib = ib - brch_Ipre[idx + 1]
                                ic = ic - brch_Ipre[idx + 2]
                            else:
                                ia = (
                                    ia
                                    - brch_Ipre[idx]
                                    + brch_Ipre[idx + 6] * InjBranch[3 * k + 2]
                                )
                                ib = (
                                    ib
                                    - brch_Ipre[idx + 1]
                                    + brch_Ipre[idx + 7] * InjBranch[3 * k + 2]
                                )
                                ic = (
                                    ic
                                    - brch_Ipre[idx + 2]
                                    + brch_Ipre[idx + 8] * InjBranch[3 * k + 2]
                                )

                # lenIB = int(len(InjBranch) / 3)
                # ia = 0
                # ib = 0
                # ic = 0
                # for k in range(lenIB):
                #     idx1 = np.where(self.ini.Init_net_coe0[:,0] == InjBranch[3*k])[0]
                #     idx2 = np.where(self.ini.Init_net_coe0[idx1,1] == InjBranch[3*k+1])[0]
                #     flag_lnxm = InjBranch[3*k+2]
                #     len2 = len(idx2)
                #     if len2>0:
                #         for ii in range(len2):
                #             idx = idx1[idx2[ii]]
                #             if flag_lnxm==0:
                #                 ia = ia - brch_Ipre[idx  ] - brch_Ipre[idx+3]
                #                 ib = ib - brch_Ipre[idx+1] - brch_Ipre[idx+4]
                #                 ic = ic - brch_Ipre[idx+2] - brch_Ipre[idx+5]
                #             elif flag_lnxm==1:
                #                 ia = ia - brch_Ipre[idx  ]
                #                 ib = ib - brch_Ipre[idx+1]
                #                 ic = ic - brch_Ipre[idx+2]

                #     idx1 = np.where(self.ini.Init_net_coe0[:,1] == InjBranch[3*k])[0]
                #     idx2 = np.where(self.ini.Init_net_coe0[idx1,0] == InjBranch[3*k+1])[0]
                #     len2 = len(idx2)
                #     if len2>0:
                #         for ii in range(len2):
                #             idx = idx1[idx2[ii]]
                #             if flag_lnxm==0:
                #                 ia = ia + brch_Ipre[idx  ] - brch_Ipre[idx+6]
                #                 ib = ib + brch_Ipre[idx+1] - brch_Ipre[idx+7]
                #                 ic = ic + brch_Ipre[idx+2] - brch_Ipre[idx+8]
                #             elif flag_lnxm==1:
                #                 ia = ia + brch_Ipre[idx  ]
                #                 ib = ib + brch_Ipre[idx+1]
                #                 ic = ic + brch_Ipre[idx+2]

                i_inj[3 * j + 0] = ia
                i_inj[3 * j + 1] = ib
                i_inj[3 * j + 2] = ic

            self.fft_iabc.append(i_inj)

            if len(self.fft_vabc) > self.fft_N:
                self.fft_vabc.popleft()
                self.fft_iabc.popleft()

            if len(self.fft_vabc) == self.fft_N:
                fft_res = np.fft.rfft(self.fft_vabc, self.fft_N, 0)

                vm = np.abs(fft_res[1, :]) * 2 / self.fft_N
                va = np.angle(fft_res[1, :])

                # plt.plot(np.abs(fft_res[1,:]) * 2 / self.fft_N)
                # plt.show()

                # plt.plot(np.abs(fft_res[2,:]) * 2 / self.fft_N)
                # plt.show()

                # plt.plot(np.abs(fft_res[3,:]) * 2 / self.fft_N)
                # plt.show()
                ## save the latest state only
                self.fft_vma = np.concatenate((vm, va))
                self.fft_vpn0 = numba_update_vpn0(
                    self.nbus + self.nibrepri, vm, va, Ainv
                )

                fft_res = np.fft.rfft(np.array(self.fft_iabc), self.fft_N, 0)
                im = np.abs(fft_res[1, :]) * 2 / self.fft_N
                ia = np.angle(fft_res[1, :])

                self.fft_ima = np.concatenate((im, ia))
                self.fft_ipn0 = numba_update_ipn0(1, im, ia, Ainv)

                self.Sinj = np.zeros(nbus_b, dtype=complex)
                self.Sinj_new = np.zeros(nbus_b, dtype=complex)
                n3bus = len(bus_emt) * 3
                for j in range(nbus_b):
                    busb = int(self.emt_zones[self.EMT_N - 1]["bus_b"][j])
                    idx = np.where(self.pfd.bus_num == busb)[0][0]

                    vbus_b = complex(
                        self.fft_vma[idx]
                        * np.cos(self.fft_vma[idx + n3bus] - self.pfd.ws * self.ts),
                        self.fft_vma[idx]
                        * np.sin(self.fft_vma[idx + n3bus] - self.pfd.ws * self.ts),
                    )
                    ibus_b = complex(
                        self.fft_ima[3 * j] * np.cos(self.fft_ima[3 * nbus_b + 3 * j]),
                        self.fft_ima[3 * j] * np.sin(self.fft_ima[3 * nbus_b + 3 * j]),
                    )
                    Sinj = np.array(vbus_b * np.conjugate(ibus_b) * self.pfd.basemva)
                    self.Sinj[j] = Sinj

                    # vat = self.fft_vabc[-1][idx]
                    # vbt = self.fft_vabc[-1][idx+len(self.pfd.bus_num)]
                    # vct = self.fft_vabc[-1][idx+2*len(self.pfd.bus_num)]
                    # iat = self.fft_iabc[-1][3*j]
                    # ibt = self.fft_iabc[-1][3*j+1]
                    # ict = self.fft_iabc[-1][3*j+2]
                    # Pinj_new = (vat*iat + vbt*ibt + vct*ict)*2/3
                    # Qinj_new = ((vbt - vct) * iat + (vct - vat) * ibt + (vat - vbt) * ict)/np.sqrt(3)*2/3
                    # Sinj_new = np.complex(Pinj_new, Qinj_new)
                    # self.Sinj_new[j] = Sinj_new

                # print([self.Sinj,self.Sinj_new])
            else:
                ## save the latest state only
                self.fft_vma = np.asarray(self.fft_vma)
                self.fft_vpn0 = np.asarray(self.fft_vpn0)

                self.fft_ima = np.asarray(self.fft_ima)
                self.fft_ipn0 = np.asarray(self.fft_ipn0)

            ## End if

            # print(self.Sinj)
            # if self.use_helics:
            # print([len(self.x)*self.ts, self.Sinj])
            ## End if

            try:  # Added Min (from Deepthi to save the power) 01302024
                # real_powers = [ str( self.Sinj[iii].real) for iii in range(len(self.Sinj))]
                real_powers = [
                    str(self.Sinj[iii].real) for iii in range(len(self.Sinj))
                ]
                real_powers_str = ",".join(real_powers)
                print(real_powers_str)
                # imag_powers = [ str( self.Sinj[iii].imag) for iii in range(len(self.Sinj))]
                # imag_powers_str = ",".join(imag_powers)
                # print(imag_powers_str)
            except Exception:
                pass
        ## End if compute_phasor

        return

    def helics_publish(self):
        if self.use_helics:
            pub = self.pub

            # #--------------------------------------------------------------------------------
            # ## Added on 8/14/2023 to introduce a smoothing effect to calculated PQ in ParaEMT
            # Tmea = 0.02
            # # Pmea_last can be initialized as self.Pinj before the first step of the simulation
            # Pmea = (1 - self.ts/Tmea) * Pmea_last + self.ts/Tmea * self.Pinj
            # Pmea_last = Pmea

            # # similar for Qmea
            # Qmea = (1 - self.ts/Tmea) * Qmea_last + self.ts/Tmea * self.Qinj
            # Qmea_last = Qmea
            # #--------------------------------------------------------------------------------

            print("broadcast values : ", self.Pinj, self.Qinj)
            for y in self.pub.keys():
                if y == "emt_Pinj_9_10":
                    h.helicsPublicationPublishDouble(pub[y], float(self.Pinj))
                if y == "emt_Qinj_9_10":
                    h.helicsPublicationPublishDouble(pub[y], float(self.Qinj))
            ## End for
        ## End if
        return

    def updateIhis(self, ini):
        numba_updateIhis(
            self.brch_range[:, self.rank],
            # Altered arguments
            self.brch_Ihis,
            self.brch_Ipre,
            self.node_Ihis,
            # Constant arguments
            self.Vsol,
            self.net_coe, #ini.Init_net_coe0,
            ini.Init_net_N,
        )

        return

    def helics_update(self):
        if self.use_helics:
            self.currenttime = h.helicsFederateRequestTime(self.cfed, self.deltat)

        return

    def force_numba_compilation(self, ts):
        t0 = time.time()

        pfd = self.pfd
        ini = self.ini
        dyd = self.dyd

        unused = numba_predictX(
            self.gen_range[:, self.rank],
            np.zeros(self.xp.pd_w.shape),
            np.zeros(self.xp.pd_id.shape),
            np.zeros(self.xp.pd_iq.shape),
            np.zeros(self.xp.pd_EFD.shape),
            np.zeros(self.xp.pd_u_d.shape),
            np.zeros(self.xp.pd_u_q.shape),
            np.zeros(self.xp.pd_dt.shape),
            np.zeros(self.xp.pv_dt_1.shape),
            np.zeros(self.xp.pv_w_1.shape),
            np.zeros(self.xp.pv_id_1.shape),
            np.zeros(self.xp.pv_iq_1.shape),
            np.zeros(self.xp.pv_ifd_1.shape),
            np.zeros(self.xp.pv_i1d_1.shape),
            np.zeros(self.xp.pv_i1q_1.shape),
            np.zeros(self.xp.pv_i2q_1.shape),
            np.zeros(self.xp.pv_v1_1.shape),
            np.zeros(self.xp.pv_EFD_1.shape),
            np.zeros(self.xp.pv_ed_1.shape),
            np.zeros(self.xp.pv_eq_1.shape),
            np.zeros(self.xp.pv_psyd_1.shape),
            np.zeros(self.xp.pv_psyq_1.shape),
            np.zeros(self.xp.pv_te_1.shape),
            np.zeros(self.xp.pv_u_d_1.shape),
            np.zeros(self.xp.pv_u_q_1.shape),
            np.zeros(self.xp.pv_i_d_1.shape),
            np.zeros(self.xp.pv_i_q_1.shape),
            np.zeros(self.xp.pv_p1_1.shape),
            np.zeros(self.xp.pv_p2_1.shape),
            np.zeros(self.xp.pv_p3_1.shape),
            np.zeros(self.xp.pv_pm_1.shape),
            np.zeros(self.x_pv_1.shape),
            np.zeros(self.x_pv_1.shape),
            np.zeros(self.x_pv_1.shape),
            # pfd.gen_bus,
            pfd.ws,
            dyd.gen_genrou_odr,
            dyd.exc_sexs_xi_st,
            dyd.exc_sexs_odr,
            ts,
            3,
        )

        unused = numba_updateIg(
            self.gen_range[:, self.rank],
            np.zeros(self.Igs.shape),
            np.zeros(self.Isg.shape),
            np.zeros(self.x_pv_1.shape),
            np.zeros(self.ed_mod.shape),
            np.zeros(self.eq_mod.shape),
            np.zeros(self.theta.shape),
            np.zeros(self.xp.pv_his_d_1.shape),
            np.zeros(self.xp.pv_his_fd_1.shape),
            np.zeros(self.xp.pv_his_1d_1.shape),
            np.zeros(self.xp.pv_his_q_1.shape),
            np.zeros(self.xp.pv_his_1q_1.shape),
            np.zeros(self.xp.pv_his_2q_1.shape),
            np.zeros(self.xp.pv_his_red_d_1.shape),
            np.zeros(self.xp.pv_his_red_q_1.shape),
            pfd.gen_bus,
            pfd.bus_num,
            dyd.base_Is,
            dyd.ec_Rfd,
            dyd.ec_Lad,
            dyd.gen_genrou_odr,
            ini.Init_mac_alpha,
            ini.Init_mac_Rd,
            ini.Init_mac_Rq,
            ini.Init_mac_Rd2,
            ini.Init_mac_Rq2,
            ini.Init_mac_Rd_coe,
            ini.Init_mac_Rq_coe,
            ini.Init_mac_Rav,
            ini.Init_net_IbaseA,
            self.xp.pv_i_d_1,
            self.xp.pv_u_d_1,
            self.xp.pv_EFD_1,
            self.xp.pv_i_q_1,
            self.xp.pv_u_q_1,
            self.xp.pd_EFD,
            self.xp.pd_u_d,
            self.xp.pd_u_q,
            self.xp.pd_id,
            self.xp.pd_iq,
            self.xp.pd_dt,
            self.flag_gentrip,
            self.i_gentrip,
        )

        numba_updateIibr(
            self.ibr_range[:, self.rank],
            np.zeros(self.Igi.shape),
            np.zeros(self.Iibr.shape),
            pfd.ibr_bus,
            pfd.bus_num,
            dyd.ibr_Ibase,
            ini.Init_net_IbaseA,
            dyd.ibr_wecc_odr,
            self.Vsol,
            self.x_ibr_pv_1,
            self.ts,
            self.x_bus_pv_1,
            dyd.bus_odr,
        )

        if self.loadmodel_option == 2:
            numba_updateIl(
                self.load_range[:, self.rank],
                np.zeros(self.Il.shape),
                np.zeros(self.Ild.shape),
                self.x_load_pv_1,
                self.x_bus_pv_1,
                pfd.bus_num,
                pfd.load_bus,
                pfd.ws,
                dyd.bus_odr,
                dyd.load_odr,
                1,
                self.ts,
                self.t_release_f,
            )

        numba_BusMea(
            self.bus_range[:, self.rank],
            np.zeros(self.nbus * dyd.bus_odr),
            self.Vsol,
            self.x_bus_pv_1,
            self.nbus,
            self.ts,
            self.t_release_f,
            pfd.ws,
            dyd.bus_odr,
            dyd.vm_te,
            dyd.pll_ke,
            dyd.pll_te,
            1,
        )

        numba_updateX(
            self.gen_range[:, self.rank],
            np.ones(self.x_pv_1.shape),
            np.ones(self.xp.nx_ed.shape),
            np.ones(self.xp.nx_eq.shape),
            np.ones(self.xp.nx_id.shape),
            np.ones(self.xp.nx_iq.shape),
            np.ones(self.xp.nx_ifd.shape),
            np.ones(self.xp.nx_i1d.shape),
            np.ones(self.xp.nx_i1q.shape),
            np.ones(self.xp.nx_i2q.shape),
            np.ones(self.xp.nx_psyd.shape),
            np.ones(self.xp.nx_psyq.shape),
            np.ones(self.xp.nx_psyfd.shape),
            np.ones(self.xp.nx_psy1q.shape),
            np.ones(self.xp.nx_psy1d.shape),
            np.ones(self.xp.nx_psy2q.shape),
            np.ones(self.xp.nx_te.shape),
            np.ones(self.xp.nx_w.shape),
            np.ones(self.xp.nx_EFD.shape),
            np.ones(self.xp.nx_dt.shape),
            np.ones(self.xp.nx_v1.shape),
            np.ones(self.xp.nx_pm.shape),
            self.xp.pd_dt,
            self.xp.pd_EFD,
            self.xp.pv_his_fd_1,
            self.xp.pv_his_1d_1,
            self.xp.pv_his_1q_1,
            self.xp.pv_his_2q_1,
            self.xp.pv_dt_1,
            np.ones(self.xp.pv_w_1.shape),
            self.xp.pv_EFD_1,
            pfd.gen_bus,
            pfd.bus_num,
            pfd.ws,
            pfd.basemva,
            pfd.gen_MVA_base,
            dyd.gen_H,
            dyd.gen_D,
            dyd.gen_genrou_n,
            dyd.gen_genrou_odr,
            dyd.gen_genrou_xi_st,
            dyd.ec_Rfd,
            dyd.ec_Lad,
            dyd.ec_Laq,
            dyd.ec_Ll,
            dyd.ec_Lffd,
            dyd.ec_L11q,
            dyd.ec_L11d,
            dyd.ec_Lf1d,
            dyd.ec_L22q,
            dyd.pss_ieeest_A1,
            dyd.pss_ieeest_A2,
            dyd.pss_ieeest_A3,
            dyd.pss_ieeest_A4,
            dyd.pss_ieeest_A5,
            dyd.pss_ieeest_A6,
            dyd.pss_ieeest_T1,
            dyd.pss_ieeest_T2,
            dyd.pss_ieeest_T3,
            dyd.pss_ieeest_T4,
            dyd.pss_ieeest_T5,
            dyd.pss_ieeest_T6,
            dyd.pss_ieeest_KS,
            dyd.pss_ieeest_LSMAX,
            dyd.pss_ieeest_LSMIN,
            dyd.pss_ieeest_VCL,
            dyd.pss_ieeest_VCU,
            dyd.pss_ieeest_idx,
            dyd.pss_ieeest_odr,
            dyd.pss_ieeest_xi_st,
            dyd.exc_sexs_TA,
            dyd.exc_sexs_TB,
            dyd.exc_sexs_K,
            dyd.exc_sexs_TE,
            dyd.exc_sexs_Emin,
            dyd.exc_sexs_Emax,
            dyd.exc_sexs_idx,
            dyd.exc_sexs_n,
            dyd.exc_sexs_odr,
            dyd.exc_sexs_xi_st,
            dyd.gov_type,
            dyd.gov_tgov1_bus,
            dyd.gov_tgov1_id,
            dyd.gov_tgov1_Dt,
            dyd.gov_tgov1_R,
            dyd.gov_tgov1_T1,
            dyd.gov_tgov1_T2,
            dyd.gov_tgov1_T3,
            dyd.gov_tgov1_Vmax,
            dyd.gov_tgov1_Vmin,
            dyd.gov_tgov1_idx,
            dyd.gov_tgov1_n,
            dyd.gov_tgov1_odr,
            dyd.gov_tgov1_xi_st,
            dyd.gov_hygov_bus,
            dyd.gov_hygov_id,
            dyd.gov_hygov_At,
            dyd.gov_hygov_Dturb,
            dyd.gov_hygov_GMAX,
            dyd.gov_hygov_GMIN,
            dyd.gov_hygov_R,
            dyd.gov_hygov_TW,
            dyd.gov_hygov_Tf,
            dyd.gov_hygov_Tg,
            dyd.gov_hygov_Tr,
            dyd.gov_hygov_VELM,
            dyd.gov_hygov_qNL,
            dyd.gov_hygov_r,
            dyd.gov_hygov_idx,
            dyd.gov_hygov_n,
            dyd.gov_hygov_odr,
            dyd.gov_hygov_xi_st,
            dyd.gov_gast_bus,
            dyd.gov_gast_id,
            dyd.gov_gast_R,
            dyd.gov_gast_LdLmt,
            dyd.gov_gast_KT,
            dyd.gov_gast_T1,
            dyd.gov_gast_T2,
            dyd.gov_gast_T3,
            dyd.gov_gast_VMIN,
            dyd.gov_gast_VMAX,
            dyd.gov_gast_Dturb,
            dyd.gov_gast_idx,
            dyd.gov_gast_n,
            dyd.gov_gast_odr,
            dyd.gov_gast_xi_st,
            dyd.bus_odr,
            ini.Init_mac_Rav,
            ini.Init_mac_Rd1,
            ini.Init_mac_Rd1inv,
            ini.Init_mac_Rq1,
            ini.Init_mac_Rq1inv,
            ini.Init_mac_Gequiv,
            ini.tgov1_2gen,
            ini.hygov_2gen,
            ini.gast_2gen,
            self.vref,
            self.gref,
            self.Vsol,
            self.Isg,
            self.ed_mod,
            self.eq_mod,
            self.vref_1,
            self.x_bus_pv_1,
            self.ts,
            self.flag_gentrip,
            self.i_gentrip,
        )

        numba_updateXibr(
            self.ibr_range[:, self.rank],
            # Altered Arguments
            np.zeros(self.x_ibr_pv_1.shape),
            # Constant Arguments
            pfd.ibr_bus,
            pfd.bus_num,
            pfd.ws,
            pfd.basemva,
            pfd.ibr_MVA_base,
            dyd.ibr_regca_Volim,
            dyd.ibr_regca_Khv,
            dyd.ibr_regca_Lvpnt0,
            dyd.ibr_regca_Lvpnt1,
            dyd.ibr_regca_Tg,
            dyd.ibr_regca_Iqrmax,
            dyd.ibr_regca_Iqrmin,
            dyd.ibr_regca_Tfltr,
            dyd.ibr_regca_Zerox,
            dyd.ibr_regca_Brkpt,
            dyd.ibr_regca_Rrpwr,
            dyd.ibr_reecb_PQFLAG,
            dyd.ibr_reecb_PFFLAG,
            dyd.ibr_reecb_VFLAG,
            dyd.ibr_reecb_QFLAG,
            dyd.ibr_reecb_Imax,
            dyd.ibr_reecb_Vdip,
            dyd.ibr_reecb_Vup,
            dyd.ibr_reecb_Trv,
            dyd.ibr_reecb_dbd1,
            dyd.ibr_reecb_dbd2,
            dyd.ibr_reecb_Kqv,
            dyd.ibr_reecb_Iqll,
            dyd.ibr_reecb_Iqhl,
            dyd.ibr_reecb_Tp,
            dyd.ibr_reecb_Qmin,
            dyd.ibr_reecb_Qmax,
            dyd.ibr_reecb_Kqp,
            dyd.ibr_reecb_Kqi,
            dyd.ibr_reecb_Vmin,
            dyd.ibr_reecb_Vmax,
            dyd.ibr_reecb_Kvp,
            dyd.ibr_reecb_Kvi,
            dyd.ibr_reecb_Tiq,
            dyd.ibr_reecb_dPmin,
            dyd.ibr_reecb_dPmax,
            dyd.ibr_reecb_Pmin,
            dyd.ibr_reecb_Pmax,
            dyd.ibr_reecb_Tpord,
            dyd.ibr_repca_FFlag,
            dyd.ibr_repca_VCFlag,
            dyd.ibr_repca_RefFlag,
            dyd.ibr_repca_fdbd1,
            dyd.ibr_repca_fdbd2,
            dyd.ibr_repca_Ddn,
            dyd.ibr_repca_Dup,
            dyd.ibr_repca_Tp,
            dyd.ibr_repca_femin,
            dyd.ibr_repca_femax,
            dyd.ibr_repca_Kpg,
            dyd.ibr_repca_Kig,
            dyd.ibr_repca_Pmin,
            dyd.ibr_repca_Pmax,
            dyd.ibr_repca_Tg,
            dyd.ibr_repca_Rc,
            dyd.ibr_repca_Xc,
            dyd.ibr_repca_Kc,
            dyd.ibr_repca_Tfltr,
            dyd.ibr_repca_dbd1,
            dyd.ibr_repca_dbd2,
            dyd.ibr_repca_emin,
            dyd.ibr_repca_emax,
            dyd.ibr_repca_Vfrz,
            dyd.ibr_repca_Kp,
            dyd.ibr_repca_Ki,
            dyd.ibr_repca_Qmin,
            dyd.ibr_repca_Qmax,
            dyd.ibr_repca_Tft,
            dyd.ibr_repca_Tfv,
            # dyd.ibr_pll_ke,
            # dyd.ibr_pll_te,
            dyd.ibr_wecc_odr,
            ini.Init_ibr_regca_Qgen0,
            ini.Init_ibr_reecb_pfaref,
            ini.Init_ibr_reecb_Vref0,
            ini.Init_ibr_repca_Pref_out,
            self.Vsol,
            self.x_bus_pv_1,
            dyd.bus_odr,
            self.Iibr,
            ts,
        )

        if self.loadmodel_option == 2:
            numba_updateXl(
                self.load_range[:, self.rank],
                self.x_load_pv_1.copy(),
                pfd.load_bus,
                pfd.bus_num,
                dyd.load_odr,
                dyd.bus_odr,
                self.x_load_pv_1,
                self.Vsol,
                self.Ild,
                1,
            )

        numba_updateIhis(
            self.brch_range[:, self.rank],
            np.zeros(self.brch_Ihis.shape),
            np.zeros(self.brch_Ipre.shape),
            np.zeros(self.node_Ihis.shape),
            self.Vsol,
            ini.Init_net_coe0,
            ini.Init_net_N,
        )

        numba_update_vpn0(
            self.nbus,
            np.zeros(3 * self.nbus),
            np.zeros(3 * self.nbus),
            Ainv,
        )

        t1 = time.time()

        self.nmb_comp_time = t1 - t0

        return

    # def CalcPqInj4GridPack(self, pfd, dyd, tn):
    #     # FFT
    #     if len(self.fft_iabc) == self.fft_N:
    #         data = np.array(self.fft_iabc)
    #         fft_res = np.fft.rfft(data, self.fft_N, 0)
    #         im = np.zeros(3)
    #         ia = np.zeros(3)
    #         for i in range(3):
    #             im[i] = abs(fft_res[1][i]) * 2 / self.fft_N
    #             ia[i] = np.angle(fft_res[1][i])

    #         ima = np.concatenate((im, ia))

    #         nbus = 1
    #         ipn0 = np.zeros(nbus*6)
    #         for i in range(nbus):
    #             iabc_phasor = np.asarray([ima[i]*np.exp(1j*ima[i+3*nbus]), ima[i+nbus]*np.exp(1j*ima[i+4*nbus]), ima[i+2*nbus]*np.exp(1j*ima[i+5*nbus])])
    #             ipn0i = np.dot(Ainv, iabc_phasor)
    #             ipn0[i] = abs(ipn0i[2])
    #             ipn0[i+3*nbus] = np.angle(ipn0i[2])
    #             ipn0[i+nbus] = abs(ipn0i[1])
    #             ipn0[i + 4 * nbus] = np.angle(ipn0i[1])
    #             ipn0[i+2*nbus] = abs(ipn0i[0])
    #             ipn0[i + 5 * nbus] = np.angle(ipn0i[0])

    #         ## save the latest state only
    #         self.fft_ima = ima
    #         self.fft_ipn0 = ipn0

    #     else:
    #         ## save the latest state only
    #         self.fft_ima = np.asarray(self.fft_ima)
    #         self.fft_ipn0 = np.asarray(self.fft_ipn0)

    #     v10 = complex(self.fft_vma[2] * np.cos(self.fft_vma[14]), self.fft_vma[2] * np.sin(self.fft_vma[14]))
    #     i11_10 = complex(self.fft_ima[0] * np.cos(self.fft_ima[3]), self.fft_ima[0] * np.sin(self.fft_ima[3]))
    #     self.Sinj = v10 * np.conjugate(i11_10) * pfd.basemva
    #     self.Pinj = self.Sinj.real
    #     self.Qinj = self.Sinj.imag

    def StepChange(self, dyd, ini, tn):
        if self.flag_sc == 1 and tn * self.ts >= self.t_sc:
            self.flag_sc = 0

            if self.flag_exc_gov == 1:
                # gov pm
                if dyd.gov_type[self.i_gen_sc] == dyd.gov_model_map["TGOV1"]:
                    idx_gov = np.where(dyd.gov_tgov1_idx == self.i_gen_sc)[0][0]
                    # self.gref[ini.tgov1_2gen[idx_gov]] = self.gref[ini.tgov1_2gen[idx_gov]] + self.dsp
                    self.gref[ini.tgov1_2gen[idx_gov]] += self.dsp
                    # self.flag_sc = 0

                elif dyd.gov_type[self.i_gen_sc] == dyd.gov_model_map["HYGOV"]:
                    idx_gov = np.where(dyd.gov_hygov_idx == self.i_gen_sc)[0][0]
                    # self.gref[ini.hygov_2gen[idx_gov]] = self.gref[ini.hygov_2gen[idx_gov]] + self.dsp
                    self.gref[ini.hygov_2gen[idx_gov]] += self.dsp
                    # self.flag_sc = 0

                elif dyd.gov_type[self.i_gen_sc] == dyd.gov_model_map["GAST"]:
                    idx_gov = np.where(dyd.gov_gast_idx == self.i_gen_sc)[0][0]
                    # self.gref[ini.gast_2gen[idx_gov]] = self.gref[ini.gast_2gen[idx_gov]] + self.dsp
                    self.gref[ini.gast_2gen[idx_gov]] += self.dsp
                    # self.flag_sc = 0

            elif self.flag_exc_gov == 0:
                # ini.Init_mac_vref[self.i_gen_sc] = ini.Init_mac_vref[self.i_gen_sc] + self.dsp
                # ini.Init_mac_vref[self.i_gen_sc] += self.dsp
                self.vref[self.i_gen_sc] += self.dsp
                # self.flag_sc = 0

        return

    def GenTrip(self, pfd, dyd, ini, tn):
        matrix_changed = False

        if self.t_gentrip and tn * self.ts >= self.t_gentrip:
            if self.flag_gentrip == 1:
                genbus_idx = int(
                    np.where(pfd.bus_num == pfd.gen_bus[self.i_gentrip])[0]
                )
                nbus = self.nbus
                isg_idx = 3 * self.i_gentrip

                self.Igs[genbus_idx] -= self.Isg[isg_idx]
                self.Igs[genbus_idx + nbus] -= self.Isg[isg_idx + 1]
                self.Igs[genbus_idx + 2 * nbus] -= self.Isg[isg_idx + 2]

                # self.Isg[:] = 0.0

                ini.InitNet(
                    pfd, dyd, self.ts, self.loadmodel_option
                )  # to re-create rows, cols, data for G0
                ini.MergeIbrG(pfd, dyd, self.ts, [])
                ini.MergeMacG(pfd, dyd, self.ts, self.i_gentrip)
                self.flag_gentrip = 0

                matrix_changed = True

        return matrix_changed

    def Re_Init(self, pfd, dyd, ini, tn):
        nbus = self.nbus
        ngen = self.ngen
        nebr = self.nibrepri

        # updateI_BU
        brch_Ihis = self.brch_Ihis
        Vsol = self.Vsol_1
        Init_net_coe0 = self.net_coe

        node_Ihis = self.node_Ihis
        node_Ihis[:] = 0.0
        brch_Ipre = self.brch_Ipre

        brch_range = self.brch_range[:, self.rank]
        for i in range(brch_range[0], brch_range[1]):
            if np.sign(Init_net_coe0[i, 3]) == 1:
                c1 = 1
                c2 = 0
            else:
                c1 = 0
                c2 = 1

            Fidx = int(Init_net_coe0[i, 0].real)
            Tidx = int(Init_net_coe0[i, 1].real)

            #### IF CLAUSE ####

            if Init_net_coe0[i, 1] == -1:
                if Init_net_coe0[i, 2] == 0:
                    continue
                brch_Ihis_temp = (
                    c1 * Init_net_coe0[i, 3] * brch_Ipre[i]
                    + c2 * np.real(Init_net_coe0[i, 4]) * Vsol[Fidx]
                )

            #### ELSE CLAUSE ####

            else:
                brch_Ihis_temp = c1 * Init_net_coe0[i, 3] * brch_Ipre[i] + c2 * np.real(
                    Init_net_coe0[i, 4]
                ) * (Vsol[Fidx] - Vsol[Tidx])
                node_Ihis[Tidx] += brch_Ihis_temp.real

            #### END IF-ELSE ####

            brch_Ihis[i] = brch_Ihis_temp.real
            node_Ihis[Fidx] -= brch_Ihis_temp.real

        #### END FOR LOOP ####

        ## predictX

        # numba_predictX(
        #     self.gen_range[:,self.rank],
        #     ## Altered Arguments ##
        #     # were returned directly
        #     self.xp.pd_w,
        #     self.xp.pd_id,
        #     self.xp.pd_iq,
        #     self.xp.pd_EFD,
        #     self.xp.pd_u_d,
        #     self.xp.pd_u_q,
        #     self.xp.pd_dt,
        #     # were in point_one_tuple
        #     self.xp.pv_dt_1,
        #     self.xp.pv_w_1,
        #     self.xp.pv_id_1,
        #     self.xp.pv_iq_1,
        #     self.xp.pv_ifd_1,
        #     self.xp.pv_i1d_1,
        #     self.xp.pv_i1q_1,
        #     self.xp.pv_i2q_1,
        #     self.xp.pv_v1_1,
        #     self.xp.pv_EFD_1,
        #     self.xp.pv_ed_1,
        #     self.xp.pv_eq_1,
        #     self.xp.pv_psyd_1,
        #     self.xp.pv_psyq_1,
        #     self.xp.pv_te_1,
        #     self.xp.pv_u_d_1,
        #     self.xp.pv_u_q_1,
        #     self.xp.pv_i_d_1,
        #     self.xp.pv_i_q_1,
        #     self.xp.pv_p1_1,
        #     self.xp.pv_p2_1,
        #     self.xp.pv_p3_1,
        #     self.xp.pv_pm_1,
        #     ## Constant Arguments ##
        #     x_pv_1,
        #     x_pv_2,
        #     x_pv_3,
        #     pfd.ws,
        #     dyd.gen_genrou_odr,
        #     dyd.exc_sexs_xi_st,
        #     dyd.exc_sexs_odr,
        #     ts,
        #     xlen,
        # )

        pv_dt_1 = np.zeros(ngen)
        pd_dt = np.zeros(ngen)

        pv_w_1 = np.zeros(ngen)
        pv_w_2 = np.zeros(ngen)
        pd_w = np.zeros(ngen)

        pv_id_1 = np.zeros(ngen)
        pv_id_2 = np.zeros(ngen)
        pd_id = np.zeros(ngen)

        pv_iq_1 = np.zeros(ngen)
        pv_iq_2 = np.zeros(ngen)
        pd_iq = np.zeros(ngen)

        pv_ifd_1 = np.zeros(ngen)
        pv_i1d_1 = np.zeros(ngen)
        pv_i1q_1 = np.zeros(ngen)
        pv_i2q_1 = np.zeros(ngen)

        pv_ed_1 = np.zeros(ngen)
        pv_eq_1 = np.zeros(ngen)

        pv_EFD_1 = np.zeros(ngen)
        pv_EFD_2 = np.zeros(ngen)
        pd_EFD = np.zeros(ngen)

        pv_psyd_1 = np.zeros(ngen)
        pv_psyd_2 = np.zeros(ngen)

        pv_psyq_1 = np.zeros(ngen)
        pv_psyq_2 = np.zeros(ngen)

        pv_u_d_1 = np.zeros(ngen)
        pv_u_d_2 = np.zeros(ngen)
        pd_u_d = np.zeros(ngen)

        pv_u_q_1 = np.zeros(ngen)
        pv_u_q_2 = np.zeros(ngen)
        pd_u_q = np.zeros(ngen)

        gen_range = self.gen_range[:, self.rank]
        for i in range(gen_range[0], gen_range[1]):
            idx = i * dyd.gen_genrou_odr + dyd.gen_genrou_xi_st

            pv_dt_1[i] = self.x_pred[0][idx + 0]

            pv_w_1[i] = self.x_pred[0][idx + 1]
            pv_w_2[i] = self.x_pred[1][idx + 1]

            pv_id_1[i] = self.x_pred[0][idx + 2]
            pv_id_2[i] = self.x_pred[1][idx + 2]

            pv_iq_1[i] = self.x_pred[0][idx + 3]
            pv_iq_2[i] = self.x_pred[1][idx + 3]

            pv_ifd_1[i] = self.x_pred[0][idx + 4]
            pv_i1d_1[i] = self.x_pred[0][idx + 5]
            pv_i1q_1[i] = self.x_pred[0][idx + 6]
            pv_i2q_1[i] = self.x_pred[0][idx + 7]

            pv_ed_1[i] = self.x_pred[0][idx + 8]
            pv_eq_1[i] = self.x_pred[0][idx + 9]

            pv_psyd_1[i] = self.x_pred[0][idx + 10]
            pv_psyd_2[i] = self.x_pred[1][idx + 10]

            pv_psyq_1[i] = self.x_pred[0][idx + 11]
            pv_psyq_2[i] = self.x_pred[1][idx + 11]

            # exc
            idx_exc = i * dyd.exc_sexs_odr + dyd.exc_sexs_xi_st
            pv_EFD_1[i] = self.x_pred[0][idx_exc + 1]
            pv_EFD_2[i] = self.x_pred[1][idx_exc + 1]

            pv_u_d_1[i] = -pv_psyq_1[i] * pv_w_1[i] / pfd.ws
            pv_u_d_2[i] = -pv_psyq_2[i] * pv_w_2[i] / pfd.ws

            pv_u_q_1[i] = pv_psyd_1[i] * pv_w_1[i] / pfd.ws
            pv_u_q_2[i] = pv_psyd_2[i] * pv_w_2[i] / pfd.ws

            pd_w[i] = 2.0 * pv_w_1[i] - pv_w_2[i]
            pd_id[i] = 2.0 * pv_id_1[i] - pv_id_2[i]
            pd_iq[i] = 2.0 * pv_iq_1[i] - pv_iq_2[i]
            pd_EFD[i] = 2.0 * pv_EFD_1[i] - pv_EFD_2[i]
            pd_u_d[i] = 2.0 * pv_u_d_1[i] - pv_u_d_2[i]
            pd_u_q[i] = 2.0 * pv_u_q_1[i] - pv_u_q_2[i]

            pd_dt[i] = pv_dt_1[i] + self.ts / 2 * (pd_w[i] + pv_w_1[i]) / 2

        ####  updateIg

        # numba_updateIg(
        #     # Indices for looping
        #     self.gen_range[:,self.rank],
        #     # Altered Arguments
        #     self.Igs,
        #     self.Isg,
        #     self.x_pv_1,
        #     self.ed_mod,
        #     self.eq_mod,
        #     self.theta,
        #     self.xp.pv_his_d_1,
        #     self.xp.pv_his_fd_1,
        #     self.xp.pv_his_1d_1,
        #     self.xp.pv_his_q_1,
        #     self.xp.pv_his_1q_1,
        #     self.xp.pv_his_2q_1,
        #     self.xp.pv_his_red_d_1,
        #     self.xp.pv_his_red_q_1,
        #     # Constant Arguments
        #     # pfd
        #     pfd.gen_bus,
        #     pfd.bus_num,
        #     # dyd
        #     dyd.base_Is,
        #     dyd.ec_Rfd,
        #     dyd.ec_Lad,
        #     dyd.gen_genrou_odr,
        #     # ini
        #     ini.Init_mac_alpha,
        #     ini.Init_mac_Rd,
        #     ini.Init_mac_Rq,
        #     ini.Init_mac_Rd2,
        #     ini.Init_mac_Rq2,
        #     ini.Init_mac_Rd_coe,
        #     ini.Init_mac_Rq_coe,
        #     ini.Init_mac_Rav,
        #     ini.Init_net_IbaseA,
        #     # self.xp
        #     self.xp.pv_i_d_1,
        #     self.xp.pv_u_d_1,
        #     self.xp.pv_EFD_1,
        #     self.xp.pv_i_q_1,
        #     self.xp.pv_u_q_1,
        #     self.xp.pd_EFD,
        #     self.xp.pd_u_d,
        #     self.xp.pd_u_q,
        #     self.xp.pd_id,
        #     self.xp.pd_iq,
        #     self.xp.pd_dt,
        #     self.flag_gentrip,
        #     self.i_gentrip,
        # )

        # TODO: Should this result get overwritten and propagated forward?
        # Igs = self.Igs
        # Igs[:] = 0.0
        Igs = np.zeros(self.Igs.shape)

        Ias_n = Igs[:nbus]
        Ibs_n = Igs[nbus : 2 * nbus]
        Ics_n = Igs[2 * nbus :]

        pv_his_d_1 = np.zeros(ngen)
        pv_his_fd_1 = np.zeros(ngen)
        pv_his_1d_1 = np.zeros(ngen)
        pv_his_q_1 = np.zeros(ngen)
        pv_his_1q_1 = np.zeros(ngen)
        pv_his_2q_1 = np.zeros(ngen)
        pv_his_red_d_1 = np.zeros(ngen)
        pv_his_red_q_1 = np.zeros(ngen)
        ed_mod = np.zeros(ngen)
        eq_mod = np.zeros(ngen)
        theta = np.zeros(ngen)

        for i in range(gen_range[0], gen_range[1]):
            if self.flag_gentrip == 0 and i == self.i_gentrip:
                continue

            EFD2efd = dyd.ec_Rfd[i] / dyd.ec_Lad[i]

            pv_i_d_1 = [-pv_id_1[i], pv_ifd_1[i], pv_i1d_1[i]]
            pv_i_q_1 = [-pv_iq_1[i], pv_i1q_1[i], pv_i2q_1[i]]

            temp1 = np.sum(ini.Init_mac_Rd2[i, 0, :] * pv_i_d_1)
            pv_his_d_1_temp = (
                -ini.Init_mac_alpha[i] * pv_ed_1[i]
                + ini.Init_mac_alpha[i] * pv_u_d_1[i]
                + temp1
            )
            pv_his_d_1[i] = pv_his_d_1_temp

            temp2 = np.sum(ini.Init_mac_Rd2[i, 1, :] * pv_i_d_1)
            pv_his_fd_1_temp = -ini.Init_mac_alpha[i] * pv_EFD_1[i] * EFD2efd + temp2
            pv_his_fd_1[i] = pv_his_fd_1_temp

            temp3 = np.sum(ini.Init_mac_Rd2[i, 2, :] * pv_i_d_1)
            pv_his_1d_1[i] = temp3

            temp4 = np.sum(ini.Init_mac_Rq2[i, 0, :] * pv_i_q_1)
            pv_his_q_1_temp = (
                -ini.Init_mac_alpha[i] * pv_eq_1[i]
                + ini.Init_mac_alpha[i] * pv_u_q_1[i]
                + temp4
            )
            pv_his_q_1[i] = pv_his_q_1_temp

            temp5 = np.sum(ini.Init_mac_Rq2[i, 1, :] * pv_i_q_1)
            pv_his_1q_1[i] = temp5

            temp6 = np.sum(ini.Init_mac_Rq2[i, 2, :] * pv_i_q_1)
            pv_his_2q_1[i] = temp6

            pv_his_red_d_1_temp = pv_his_d_1_temp - (
                ini.Init_mac_Rd_coe[i, 0] * (pv_his_fd_1_temp - pd_EFD[i] * EFD2efd)
                + ini.Init_mac_Rd_coe[i, 1] * temp3
            )
            pv_his_red_q_1_temp = pv_his_q_1_temp - (
                ini.Init_mac_Rq_coe[i, 0] * temp5 + ini.Init_mac_Rq_coe[i, 1] * temp6
            )
            pv_his_red_d_1[i] = pv_his_red_d_1_temp
            pv_his_red_q_1[i] = pv_his_red_q_1_temp

            ed_temp = pd_u_d[i] + pv_his_red_d_1_temp
            eq_temp = pd_u_q[i] + pv_his_red_q_1_temp

            ed_mod_temp = (
                ed_temp - (ini.Init_mac_Rd[i] - ini.Init_mac_Rq[i]) / 2.0 * pd_id[i]
            )
            eq_mod_temp = (
                eq_temp + (ini.Init_mac_Rd[i] - ini.Init_mac_Rq[i]) / 2.0 * pd_iq[i]
            )

            id_src_temp = ed_mod_temp / ini.Init_mac_Rav[i]
            iq_src_temp = eq_mod_temp / ini.Init_mac_Rav[i]

            ed_mod[i] = ed_mod_temp
            eq_mod[i] = eq_mod_temp

            # theta
            genbus_idx = np.where(pfd.bus_num == pfd.gen_bus[i])[0][0]
            theta[i] = pd_dt[i] - np.pi / 2.0

            iPk = np.array(
                [
                    [np.cos(theta[i]), -np.sin(theta[i]), 1.0],
                    [
                        np.cos(theta[i] - np.pi * 2.0 / 3.0),
                        -np.sin(theta[i] - np.pi * 2.0 / 3.0),
                        1.0,
                    ],
                    [
                        np.cos(theta[i] + np.pi * 2.0 / 3.0),
                        -np.sin(theta[i] + np.pi * 2.0 / 3.0),
                        1.0,
                    ],
                ]
            )
            res = iPk[:, 0] * id_src_temp + iPk[:, 1] * iq_src_temp

            coef = dyd.base_Is[i] / (ini.Init_net_IbaseA[genbus_idx] * 1000.0)
            Ias_n[genbus_idx] += res[0] * coef
            Ibs_n[genbus_idx] += res[1] * coef
            Ics_n[genbus_idx] += res[2] * coef

        #### udpate Igi
        nbus = len(pfd.bus_num)
        nibr = len(pfd.ibr_bus)

        # TODO: Should this result get overwritten and propagated forward?
        Igi = np.zeros(self.Igi.shape)
        # Igi = self.Igi
        # Igi[:] = 0.0

        Iai_n = Igi[:nbus]
        Ibi_n = Igi[nbus : 2 * nbus]
        Ici_n = Igi[2 * nbus :]

        if dyd.ibr_wecc_n > 0:
            ibr_range = self.ibr_range[:, self.rank]
            for i in range(ibr_range[0], ibr_range[1]):
                ibrbus_idx = np.where(pfd.bus_num == pfd.ibr_bus[i])[0][0]

                regca_s0_1 = self.x_ibr_pv_1[i * dyd.ibr_wecc_odr + 0]
                regca_s1_1 = self.x_ibr_pv_1[i * dyd.ibr_wecc_odr + 1]
                regca_i1_1 = self.x_ibr_pv_1[i * dyd.ibr_wecc_odr + 5]
                regca_i2_1 = self.x_ibr_pv_1[i * dyd.ibr_wecc_odr + 6]

                pll_de_1 = self.x_bus_pv_1[ibrbus_idx * dyd.bus_odr + 1]
                pll_we_1 = self.x_bus_pv_1[ibrbus_idx * dyd.bus_odr + 2]

                theta = pll_de_1 + self.ts * pll_we_1 * 2 * np.pi * 60
                iPk = np.asarray(
                    [
                        [np.cos(theta), -np.sin(theta), 1.0],
                        [
                            np.cos(theta - np.pi * 2.0 / 3.0),
                            -np.sin(theta - np.pi * 2.0 / 3.0),
                            1.0,
                        ],
                        [
                            np.cos(theta + np.pi * 2.0 / 3.0),
                            -np.sin(theta + np.pi * 2.0 / 3.0),
                            1.0,
                        ],
                    ]
                )
                ip = regca_s0_1 * regca_i2_1
                iq = -regca_s1_1 - regca_i1_1
                res = []
                for j in range(3):
                    res.append(iPk[j][0] * ip + iPk[j][1] * iq)
                ## End for

                Iai_n[ibrbus_idx] = Iai_n[ibrbus_idx] + res[0] * dyd.ibr_Ibase[i] / (
                    ini.Init_net_IbaseA[ibrbus_idx] * 1000.0
                )
                Ibi_n[ibrbus_idx] = Ibi_n[ibrbus_idx] + res[1] * dyd.ibr_Ibase[i] / (
                    ini.Init_net_IbaseA[ibrbus_idx] * 1000.0
                )
                Ici_n[ibrbus_idx] = Ici_n[ibrbus_idx] + res[2] * dyd.ibr_Ibase[i] / (
                    ini.Init_net_IbaseA[ibrbus_idx] * 1000.0
                )

            ## End for

        ## End if

        Igi_epri = np.zeros(self.Igi_epri.shape)

        if dyd.ibr_epri_n > 0:
            ebr_range = self.ebr_range[:, self.rank]
            N1 = len(pfd.bus_num)
            N3 = N1 * 3

            Nibr = dyd.ibr_epri_n
            Nbch = len(ini.Init_net_coe0) - 6 * Nibr

            for i in range(ebr_range[0], ebr_range[1]):
                ibri = self.ibr_epri[i]

                # ibri.cTime = (tn-1+1)*self.ts

                ibrbus = dyd.ibr_epri_bus[i]
                ibrid = dyd.ibr_epri_id[i]
                # if len(ibrid)==1:
                #     ibrid = ibrid + ' '

                ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0]
                ibrid_idx = np.where(pfd.ibr_id[ibrbus_idx] == ibrid)[0][0]
                ibrbus_idx = ibrbus_idx[ibrid_idx]

                bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]
                kVbase, IBR_MVA_base = (
                    pfd.bus_basekV[bus_idx],
                    pfd.ibr_MVA_base[ibrbus_idx],
                )
                kAbase = IBR_MVA_base / kVbase / np.sqrt(3.0)

                coe_idx = Nbch + 6 * i

                # Should be updated by the network solution based on model output
                # from previous time step
                # Va, Vb, Vc
                # ibri.cExternalInputs[0] = kVbase*np.sqrt(2.0/3.0)*self.Vsol[N3+i]
                # ibri.cExternalInputs[1] = kVbase*np.sqrt(2.0/3.0)*self.Vsol[N3+i+dyd.ibr_epri_n]
                # ibri.cExternalInputs[2] = kVbase*np.sqrt(2.0/3.0)*self.Vsol[N3+i+2*dyd.ibr_epri_n]

                # # Ia, Ib, Ic
                # ibri.cExternalInputs[3] = -kAbase*np.sqrt(2.0)*self.brch_Ipre[coe_idx] * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]
                # ibri.cExternalInputs[4] = -kAbase*np.sqrt(2.0)*self.brch_Ipre[coe_idx+1] * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]
                # ibri.cExternalInputs[5] = -kAbase*np.sqrt(2.0)*self.brch_Ipre[coe_idx+2] * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]

                # # IaL1, IbL1, IcL1
                # ibri.cExternalInputs[6] = kAbase*np.sqrt(2.0)*(-self.brch_Ipre[coe_idx]+self.brch_Ipre[coe_idx+3]) * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]
                # ibri.cExternalInputs[7] = kAbase*np.sqrt(2.0)*(-self.brch_Ipre[coe_idx+1]+self.brch_Ipre[coe_idx+4]) * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]
                # ibri.cExternalInputs[8] = kAbase*np.sqrt(2.0)*(-self.brch_Ipre[coe_idx+2]+self.brch_Ipre[coe_idx+5]) * pfd.basemva / pfd.ibr_MVA_base[ibrbus_idx]

                IaL1_1 = -self.brch_Ipre[coe_idx] + self.brch_Ipre[coe_idx + 3]
                IbL1_1 = -self.brch_Ipre[coe_idx + 1] + self.brch_Ipre[coe_idx + 4]
                IcL1_1 = -self.brch_Ipre[coe_idx + 2] + self.brch_Ipre[coe_idx + 5]
                Va_1 = self.Vsol[N3 + i]
                Vb_1 = self.Vsol[N3 + i + dyd.ibr_epri_n]
                Vc_1 = self.Vsol[N3 + i + 2 * dyd.ibr_epri_n]
                Ea_1 = ibri.cExternalOutputs[0] / (kVbase * np.sqrt(2.0 / 3.0))
                Eb_1 = ibri.cExternalOutputs[1] / (kVbase * np.sqrt(2.0 / 3.0))
                Ec_1 = ibri.cExternalOutputs[2] / (kVbase * np.sqrt(2.0 / 3.0))

                # # print(ibri.cExternalInputs[0:12])
                # # print(ibri.cExternalOutputs[0:12])
                # # Call main function from the DLL to update outputs based on updated inputs
                # ierr = Model_Outputs(ibri) # Return:    Integer status 0 (normal), 1 if messages are written, 2 for errors.  See IEEE_Cigre_DLLInterface_types.h
                # # print(ibri.cExternalInputs[0:12])
                # # print(ibri.cExternalOutputs[0:12])
                # # print('\n')

                # Ea = ibri.cExternalOutputs[0]/(kVbase*np.sqrt(2.0/3.0))
                # Eb = ibri.cExternalOutputs[1]/(kVbase*np.sqrt(2.0/3.0))
                # Ec = ibri.cExternalOutputs[2]/(kVbase*np.sqrt(2.0/3.0))

                # # update Iibr
                # Iibr_a = Ea/ini.Init_ibr_epri_Req[i] + (Ea_1-Va_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IaL1_1
                # Iibr_b = Eb/ini.Init_ibr_epri_Req[i] + (Eb_1-Vb_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IbL1_1
                # Iibr_c = Ec/ini.Init_ibr_epri_Req[i] + (Ec_1-Vc_1)*ini.Init_ibr_epri_Gv1[i] + ini.Init_ibr_epri_icf[i]*IcL1_1

                # update Iibr
                Iibr_a = IaL1_1 + Va_1 * ini.Init_ibr_epri_Gv1[i]
                Iibr_b = IbL1_1 + Vb_1 * ini.Init_ibr_epri_Gv1[i]
                Iibr_c = IcL1_1 + Vc_1 * ini.Init_ibr_epri_Gv1[i]

                # ===================================
                # considering interfacing resistance
                # if flag_itfc_R:
                #     Rii = np.dot(Ritfc,np.asarray([Iibr_a,Iibr_b,Iibr_c]))
                #     v_int = Rii + np.asarray([Va_1,Vb_1,Vc_1])
                #     iinj = np.dot(Gitfc ,  v_int )

                #     Iibr_a_itfc = iinj[0] - Iibr_a
                #     Iibr_b_itfc = iinj[1] - Iibr_b
                #     Iibr_c_itfc = iinj[2] - Iibr_c

                #     iinj = np.dot(Gitfc ,  np.asarray([Va_1,Vb_1,Vc_1]) )
                #     Iibr_a_itfc = iinj[0]
                #     Iibr_b_itfc = iinj[1]
                #     Iibr_c_itfc = iinj[2]

                # else:
                #     Iibr_a_itfc = 0
                #     Iibr_b_itfc = 0
                #     Iibr_c_itfc = 0
                Iibr_a_itfc = 0
                Iibr_b_itfc = 0
                Iibr_c_itfc = 0
                # ===================================

                Igi_epri[3 * N1 + i] = self.Igi_epri[3 * N1 + i] + Iibr_a + Iibr_a_itfc
                Igi_epri[3 * N1 + dyd.ibr_epri_n + i] = (
                    self.Igi_epri[3 * N1 + dyd.ibr_epri_n + i] + Iibr_b + Iibr_b_itfc
                )
                Igi_epri[3 * N1 + 2 * dyd.ibr_epri_n + i] = (
                    self.Igi_epri[3 * N1 + 2 * dyd.ibr_epri_n + i]
                    + Iibr_c
                    + Iibr_c_itfc
                )

        # Vsol = self.solveV(ini.admittance_mode,
        #                    Igs,
        #                    Igi + Igi_epri,
        #                    node_Ihis,
        #                    None)
        Vsol = self.Vsol

        # UpdateIhis

        # numba_updateIhis(
        #     self.brch_range[:,self.rank],
        #     # Altered arguments
        #     self.brch_Ihis,
        #     self.brch_Ipre,
        #     self.node_Ihis,
        #     # Constant arguments
        #     self.Vsol,
        #     ini.Init_net_coe0,
        #     ini.Init_net_N
        # )

        node_Ihis_out = np.zeros(ini.Init_net_N)
        # brch_Ihis = np.zeros(len(Init_net_coe0))

        for i in range(brch_range[0], brch_range[1]):
            Fidx = int(Init_net_coe0[i, 0].real)
            Tidx = int(Init_net_coe0[i, 1].real)

            #### IF CLAUSE ####
            if Tidx == -1:
                if Init_net_coe0[i, 2] == 0:
                    continue
                # brch_Ipre[i] = Vsol[Fidx]/Init_net_coe0[i,2].real + brch_Ihis[i]
                brch_Ihis_temp = (
                    Init_net_coe0[i, 3] * brch_Ipre[i]
                    + Init_net_coe0[i, 4] * Vsol[Fidx]
                )

            #### ELSE CLAUSE ####
            else:
                # brch_Ipre[i] = (Vsol[Fidx] - Vsol[Tidx])/Init_net_coe0[i,2].real + brch_Ihis[i]
                brch_Ihis_temp = Init_net_coe0[i, 3] * brch_Ipre[i] + Init_net_coe0[
                    i, 4
                ] * (Vsol[Fidx] - Vsol[Tidx])
                node_Ihis_out[Tidx] += brch_Ihis_temp.real

            # brch_Ihis[i] = brch_Ihis_temp.real
            node_Ihis_out[Fidx] -= brch_Ihis_temp.real

        #### END FOR ####

        self.brch_Ihis = brch_Ihis
        self.node_Ihis = node_Ihis_out
        self.flag_reinit = 0

        return

    def dump_res(
        self,
        SimMod,
        snapshot_mode,
        output_snp_ful,
        output_snp_1pt,
        output_res,
    ):
        dyd = self.dyd
        ini = self.ini
        pfd = self.pfd

        for i in range(self.nibrepri):
            ibri = self.ibr_epri[i]
            for j in range(20):
                ini.snp_ibrepri_par[i][j] = ibri.cParameters[j]
            for j in range(12):
                ini.snp_ibrepri_inp[i][j] = ibri.cExternalInputs[j]
            for j in range(12):
                ini.snp_ibrepri_out[i][j] = ibri.cExternalOutputs[j]
            for j in range(11):
                ini.snp_ibrepri_sta[i][j] = ibri.cDoubleStates[j]

        # remove SuperLU objects to be compatible with pickle
        ini.Init_net_G0_lu = []
        ini.Init_net_G1_lu = []
        ini.Init_net_G2_lu = []
        # TODO: Do we need to do something to release the memory here?
        self.ibr_epri = []
        self.Glu = []

        # output and plot
        x = []
        for k, v in self.x.items():
            x.append(v.tolist())
        self.x = np.transpose(x)

        if dyd.ibr_wecc_n > 0:
            x_ibr = []
            for k, v in self.x_ibr.items():
                x_ibr.append(v.tolist())
            self.x_ibr = np.transpose(x_ibr)

        if dyd.ibr_epri_n > 0:
            x_ibr_epri = []
            for k, v in self.x_ibr_epri.items():
                x_ibr_epri.append(v.tolist())
            self.x_ibr_epri = np.transpose(x_ibr_epri)

        vv = []
        for k, v in self.v.items():
            vv.append(v.tolist())
        self.v = np.transpose(vv)

        vv = []
        for k, v in self.x_bus.items():
            vv.append(v.tolist())
        self.x_bus = np.transpose(vv)

        vv = []
        for k, v in self.x_load.items():
            vv.append(v.tolist())
        self.x_load = np.transpose(vv)

        self.t = np.asarray(self.t)
        if snapshot_mode == 0:
            save_1p = 1
            save_ful = 0

        if snapshot_mode == 1:
            save_1p = 0
            save_ful = 1

        if snapshot_mode == 2:
            save_1p = 1
            save_ful = 1

        if save_1p == 1:
            x = np.squeeze(np.delete(self.x, range(0, len(self.x[0]) - 1, 1), 1))
            self.x = {}
            self.x[0] = x

            x = np.squeeze(
                np.delete(self.x_bus, range(0, len(self.x_bus[0]) - 1, 1), 1)
            )
            self.x_bus = {}
            self.x_bus[0] = x

            if dyd.ibr_wecc_n > 0:
                x = np.squeeze(
                    np.delete(self.x_ibr, range(0, len(self.x_ibr[0]) - 1, 1), 1)
                )
                self.x_ibr = {}
                self.x_ibr[0] = x

            if dyd.ibr_epri_n > 0:
                x = np.squeeze(
                    np.delete(
                        self.x_ibr_epri, range(0, len(self.x_ibr_epri[0]) - 1, 1), 1
                    )
                )
                self.x_ibr_epri = {}
                self.x_ibr_epri[0] = x

            if len(pfd.load_bus) > 0:
                x = np.squeeze(
                    np.delete(self.x_load, range(0, len(self.x_load[0]) - 1, 1), 1)
                )
                self.x_load = {}
                self.x_load[0] = x

            x = np.squeeze(np.delete(self.v, range(0, len(self.v[0]) - 1, 1), 1))
            self.v = {}
            self.v[0] = x

            pickle.dump([pfd, dyd, ini, self], open(output_snp_1pt, "wb"))

        if save_ful == 1:
            if SimMod == 0:
                pickle.dump([pfd, dyd, ini, self], open(output_snp_ful, "wb"))
            if SimMod == 1:
                pickle.dump([pfd, dyd, ini, self], open(output_res, "wb"))
        return
