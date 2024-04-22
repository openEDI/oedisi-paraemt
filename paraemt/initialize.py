
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from functools import reduce

from paraemt.lib_numba import numba_InitNet
from paraemt.lib_numba import numba_set_coo

# --------------------------------------------------------------------------------------------------
# EMT initializaiton

class Initialize():
    def __init__(self, pfd, dyd):
        nbus = len(pfd.bus_num)
        ngen = len(pfd.gen_bus)
        nload = len(pfd.load_bus)


        self.Init_x = np.asarray([])
        self.Init_x_ibr = np.asarray([])
        self.Init_x_ibr_epri = np.asarray([])
        self.Init_x_bus = np.asarray([])
        self.Init_x_load = np.asarray([])

        # network
        self.Init_net_VbaseA = np.asarray([])
        self.Init_net_ZbaseA = np.asarray([])
        self.Init_net_IbaseA = np.asarray([])
        self.Init_net_YbaseA = np.asarray([])

        self.Init_net_StA = np.asarray([])  # phasor value
        self.Init_net_Vt = np.asarray([])
        self.Init_net_It = np.asarray([])
        self.Init_net_Vtha = np.asarray([])

        self.Init_net_N = np.asarray([])
        self.Init_net_N1 = np.asarray([])
        # self.Init_gen_N = 0
        # self.Init_ibr_N = 0

        self.Init_net_G0 = np.asarray([])
        self.Init_net_coe0 = np.asarray([])
        self.Init_net_G0_inv = np.asarray([])
        self.Init_net_G0_data = []
        self.Init_net_G0_rows = []
        self.Init_net_G0_cols = []

        # self.Init_net_Gt0 = np.asarray([])

        # fault-on and post-fault G matrces
        self.Init_net_G1 = np.asarray([])
        self.Init_net_coe1 = np.asarray([])
        self.Init_net_G1_inv = np.asarray([])
        self.Init_net_G1_data = []
        self.Init_net_G1_rows = []
        self.Init_net_G1_cols = []

        self.Init_net_G2 = np.asarray([])
        self.Init_net_coe2 = np.asarray([])
        self.Init_net_G2_inv = np.asarray([])
        self.Init_net_G2_data = []
        self.Init_net_G2_rows = []
        self.Init_net_G2_cols = []


        self.Init_net_V = np.asarray([])  # instantaneous value
        self.Init_brch_Ipre = np.asarray([])
        self.Init_brch_Ihis = np.asarray([])
        self.Init_node_Ihis = np.asarray([])


        # machine
        self.Init_mac_phy = np.zeros(ngen)
        self.Init_mac_IgA = np.zeros(ngen, dtype=complex)
        self.Init_mac_dt = np.zeros(ngen)
        self.Init_mac_ed = np.zeros(ngen)
        self.Init_mac_eq = np.zeros(ngen)
        self.Init_mac_id = np.zeros(ngen)
        self.Init_mac_iq = np.zeros(ngen)

        self.Init_mac_i1d = np.zeros(ngen)
        self.Init_mac_i1q = np.zeros(ngen)
        self.Init_mac_i2q = np.zeros(ngen)
        self.Init_mac_psyd = np.zeros(ngen)
        self.Init_mac_psyq = np.zeros(ngen)
        self.Init_mac_ifd = np.zeros(ngen)
        self.Init_mac_psyfd = np.zeros(ngen)
        self.Init_mac_psy1d = np.zeros(ngen)
        self.Init_mac_psy1q = np.zeros(ngen)
        self.Init_mac_psy2q = np.zeros(ngen)
        self.Init_mac_te = np.zeros(ngen)
        self.Init_mac_qe = np.zeros(ngen)

        # machine excitation system
        self.Init_mac_v1 = np.zeros(dyd.exc_sexs_n)
        self.Init_mac_vref = np.zeros(dyd.exc_sexs_n)
        self.Init_mac_EFD = np.zeros(ngen)

        # machine governor system
        self.Init_mac_pref = np.zeros(ngen)
        self.Init_mac_pm = np.zeros(ngen)
        self.Init_mac_gref = np.zeros(ngen)

        self.Init_tgov1_p1 = np.zeros(dyd.gov_tgov1_n)
        self.Init_tgov1_p2 = np.zeros(dyd.gov_tgov1_n)
        self.Init_tgov1_gref = np.zeros(dyd.gov_tgov1_n)
        self.tgov1_2gen = np.zeros(dyd.gov_tgov1_n, dtype=int)

        self.Init_hygov_xe = np.zeros(dyd.gov_hygov_n)
        self.Init_hygov_xc = np.zeros(dyd.gov_hygov_n)
        self.Init_hygov_xg = np.zeros(dyd.gov_hygov_n)
        self.Init_hygov_xq = np.zeros(dyd.gov_hygov_n)
        self.Init_hygov_gref = np.zeros(dyd.gov_hygov_n)
        self.hygov_2gen = np.zeros(dyd.gov_hygov_n, dtype=int)

        self.Init_gast_p1 = np.zeros(dyd.gov_gast_n)
        self.Init_gast_p2 = np.zeros(dyd.gov_gast_n)
        self.Init_gast_p3 = np.zeros(dyd.gov_gast_n)
        self.Init_gast_gref = np.zeros(dyd.gov_gast_n)
        self.gast_2gen = np.zeros(dyd.gov_gast_n, dtype=int)

        # pss
        self.Init_ieeest_y1 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y2 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y3 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y4 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y5 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y6 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_y7 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_x1 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_x2 = np.zeros(dyd.pss_ieeest_n)
        self.Init_ieeest_vs = np.zeros(dyd.pss_n)

        # machine conductance
        self.Init_mac_Ld = np.asarray([])
        self.Init_mac_Lq = np.asarray([])
        self.Init_mac_Requiv = np.asarray([])
        self.Init_mac_Gequiv = np.asarray([])
        self.Init_mac_Rd = np.asarray([])
        self.Init_mac_Rq = np.asarray([])
        self.Init_mac_Rd1 = np.asarray([])
        self.Init_mac_Rd2 = np.asarray([])
        self.Init_mac_Rq1 = np.asarray([])
        self.Init_mac_Rq2 = np.asarray([])
        self.Init_mac_Rav = np.asarray([])
        self.Init_mac_alpha = np.asarray([])

        self.Init_mac_Rd1inv = np.asarray([])
        self.Init_mac_Rq1inv = np.asarray([])
        self.Init_mac_Rd_coe = np.asarray([])
        self.Init_mac_Rq_coe = np.asarray([])

        self.Init_mac_H = np.zeros(ngen)

        # IBR - REGCA
        self.Init_ibr_regca_s0 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_s1 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_s2 = np.zeros(dyd.ibr_wecc_n)

        self.Init_ibr_regca_Vmp = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_Vap = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_i1 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_Qgen0 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_i2 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_regca_ip2rr = np.zeros(dyd.ibr_wecc_n)

        # IBR - REECB
        self.Init_ibr_reecb_s0 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_s1 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_s2 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_s3 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_s4 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_s5 = np.zeros(dyd.ibr_wecc_n)

        self.Init_ibr_reecb_Vref0 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_pfaref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_Ipcmd = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_Iqcmd = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_Pref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_Qext = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_q2vPI = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_reecb_v2iPI = np.zeros(dyd.ibr_wecc_n)

        # IBR - REPCA
        self.Init_ibr_repca_s0 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s1 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s2 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s3 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s4 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s5 = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_s6 = np.zeros(dyd.ibr_wecc_n)

        self.Init_ibr_repca_Vref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_Qref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_Freq_ref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_Plant_pref = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_LineMW = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_LineMvar = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_LineMVA = np.zeros(dyd.ibr_wecc_n, dtype=complex)
        self.Init_ibr_repca_QVdbout = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_fdbout = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_Pref_out = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_vq2qPI = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_repca_p2pPI = np.zeros(dyd.ibr_wecc_n)

        # IBR - PLL
        self.Init_ibr_pll_ze = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_pll_de = np.zeros(dyd.ibr_wecc_n)
        self.Init_ibr_pll_we = np.zeros(dyd.ibr_wecc_n)

        # IBR - EPRI
        self.Init_ibr_epri_Pref = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Qref = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Vref = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Idref = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Iqref = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Vd = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Vq = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_fPLL = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_thetaPLL = np.zeros(dyd.ibr_epri_n)

        self.Init_ibr_epri_Id_L1 = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Iq_L1 = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Id = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Iq = np.zeros(dyd.ibr_epri_n)
        
        self.Init_ibr_epri_Va = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Vb = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Vc = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Ia = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Ib = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Ic = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_IaL1 = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_IbL1 = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_IcL1 = np.zeros(dyd.ibr_epri_n)

        self.Init_ibr_epri_Ea = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Eb = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Ec = np.zeros(dyd.ibr_epri_n)

        self.Init_ibr_epri_Vt = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)
        self.Init_ibr_epri_Vct = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)
        self.Init_ibr_epri_Vin = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)
        self.Init_ibr_epri_It = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)
        self.Init_ibr_epri_IL1 = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)
        self.Init_ibr_epri_Ict = np.zeros(dyd.ibr_epri_n, dtype=np.complex128)

        self.Init_ibr_epri_Req = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_icf = np.zeros(dyd.ibr_epri_n)
        self.Init_ibr_epri_Gv1 = np.zeros(dyd.ibr_epri_n)

        # PLL for bus freq/ang
        self.Init_pll_ze = np.zeros(nbus)
        self.Init_pll_de = np.zeros(nbus)
        self.Init_pll_we = np.zeros(nbus)

        # volt mag measurement
        self.Init_vt = np.zeros(nbus)  # calculated volt mag
        self.Init_vtm = np.zeros(nbus)  # measured volt mag (after calc)
        self.Init_dvtm = np.zeros(nbus)  # measured dvm/dt

        # load
        self.Init_ZL_ang = np.zeros(nload)
        self.Init_ZL_mag = np.zeros(nload)
        self.Init_PL = np.zeros(nload)
        self.Init_QL = np.zeros(nload)

        # save IBR condition into a snapshot
        self.snp_ibrepri_par = np.zeros((dyd.ibr_epri_n,20))
        self.snp_ibrepri_inp = np.zeros((dyd.ibr_epri_n,12))
        self.snp_ibrepri_out = np.zeros((dyd.ibr_epri_n,12))
        self.snp_ibrepri_sta = np.zeros((dyd.ibr_epri_n,11))

        return



    def InitNet(self, pfd, dyd, ts, loadmodel_option):
        (self.Init_net_VbaseA,
         self.Init_net_ZbaseA,
         self.Init_net_IbaseA,
         self.Init_net_YbaseA,
         self.Init_net_StA,
         self.Init_net_Vt,
         self.Init_net_It,
         self.Init_net_N,
         self.Init_net_coe0,
         self.Init_net_Vtha,
         self.Init_net_G0_rows,
         self.Init_net_G0_cols,
         self.Init_net_G0_data,
        ) = numba_InitNet(
            pfd.basemva,
            pfd.ws,
            pfd.bus_num,
            pfd.bus_basekV,
            pfd.bus_Vm,
            pfd.bus_Va,
            pfd.gen_bus,
            pfd.gen_MW,
            pfd.gen_Mvar,
            pfd.line_from,
            pfd.line_to,
            pfd.line_RX,
            pfd.line_chg,
            pfd.xfmr_from,
            pfd.xfmr_to,
            pfd.xfmr_RX,
            pfd.load_bus,
            pfd.load_MW,
            pfd.load_Mvar,
            pfd.shnt_bus,
            pfd.shnt_gb,
            pfd.shnt_sw_bus,
            pfd.shnt_sw_gb,
            dyd.ibr_epri_n,
            ts,
            loadmodel_option,
        )

        return


    def InitMac(self, pfd, dyd):
        for i in range(len(pfd.gen_bus)):
            genbus = pfd.gen_bus[i]
            genbus_idx = np.where(pfd.bus_num == genbus)[0][0]

            S_temp = math.sqrt(pfd.gen_MW[i] * pfd.gen_MW[i] + pfd.gen_Mvar[i] * pfd.gen_Mvar[i])
            if np.abs(S_temp)>1e-10:
                phy_temp = math.asin(pfd.gen_Mvar[i] / S_temp)
            else:
                phy_temp = 0.0
            IgA_temp = self.Init_net_It[i] * self.Init_net_IbaseA[genbus_idx] / (dyd.base_Is[i] / 1000.0)
            dt_temp = np.sign(pfd.gen_MW[i])*math.atan((dyd.ec_Lq[i] * abs(IgA_temp) * math.cos(
                phy_temp) - dyd.ec_Ra[i] * abs(IgA_temp) * math.sin(phy_temp)) / (
                                        abs(self.Init_net_Vt[genbus_idx]) + dyd.ec_Ra[i] * abs(
                                    IgA_temp) * math.cos(phy_temp) + dyd.ec_Lq[i] * abs(
                                    IgA_temp) * math.sin(phy_temp)))
            dt0_temp = dt_temp + pfd.bus_Va[genbus_idx]

            ed_temp = abs(self.Init_net_Vt[genbus_idx]) * math.sin(dt_temp)
            eq_temp = abs(self.Init_net_Vt[genbus_idx]) * math.cos(dt_temp)

            if pfd.gen_MW[i] < 0:
                id_temp = -abs(IgA_temp) * math.sin(dt_temp - phy_temp)
                iq_temp = -abs(IgA_temp) * math.cos(dt_temp - phy_temp)
            else:
                id_temp = abs(IgA_temp) * math.sin(dt_temp + phy_temp)
                iq_temp = abs(IgA_temp) * math.cos(dt_temp + phy_temp)
            ## End if

            i1d_temp = 0.0
            i1q_temp = 0.0
            i2q_temp = 0.0
            psyd_temp = eq_temp + dyd.ec_Ra[i] * iq_temp
            psyq_temp = - (ed_temp + dyd.ec_Ra[i] * id_temp)
            ifd_temp = (eq_temp + dyd.ec_Ld[i] * id_temp + dyd.ec_Ra[i] * iq_temp) / dyd.ec_Lad[i]
            efd_temp = dyd.ec_Rfd[i] * ifd_temp
            psyfd_temp = dyd.ec_Lffd[i] * ifd_temp - dyd.ec_Lad[i] * id_temp
            psy1d_temp = dyd.ec_Lad[i] * (ifd_temp - id_temp)
            psy1q_temp = - dyd.ec_Laq[i] * iq_temp
            psy2q_temp = - dyd.ec_Laq[i] * iq_temp


            ## used for a while -------------------------------------
            # pref_temp = ed_temp * id_temp + eq_temp * iq_temp
            # qe_temp = eq_temp * id_temp - ed_temp * iq_temp
            ## ---------------------------------------------------


            pref_temp = psyd_temp * iq_temp - psyq_temp * id_temp
            qe_temp = psyd_temp * id_temp + psyq_temp * iq_temp



            self.Init_mac_phy[i] = phy_temp
            self.Init_mac_IgA[i] = IgA_temp
            self.Init_mac_dt[i] = dt0_temp
            self.Init_mac_ed[i] = ed_temp
            self.Init_mac_eq[i] = eq_temp
            self.Init_mac_id[i] = id_temp
            self.Init_mac_iq[i] = iq_temp
            self.Init_mac_i1d[i] = i1d_temp
            self.Init_mac_i1q[i] = i1q_temp
            self.Init_mac_i2q[i] = i2q_temp
            self.Init_mac_psyd[i] = psyd_temp
            self.Init_mac_psyq[i] = psyq_temp
            self.Init_mac_ifd[i] = ifd_temp
            self.Init_mac_psyfd[i] = psyfd_temp
            self.Init_mac_psy1d[i] = psy1d_temp
            self.Init_mac_psy1q[i] = psy1q_temp
            self.Init_mac_psy2q[i] = psy2q_temp
            # self.Init_mac_te[i] = pref_temp
            self.Init_mac_te[i] = ed_temp * id_temp + eq_temp * iq_temp
            self.Init_mac_qe[i] = qe_temp

            # Efd initialized for excitation system
            self.Init_mac_EFD[i] = efd_temp*dyd.ec_Lad[i]/dyd.ec_Rfd[i]

            # Pref initialized for governor system
            self.Init_mac_pref[i] = pref_temp


    def InitExc(self, pfd, dyd):
        for i in range(dyd.exc_sexs_n):
            genbus = pfd.gen_bus[i]
            genbus_idx = int(np.where(pfd.bus_num == genbus)[0])

            v1 = self.Init_mac_EFD[i] / dyd.exc_sexs_K[i]
            vref = v1 + pfd.bus_Vm[genbus_idx]

            self.Init_mac_v1[i] = v1
            self.Init_mac_vref[i] = vref




    def InitGov(self, pfd, dyd):
        # TGOV1
        for govi in range(dyd.gov_tgov1_n):
            genbus = dyd.gov_tgov1_bus[govi]
            idx = np.where(pfd.gen_bus == genbus)[0]
            if len(idx) > 1:
                tempid = dyd.gov_tgov1_id[govi]
                # if len(tempid) == 1:
                #     tempid = tempid + ' '  # PSSE gen ID always has two digits
                idx1 = np.where(pfd.gen_id[idx] == tempid)[0][0]
                idx = idx[idx1]
            self.tgov1_2gen[govi] = int(idx)

            self.Init_mac_pm[idx] = self.Init_mac_pref[idx]
            self.Init_mac_gref[idx] = self.Init_mac_pref[idx] * dyd.gov_tgov1_R[govi]
            self.Init_tgov1_p2[govi] = self.Init_mac_pref[idx]
            self.Init_tgov1_p1[govi] = self.Init_mac_pref[idx]
            self.Init_tgov1_gref[govi] = self.Init_mac_pref[idx] * dyd.gov_tgov1_R[govi]

        # HYGOV
        for govi in range(dyd.gov_hygov_n):
            genbus = dyd.gov_hygov_bus[govi]
            idx = np.where(pfd.gen_bus == genbus)[0]
            if len(idx) > 1:
                tempid = dyd.gov_hygov_id[govi]
                # if len(tempid) == 1:
                #     tempid = tempid + ' '  # PSSE gen ID always has two digits
                idx1 = np.where(pfd.gen_id[idx] == tempid)[0][0]
                idx = idx[idx1]
            self.hygov_2gen[govi] = int(idx)

            Tm0 = self.Init_mac_pref[idx]
            q0 = Tm0 / dyd.gov_hygov_At[govi] + dyd.gov_hygov_qNL[govi]
            c0 = q0
            g0 = c0
            nref = g0 * dyd.gov_hygov_R[govi]

            self.Init_hygov_xe[govi] = 0.0
            self.Init_hygov_xc[govi] = c0
            self.Init_hygov_xg[govi] = g0
            self.Init_hygov_xq[govi] = q0
            self.Init_hygov_gref[govi] = nref
            self.Init_mac_pm[idx] = Tm0
            self.Init_mac_gref[idx] = nref

        # GAST
        for govi in range(dyd.gov_gast_n):

            genbus = dyd.gov_gast_bus[govi]
            idx = np.where(pfd.gen_bus == genbus)[0]
            if len(idx) > 1:
                tempid = dyd.gov_gast_id[govi]
                # if len(tempid) == 1:
                #     tempid = tempid + ' '  # PSSE gen ID always has two digits
                idx1 = np.where(pfd.gen_id[idx] == tempid)[0][0]
                idx = idx[idx1]
            self.gast_2gen[govi] = int(idx)

            pref = self.Init_mac_pref[idx]

            self.Init_gast_p1[govi] = pref
            self.Init_gast_p2[govi] = pref
            self.Init_gast_p3[govi] = pref
            self.Init_gast_gref[govi] = pref
            self.Init_mac_pm[idx] = pref
            self.Init_mac_gref[idx] = pref

    def InitPss(self, pfd, dyd):
        for i in range(dyd.pss_ieeest_n):
            if dyd.pss_type[i] == 'IEEEST':
                self.Init_ieeest_y1[i] = 0.0
                self.Init_ieeest_y2[i] = 0.0
                self.Init_ieeest_y3[i] = 0.0
                self.Init_ieeest_y4[i] = 0.0
                self.Init_ieeest_y5[i] = 0.0
                self.Init_ieeest_y6[i] = 0.0
                self.Init_ieeest_y7[i] = 0.0
                self.Init_ieeest_x1[i] = 0.0
                self.Init_ieeest_x2[i] = 0.0
                self.Init_ieeest_vs[i] = 0.0

    def InitIbrepri(self, pfd, dyd):

        for i in range(dyd.ibr_epri_n):
            ibrbus = dyd.ibr_epri_bus[i]
            ibrid = dyd.ibr_epri_id[i]
            # if len(ibrid)==1:
            #     ibrid = ibrid + ' '

            ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0]
            ibrid_idx = np.where(pfd.ibr_id[ibrbus_idx] == ibrid)[0][0]
            ibrbus_idx = ibrbus_idx[ibrid_idx]


            bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]

            Vm = pfd.bus_Vm[bus_idx]
            Va = pfd.bus_Va[bus_idx]

            Vt = complex(Vm * np.cos(Va), Vm * np.sin(Va))
            St = complex(pfd.ibr_MW[ibrbus_idx], pfd.ibr_Mvar[ibrbus_idx]) / pfd.ibr_MVA_base[ibrbus_idx]
            It = np.conjugate(St/Vt)

            Vc = Vt + It*complex(dyd.ibr_epri_Rchoke[i], dyd.ibr_epri_Lchoke[i])
            Ic = Vc * complex(0, dyd.ibr_epri_Cfilt[i]) + Vc / dyd.ibr_epri_Rdamp[i]
            IL1 = It + Ic
            Vin = Vc + IL1*complex(dyd.ibr_epri_Rchoke[i], dyd.ibr_epri_Lchoke[i])

            thetaPLL = np.angle(Vc)

            Vd = np.real(Vc) * math.cos(thetaPLL) + np.imag(Vc) * math.sin(thetaPLL)
            Vq = np.imag(Vc) * math.cos(thetaPLL) - np.real(Vc) * math.sin(thetaPLL)  

            Vtd = np.real(Vt) * math.cos(thetaPLL) + np.imag(Vt) * math.sin(thetaPLL)
            Vtq = np.imag(Vt) * math.cos(thetaPLL) - np.real(Vt) * math.sin(thetaPLL)

            Id = np.real(It) * math.cos(thetaPLL) + np.imag(It) * math.sin(thetaPLL)  # d - MW
            Iq = np.imag(It) * math.cos(thetaPLL) - np.real(It) * math.sin(thetaPLL)  # q - Mvar

            Pelec = Vd * Id + Vq * Iq   # P3
            Qelec = -Vd * Iq + Vq * Id   # Q3

            # P4 = Vtd * Id + Vtq * Iq
            # Q4 = -Vtd * Iq + Vtq * Id;   

            Id_L1 = np.real(IL1)*math.cos(thetaPLL) + np.imag(IL1)*math.sin(thetaPLL) # d - MW
            Iq_L1 = np.imag(IL1)*math.cos(thetaPLL) - np.real(IL1)*math.sin(thetaPLL) # q - Mvar


            # P2 = Vd * Id_L1 + Vq * Iq_L1
            # Q2 = -Vd * Iq_L1 + Vq * Id_L1  

            # Vind = np.real(Vin) * math.cos(thetaPLL) + np.imag(Vin) * math.sin(thetaPLL)
            # Vinq = np.imag(Vin) * math.cos(thetaPLL) - np.real(Vin) * math.sin(thetaPLL)

            # P1 = Vind * Id_L1 + Vinq * Iq_L1
            # Q1 = -Vind * Iq_L1 + Vinq * Id_L1     

            # self.Init_ibr_epri_Pref[i] = pfd.ibr_MW[ibrbus_idx]
            # self.Init_ibr_epri_Qref[i] = pfd.ibr_Mvar[ibrbus_idx]
            self.Init_ibr_epri_Pref[i] = Pelec * pfd.ibr_MVA_base[ibrbus_idx]
            self.Init_ibr_epri_Qref[i] = Qelec * pfd.ibr_MVA_base[ibrbus_idx]
            self.Init_ibr_epri_Vref[i] = np.abs(Vc)

            self.Init_ibr_epri_fPLL[i] = 60.0
            self.Init_ibr_epri_thetaPLL[i] = thetaPLL

            self.Init_ibr_epri_Idref[i] = Id_L1
            self.Init_ibr_epri_Iqref[i] = Iq_L1
            self.Init_ibr_epri_Id_L1[i] = Id_L1
            self.Init_ibr_epri_Iq_L1[i] = Iq_L1
            self.Init_ibr_epri_Vd[i] = Vd
            self.Init_ibr_epri_Vq[i] = Vq
            self.Init_ibr_epri_Id[i] = Id
            self.Init_ibr_epri_Iq[i] = Iq

            VcB = Vc * complex(-0.5, -0.5*np.sqrt(3.0))
            VcC = Vc * complex(-0.5, 0.5*np.sqrt(3.0))

            ItB = It * complex(-0.5, -0.5*np.sqrt(3.0))
            ItC = It * complex(-0.5, 0.5*np.sqrt(3.0))

            IbL1 = IL1 * complex(-0.5, -0.5*np.sqrt(3.0))
            IcL1 = IL1 * complex(-0.5, 0.5*np.sqrt(3.0))

            Eb = Vin * complex(-0.5, -0.5*np.sqrt(3.0))
            Ec = Vin * complex(-0.5, 0.5*np.sqrt(3.0))

            self.Init_ibr_epri_Va[i] = Vc.real
            self.Init_ibr_epri_Vb[i] = VcB.real
            self.Init_ibr_epri_Vc[i] = VcC.real
            self.Init_ibr_epri_Ia[i] = It.real
            self.Init_ibr_epri_Ib[i] = ItB.real
            self.Init_ibr_epri_Ic[i] = ItC.real # used twice!!!!!
            self.Init_ibr_epri_IaL1[i] = IL1.real
            self.Init_ibr_epri_IbL1[i] = IbL1.real
            self.Init_ibr_epri_IcL1[i] = IcL1.real

            Sbase = pfd.ibr_MVA_base[ibrbus_idx]
            kVbase = pfd.bus_basekV[bus_idx]
            kAbase = Sbase / (np.sqrt(3) * kVbase)

            self.Init_ibr_epri_Ea[i] = Vin.real * kVbase*np.sqrt(2.0/3.0)
            self.Init_ibr_epri_Eb[i] = Eb.real * kVbase*np.sqrt(2.0/3.0)
            self.Init_ibr_epri_Ec[i] = Ec.real * kVbase*np.sqrt(2.0/3.0)

            self.Init_ibr_epri_Vt[i] = Vt
            self.Init_ibr_epri_Vct[i] = Vc
            self.Init_ibr_epri_Vin[i] = Vin
            self.Init_ibr_epri_It[i] = It
            self.Init_ibr_epri_IL1[i] = IL1
            self.Init_ibr_epri_Ict[i] = Ic


            # bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]

            # # Set initial inputs, outputs, parameters, and states
            # (kVbase, IBR_MVA_base, fbase, Vdcbase) = (pfd.bus_basekV[bus_idx],
            #                                           pfd.ibr_MVA_base[ibrbus_idx],
            #                                           60.0,
            #                                           dyd.ibr_epri_Vdcbase[i])

            # kAbase = IBR_MVA_base/kVbase/np.sqrt(3.0)

            # (Imax, KiI, KiP, KiPLL, KiQ, KpI, KpP, KpPLL, KpQ, Pqflag, Vdip, Vup) = (
            #     dyd.ibr_epri_Imax[i],
            #     dyd.ibr_epri_KiI[i],
            #     dyd.ibr_epri_KiP[i],
            #     dyd.ibr_epri_KiPLL[i],
            #     dyd.ibr_epri_KiQ[i],
            #     dyd.ibr_epri_KpI[i],
            #     dyd.ibr_epri_KpP[i],
            #     dyd.ibr_epri_KpPLL[i],
            #     dyd.ibr_epri_KpQ[i],
            #     dyd.ibr_epri_Pqflag[i],
            #     dyd.ibr_epri_Vdip[i],
            #     dyd.ibr_epri_Vup[i])

            # (Cfilt, Lchoke, Rchoke, Rdamp) = (dyd.ibr_epri_Cfilt[i],
            #                                   dyd.ibr_epri_Lchoke[i],
            #                                   dyd.ibr_epri_Rchoke[i],
            #                                   dyd.ibr_epri_Rdamp[i])

            # # IBR time step
            # # ts_ibr = emt.ts
            # ts_ibr = 50e-6

            # Inputs = [kVbase*np.sqrt(2.0/3.0)*self.Init_ibr_epri_Va[i],
            #           kVbase*np.sqrt(2.0/3.0)*self.Init_ibr_epri_Vb[i],
            #           kVbase*np.sqrt(2.0/3.0)*self.Init_ibr_epri_Vc[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_Ia[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_Ib[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_Ic[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_IaL1[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_IbL1[i],
            #           kAbase*np.sqrt(2.0)*self.Init_ibr_epri_IcL1[i],
            #           self.Init_ibr_epri_Pref[i],
            #           self.Init_ibr_epri_Qref[i],
            #           self.Init_ibr_epri_Vd[i]]  # with PQ ref initialized at P3
            # Outputs = [self.Init_ibr_epri_Ea[i],
            #            self.Init_ibr_epri_Eb[i],
            #            self.Init_ibr_epri_Ec[i],
            #            self.Init_ibr_epri_Idref[i],
            #            self.Init_ibr_epri_Idref[i],
            #            self.Init_ibr_epri_Iqref[i],
            #            self.Init_ibr_epri_Iqref[i],
            #            self.Init_ibr_epri_Vd[i],
            #            self.Init_ibr_epri_Vq[i],
            #            60.0,
            #            self.Init_ibr_epri_Pref[i],
            #            self.Init_ibr_epri_Qref[i]] #Updated by DLL

            # Parameters = [kVbase,IBR_MVA_base,Vdcbase,KpI,KiI,KpPLL,KiPLL,KpP,
            #               KiP,KpQ,KiQ,Imax,Pqflag,Vdip,Vup,Rchoke,Lchoke,Cfilt,Rdamp,ts_ibr]
            # IntSt = []
            # FloatSt = []
            # DoubleSt = [0.0,0.0,self.Init_ibr_epri_thetaPLL[i],
            #             0.0,0.0,0.0,0.0,0.0,
            #             self.Init_ibr_epri_Id[i],0.0,self.Init_ibr_epri_Iq[i]] #Updated by DLL

            # # Define object of model. One object per instance of the model used in simulation
            # ibri = emt.ibr_epri[i]
            # ibri.cExternalInputs = (c_double*num_in_ports)(*Inputs)
            # ibri.cExternalOutputs = (c_double*num_out_ports)(*Outputs)
            # ibri.cParameters = (c_double*num_param)(*Parameters)
            # # Should be updated at each time step
            # ibri.cTime = 0.0
            # ibri.cIntStates = (c_int*num_int_states)(*IntSt)
            # ibri.cFloatStates = (c_float*num_float_states)(*FloatSt)
            # ibri.cDoubleStates = (c_double*num_double_states)(*DoubleSt)

            # # Call function to check parameter values. Not critical for initial test run of model use
            # # print ("Check Parameters")
            # Model_CheckParameters = wrap_function(add_lib,
            #                                       'Model_CheckParameters',
            #                                       c_int,
            #                                       [POINTER(MODELINSTANCE)])
            # return_int = Model_CheckParameters(ibri)

            # # Call function to initialize states of the model. Needs appropriate code in the DLL
            # # print ("Model Initialization")
            # Model_Initialize = wrap_function(add_lib,
            #                                  'Model_Initialize',
            #                                  c_int,
            #                                  [POINTER(MODELINSTANCE)])
            # return_int = Model_Initialize(ibri)

        ## End for

        return


    def CheckMacEq(self, pfd, dyd):
        for i in range(len(pfd.gen_bus)):
            EFD2efd = dyd.ec_Rfd[i] / dyd.ec_Lad[i]
            eq = [0] * 12
            eq[0] = self.Init_mac_ed[i] + self.Init_mac_psyq[i] + dyd.ec_Ra[i] * self.Init_mac_id[i]
            eq[1] = self.Init_mac_eq[i] - self.Init_mac_psyd[i] + dyd.ec_Ra[i] * self.Init_mac_iq[i]
            eq[2] = self.Init_mac_EFD[i] * EFD2efd - dyd.ec_Rfd[i] * self.Init_mac_ifd[i]
            eq[3] = - dyd.ec_R1d[i] * self.Init_mac_i1d[i]
            eq[4] = - dyd.ec_R1q[i] * self.Init_mac_i1q[i]
            eq[5] = - dyd.ec_R2q[i] * self.Init_mac_i2q[i]
            eq[6] = - (dyd.ec_Lad[i] + dyd.ec_Ll[i]) * self.Init_mac_id[i] + dyd.ec_Lad[i] * self.Init_mac_ifd[i] + \
                    dyd.ec_Lad[i] * self.Init_mac_i1d[i] - self.Init_mac_psyd[i]
            eq[7] = - (dyd.ec_Laq[i] + dyd.ec_Ll[i]) * self.Init_mac_iq[i] + dyd.ec_Laq[i] * self.Init_mac_i1q[i] + \
                    dyd.ec_Laq[i] * self.Init_mac_i2q[i] - self.Init_mac_psyq[i]
            eq[8] = dyd.ec_Lffd[i] * self.Init_mac_ifd[i] + dyd.ec_Lf1d[i] * self.Init_mac_i1d[i] - \
                    dyd.ec_Lad[i] * self.Init_mac_id[i] - self.Init_mac_psyfd[i]
            eq[9] = dyd.ec_Lf1d[i] * self.Init_mac_ifd[i] + dyd.ec_L11d[i] * self.Init_mac_i1d[i] - \
                    dyd.ec_Lad[i] * self.Init_mac_id[i] - self.Init_mac_psy1d[i]
            eq[10] = dyd.ec_L11q[i] * self.Init_mac_i1q[i] + dyd.ec_Laq[i] * self.Init_mac_i2q[i] - \
                     dyd.ec_Laq[i] * self.Init_mac_iq[i] - self.Init_mac_psy1q[i]
            eq[11] = dyd.ec_Laq[i] * self.Init_mac_i1q[i] + dyd.ec_L22q[i] * self.Init_mac_i2q[i] - \
                     dyd.ec_Laq[i] * self.Init_mac_iq[i] - self.Init_mac_psy2q[i]

            sos = reduce(lambda i, j: i + j * j, [eq[:1][0] ** 2] + eq[1:])
            if sos > 1e-10:
                print('Issue in machine init!!!')
                print(eq)
            else:
                pass


    def InitREGCA(self, pfd, dyd):
        for i in range(dyd.ibr_wecc_n):
            ibrbus = pfd.ibr_bus[i]
            ibrbus_idx = np.where(pfd.bus_num == ibrbus)[0]

            P = pfd.ibr_MW[i] / dyd.ibr_MVAbase[i]
            Q = pfd.ibr_Mvar[i] / dyd.ibr_MVAbase[i]
            S = np.complex128(P, Q)
            Vm = pfd.bus_Vm[ibrbus_idx]
            Va = pfd.bus_Va[ibrbus_idx]
            Vt = np.complex128(Vm*math.cos(Va), Vm*math.sin(Va))

            It = np.conj(S/Vt)

            Ip_out = np.real(It) * math.cos(Va) + np.imag(It) * math.sin(Va)
            Iq_out = np.imag(It) * math.cos(Va) - np.real(It) * math.sin(Va)

            i1 = max(0.0, (Vm - dyd.ibr_regca_Volim[i]) * dyd.ibr_regca_Khv[i])
            Iq = Iq_out + i1

            if Vm > dyd.ibr_regca_Lvpnt1[i]:
                i2 = 1.0
            elif Vm < dyd.ibr_regca_Lvpnt0[i]:
                i2 = 0.0
                print('ERROR: Volt mag at bus ' + str(pfd.bus_num[ibrbus_idx] + ' too low to initialize IBR!!'))
                sys.exit(0)
            else:
                i2 = (Vm - dyd.ibr_regca_Lvpnt0[i]) / (dyd.ibr_regca_Lvpnt1[i] - dyd.ibr_regca_Lvpnt0[i])
            Ip = Ip_out / i2


            s0 = Ip
            s1 = -Iq
            s2 = Vm

            self.Init_ibr_regca_s0[i] = s0
            self.Init_ibr_regca_s1[i] = s1
            self.Init_ibr_regca_s2[i] = s2

            self.Init_ibr_regca_Vmp[i] = Vm
            self.Init_ibr_regca_Vap[i] = Va
            self.Init_ibr_regca_i1[i] = i1
            self.Init_ibr_regca_Qgen0[i] = Q
            self.Init_ibr_regca_i2[i] = i2
            self.Init_ibr_regca_ip2rr[i] = 0.0

        ## End for

        return


    def InitREECB(self, pfd, dyd):
        for i in range(dyd.ibr_wecc_n):
            P = pfd.ibr_MW[i] / dyd.ibr_MVAbase[i]
            Q = pfd.ibr_Mvar[i] / dyd.ibr_MVAbase[i]

            Ipcmd = self.Init_ibr_regca_s0[i]
            Iqcmd = self.Init_ibr_regca_s1[i]
            Vm = self.Init_ibr_regca_s2[i]

            s0 = Vm
            s1 = P
            s4 = Q / Vm
            s5 = P

            if dyd.ibr_reecb_Vref0[i] == 0.0:
                Vref0 = Vm
            else:
                Vref0 = dyd.ibr_reecb_Vref0[i]


            self.Init_ibr_reecb_s0[i] = s0
            self.Init_ibr_reecb_s1[i] = s1
            self.Init_ibr_reecb_s2[i] = 0.0
            self.Init_ibr_reecb_s3[i] = 0.0
            self.Init_ibr_reecb_s4[i] = s4
            self.Init_ibr_reecb_s5[i] = s5

            self.Init_ibr_reecb_Vref0[i] = Vref0
            self.Init_ibr_reecb_pfaref[i] = math.atan(Q / P)
            self.Init_ibr_reecb_Ipcmd[i] = Ipcmd
            self.Init_ibr_reecb_Iqcmd[i] = Iqcmd
            self.Init_ibr_reecb_Pref[i] = Ipcmd * Vm
            self.Init_ibr_reecb_Qext[i] = Q
            self.Init_ibr_reecb_q2vPI[i] = 0.0
            self.Init_ibr_reecb_v2iPI[i] = 0.0

        ## End for

        return

    def InitREPCA(self, pfd, dyd):
        for i in range(dyd.ibr_wecc_n):
            ibrbus = pfd.ibr_bus[i]
            ibrbus_idx = np.where(pfd.bus_num == ibrbus)

            P = pfd.ibr_MW[i] / dyd.ibr_MVAbase[i]
            Q = pfd.ibr_Mvar[i] / dyd.ibr_MVAbase[i]
            S = np.complex128(P, Q)
            Vm = pfd.bus_Vm[ibrbus_idx]
            Va = pfd.bus_Va[ibrbus_idx]
            Vt = np.complex128(Vm * math.cos(Va), Vm * math.sin(Va))
            It = np.conj(S / Vt)

            if abs(dyd.ibr_repca_branch_From_bus[i]) + abs(dyd.ibr_repca_branch_To_bus[i]) == 0:
                Pbranch = P
                Qbranch = Q
                Sbranch = S
                Ibranch = It
            else:
                pass  # need to get the complex P, Q, S and I on the indicated branch

            if dyd.ibr_repca_remote_bus[i] == 0:
                Vreg = Vt
            else:
                remote_bus_idx = np.where(pfd.bus_num == dyd.ibr_repca_remote_bus[i])
                Vm_rem = pfd.bus_Vm[remote_bus_idx]
                Va_rem = pfd.bus_Va[remote_bus_idx]
                Vreg = np.complex128(Vm_rem * math.cos(Va_rem), Vm_rem * math.sin(Va_rem))

            V1_in1 = np.abs(Vreg + np.complex128(dyd.ibr_repca_Rc[i], dyd.ibr_repca_Xc[i]) * Ibranch)
            V1_in0 = Qbranch * dyd.ibr_repca_Kc[i] + Vm

            if dyd.ibr_repca_VCFlag[i] == 0:
                V1 = V1_in0
            else:
                V1 = V1_in1

            s0 = V1
            s1 = Qbranch

            if dyd.ibr_repca_FFlag[i] == 0:
                self.Init_ibr_repca_Pref_out = np.append(self.Init_ibr_repca_Pref_out, self.Init_ibr_reecb_Pref[i])
                s4 = 0.0
                s5 = 0.0
                s6 = 0.0
            else:
                s4 = Pbranch
                s5 = self.Init_ibr_reecb_Pref[i]
                s6 = self.Init_ibr_reecb_Pref[i]

            s2 = self.Init_ibr_reecb_Qext[i]
            s3 = self.Init_ibr_reecb_Qext[i]

            self.Init_ibr_repca_s0[i] = s0
            self.Init_ibr_repca_s1[i] = s1
            self.Init_ibr_repca_s2[i] = s2
            self.Init_ibr_repca_s3[i] = s3
            self.Init_ibr_repca_s4[i] = s4
            self.Init_ibr_repca_s5[i] = s5
            self.Init_ibr_repca_s6[i] = s6

            self.Init_ibr_repca_Vref[i] = s0
            self.Init_ibr_repca_Qref[i] = s1
            self.Init_ibr_repca_Freq_ref[i] = 1.0
            self.Init_ibr_repca_Plant_pref[i] = Pbranch
            self.Init_ibr_repca_LineMW[i] = Pbranch
            self.Init_ibr_repca_LineMvar[i] = Qbranch
            self.Init_ibr_repca_LineMVA[i] = Sbranch
            self.Init_ibr_repca_QVdbout[i] = 0.0
            self.Init_ibr_repca_fdbout[i] = 0.0
            self.Init_ibr_repca_vq2qPI[i] = 0.0
            self.Init_ibr_repca_p2pPI[i] = 0.0

        ## End for

        return


    def InitPLL(self, pfd, dyd):
        for i in range(dyd.ibr_wecc_n):
            ibrbus = pfd.ibr_bus[i]
            ibrbus_idx = np.where(pfd.bus_num == ibrbus)

            self.Init_ibr_pll_ze[i] = 0.0
            self.Init_ibr_pll_de[i] = pfd.bus_Va[ibrbus_idx]
            self.Init_ibr_pll_we[i] = 1.0

        ## End for

        return


    def InitBusMea(self, pfd):
        self.Init_pll_ze = np.zeros(len(pfd.bus_num))
        self.Init_pll_de = pfd.bus_Va
        self.Init_pll_we = np.ones(len(pfd.bus_num))

        self.Init_vt = pfd.bus_Vm  # calculated volt mag
        self.Init_vtm = pfd.bus_Vm  # measured volt mag (after calc)
        self.Init_dvtm = pfd.bus_Vm * 0  # measured dvm/dt

        return


    def InitLoad(self, pfd):
        for i in range(len(pfd.load_bus)):
            busi_idx = np.where(pfd.bus_num == pfd.load_bus[i])[0][0]
            self.Init_PL[i] = pfd.load_MW[i] / pfd.basemva
            self.Init_QL[i] = pfd.load_Mvar[i] / pfd.basemva
            self.Init_ZL_mag[i] = pfd.bus_Vm[busi_idx] * pfd.bus_Vm[busi_idx] / np.sqrt(
                pfd.load_MW[i] * pfd.load_MW[i] + pfd.load_Mvar[i] * pfd.load_Mvar[i]) * pfd.basemva
            if pfd.load_MW[i] > 0:
                self.Init_ZL_ang[i] = np.arctan(pfd.load_Mvar[i] / pfd.load_MW[i])
            else:
                self.Init_ZL_ang[i] = np.arctan(pfd.load_Mvar[i] / pfd.load_MW[i]) + np.pi
            ## End if
        ## End for

        return


    def MergeMacG(self, pfd, dyd, ts, i_gentrip = []):
        # self.Init_net_Gt0 = sp.coo_matrix((self.Init_net_G0_data, (self.Init_net_G0_rows, self.Init_net_G0_cols)),
        #                                  shape=(self.Init_net_N, self.Init_net_N)
        #                                  ).tolil()


        self.Init_mac_Ld = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Lq = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Requiv = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Gequiv = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Rd = np.zeros(len(pfd.gen_bus))
        self.Init_mac_Rq = np.zeros(len(pfd.gen_bus))
        self.Init_mac_Rd1 = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Rd2 = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Rq1 = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Rq2 = np.zeros((len(pfd.gen_bus), 3, 3))
        self.Init_mac_Rav = np.zeros(len(pfd.gen_bus))
        self.Init_mac_alpha = np.zeros(len(pfd.gen_bus))

        self.Init_mac_Rd1inv = np.zeros((len(pfd.gen_bus), 2, 2))
        self.Init_mac_Rd_coe = np.zeros((len(pfd.gen_bus), 2))
        self.Init_mac_Rq1inv = np.zeros((len(pfd.gen_bus), 2, 2))
        self.Init_mac_Rq_coe = np.zeros((len(pfd.gen_bus), 2))

        nexisting = self.resize_G0_data(9*len(pfd.gen_bus))

        for i in range(len(pfd.gen_bus)):
            self.Init_mac_alpha[i] = 99.0 / 101.0
            Ra = dyd.ec_Ra[i]
            L0 = dyd.ec_L0[i]

            self.Init_mac_Ld[i][0][0] = dyd.ec_Lad[i] + dyd.ec_Ll[i]
            self.Init_mac_Ld[i][0][1] = dyd.ec_Lad[i]
            self.Init_mac_Ld[i][0][2] = dyd.ec_Lad[i]
            self.Init_mac_Ld[i][1][0] = dyd.ec_Lad[i]
            self.Init_mac_Ld[i][1][1] = dyd.ec_Lffd[i]
            self.Init_mac_Ld[i][1][2] = dyd.ec_Lf1d[i]
            self.Init_mac_Ld[i][2][0] = dyd.ec_Lad[i]
            self.Init_mac_Ld[i][2][1] = dyd.ec_Lf1d[i]
            self.Init_mac_Ld[i][2][2] = dyd.ec_L11d[i]

            self.Init_mac_Lq[i][0][0] = dyd.ec_Laq[i] + dyd.ec_Ll[i]
            self.Init_mac_Lq[i][0][1] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][0][2] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][1][0] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][1][1] = dyd.ec_L11q[i]
            self.Init_mac_Lq[i][1][2] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][2][0] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][2][1] = dyd.ec_Laq[i]
            self.Init_mac_Lq[i][2][2] = dyd.ec_L22q[i]

            # previously finalized
            self.Init_mac_Rd1[i][0][0] = dyd.ec_Ra[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][0][0]
            self.Init_mac_Rd1[i][0][1] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][0][1]
            self.Init_mac_Rd1[i][0][2] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][0][2]
            self.Init_mac_Rd1[i][1][0] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][1][0]
            self.Init_mac_Rd1[i][1][1] = dyd.ec_Rfd[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][1][1]
            self.Init_mac_Rd1[i][1][2] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][1][2]
            self.Init_mac_Rd1[i][2][0] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][0]
            self.Init_mac_Rd1[i][2][1] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][1]
            self.Init_mac_Rd1[i][2][2] = dyd.ec_R1d[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][2]

            self.Init_mac_Rd2[i][0][0] = dyd.ec_Ra[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Ld[i][0][0]
            self.Init_mac_Rd2[i][0][1] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][0][1]
            self.Init_mac_Rd2[i][0][2] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][0][2]
            self.Init_mac_Rd2[i][1][0] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][1][0]
            self.Init_mac_Rd2[i][1][1] = dyd.ec_Rfd[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Ld[i][1][1]
            self.Init_mac_Rd2[i][1][2] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][1][2]
            self.Init_mac_Rd2[i][2][0] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][0]
            self.Init_mac_Rd2[i][2][1] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][1]
            self.Init_mac_Rd2[i][2][1] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Ld[i][2][1]
            self.Init_mac_Rd2[i][2][2] = dyd.ec_R1d[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Ld[i][2][2]

            self.Init_mac_Rq1[i][0][0] = dyd.ec_Ra[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][0][0]
            self.Init_mac_Rq1[i][0][1] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][0][1]
            self.Init_mac_Rq1[i][0][2] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][0][2]
            self.Init_mac_Rq1[i][1][0] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][1][0]
            self.Init_mac_Rq1[i][1][1] = dyd.ec_R1q[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][1][1]
            self.Init_mac_Rq1[i][1][2] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][1][2]
            self.Init_mac_Rq1[i][2][0] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][2][0]
            self.Init_mac_Rq1[i][2][1] = 0.0 + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][2][1]
            self.Init_mac_Rq1[i][2][2] = dyd.ec_R2q[i] + (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][2][2]

            self.Init_mac_Rq2[i][0][0] = dyd.ec_Ra[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Lq[i][0][0]
            self.Init_mac_Rq2[i][0][1] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][0][1]
            self.Init_mac_Rq2[i][0][2] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][0][2]
            self.Init_mac_Rq2[i][1][0] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][1][0]
            self.Init_mac_Rq2[i][1][1] = dyd.ec_R1q[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Lq[i][1][1]
            self.Init_mac_Rq2[i][1][2] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][1][2]
            self.Init_mac_Rq2[i][2][0] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][2][0]
            self.Init_mac_Rq2[i][2][1] = 0.0 - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * self.Init_mac_Lq[i][2][1]
            self.Init_mac_Rq2[i][2][2] = dyd.ec_R2q[i] * self.Init_mac_alpha[i] - (1 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * \
                                         self.Init_mac_Lq[i][2][2]







            temp_det = self.Init_mac_Rd1[i][1][1] * self.Init_mac_Rd1[i][2][2] - self.Init_mac_Rd1[i][1][2] * \
                       self.Init_mac_Rd1[i][2][1]
            Rd1inv = [[self.Init_mac_Rd1[i][2][2] / temp_det, - self.Init_mac_Rd1[i][1][2] / temp_det],
                      [- self.Init_mac_Rd1[i][2][1] / temp_det, self.Init_mac_Rd1[i][1][1] / temp_det]]
            templ = [self.Init_mac_Rd1[i][0][1] * Rd1inv[0][0] + self.Init_mac_Rd1[i][0][2] * Rd1inv[1][0],
                     self.Init_mac_Rd1[i][0][1] * Rd1inv[0][1] + self.Init_mac_Rd1[i][0][2] * Rd1inv[1][1]]
            self.Init_mac_Rd1inv[i] = np.asarray(Rd1inv)
            self.Init_mac_Rd[i] = self.Init_mac_Rd1[i][0][0] - (
                    templ[0] * self.Init_mac_Rd1[i][1][0] + templ[1] * self.Init_mac_Rd1[i][2][0])
            self.Init_mac_Rd_coe[i] = np.asarray(templ)

            temp_det = self.Init_mac_Rq1[i][1][1] * self.Init_mac_Rq1[i][2][2] - self.Init_mac_Rq1[i][1][2] * self.Init_mac_Rq1[i][2][1]
            Rq1inv = [[self.Init_mac_Rq1[i][2][2] / temp_det, - self.Init_mac_Rq1[i][1][2] / temp_det],
                      [- self.Init_mac_Rq1[i][2][1] / temp_det, self.Init_mac_Rq1[i][1][1] / temp_det]]
            templ = [self.Init_mac_Rq1[i][0][1] * Rq1inv[0][0] + self.Init_mac_Rq1[i][0][2] * Rq1inv[1][0],
                     self.Init_mac_Rq1[i][0][1] * Rq1inv[0][1] + self.Init_mac_Rq1[i][0][2] * Rq1inv[1][1]]
            self.Init_mac_Rq1inv[i] = np.asarray(Rq1inv)

            self.Init_mac_Rq[i] = self.Init_mac_Rq1[i][0][0] - (templ[0] * self.Init_mac_Rq1[i][1][0] + templ[1] * self.Init_mac_Rq1[i][2][0])
            self.Init_mac_Rq_coe[i] = np.asarray(templ)
            # if self.Init_mac_Rq_coe.size == 0:
            #     self.Init_mac_Rq_coe = np.asarray(templ)
            # else:
            #     self.Init_mac_Rq_coe = np.vstack((self.Init_mac_Rq_coe, np.asarray(templ)))

            self.Init_mac_Rav[i] = (self.Init_mac_Rd[i] + self.Init_mac_Rq[i]) / 2.0
            R0 = Ra + (1.0 + self.Init_mac_alpha[i]) / (ts * pfd.ws) * L0

            Rs = (R0 + 2.0 * self.Init_mac_Rav[i]) / 3.0
            Rm = (R0 - self.Init_mac_Rav[i]) / 3.0

            self.Init_mac_Requiv[i][0][0] = Rs
            self.Init_mac_Requiv[i][0][1] = Rm
            self.Init_mac_Requiv[i][0][2] = Rm
            self.Init_mac_Requiv[i][1][0] = Rm
            self.Init_mac_Requiv[i][1][1] = Rs
            self.Init_mac_Requiv[i][1][2] = Rm
            self.Init_mac_Requiv[i][2][0] = Rm
            self.Init_mac_Requiv[i][2][1] = Rm
            self.Init_mac_Requiv[i][2][2] = Rs

            tempA = np.asarray([[Rs, Rm, Rm], [Rm, Rs, Rm], [Rm, Rm, Rs]])
            tempAinv = np.linalg.inv(tempA)

            genbus_idx = int(np.where(pfd.bus_num == pfd.gen_bus[i])[0])
            self.Init_mac_Gequiv[i][0][0] = tempAinv[0][0] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][0][1] = tempAinv[0][1] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][0][2] = tempAinv[0][2] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][1][0] = tempAinv[1][0] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][1][1] = tempAinv[1][1] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][1][2] = tempAinv[1][2] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][2][0] = tempAinv[2][0] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][2][1] = tempAinv[2][1] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]
            self.Init_mac_Gequiv[i][2][2] = tempAinv[2][2] / dyd.base_Zs[i] * self.Init_net_ZbaseA[genbus_idx]

            N1 = len(pfd.bus_num)
            N2 = len(pfd.bus_num) * 2
            idx = nexisting + 9*i
            if i_gentrip:
                if i_gentrip != i:
                    self.addtoG0(genbus_idx, genbus_idx,
                                 self.Init_mac_Gequiv[i][0][0])
                    self.addtoG0(genbus_idx, genbus_idx + N1,
                                 self.Init_mac_Gequiv[i][0][1])
                    self.addtoG0(genbus_idx, genbus_idx + N2,
                                 self.Init_mac_Gequiv[i][0][2])
                    self.addtoG0(genbus_idx + N1, genbus_idx,
                                 self.Init_mac_Gequiv[i][1][0])
                    self.addtoG0(genbus_idx + N1, genbus_idx + N1,
                                 self.Init_mac_Gequiv[i][1][1])
                    self.addtoG0(genbus_idx + N1, genbus_idx + N2,
                                 self.Init_mac_Gequiv[i][1][2])
                    self.addtoG0(genbus_idx + N2, genbus_idx,
                                 self.Init_mac_Gequiv[i][2][0])
                    self.addtoG0(genbus_idx + N2, genbus_idx + N1,
                                 self.Init_mac_Gequiv[i][2][1])
                    self.addtoG0(genbus_idx + N2, genbus_idx + N2,
                                 self.Init_mac_Gequiv[i][2][2])
            else:    # 6 of them, which are mutual between 3 phases will be added

                self.addtoG0(genbus_idx, genbus_idx,
                             self.Init_mac_Gequiv[i][0][0])
                self.addtoG0(genbus_idx, genbus_idx + N1,
                             self.Init_mac_Gequiv[i][0][1])
                self.addtoG0(genbus_idx, genbus_idx + N2,
                             self.Init_mac_Gequiv[i][0][2])
                self.addtoG0(genbus_idx + N1, genbus_idx,
                             self.Init_mac_Gequiv[i][1][0])
                self.addtoG0(genbus_idx + N1, genbus_idx + N1,
                             self.Init_mac_Gequiv[i][1][1])
                self.addtoG0(genbus_idx + N1, genbus_idx + N2,
                             self.Init_mac_Gequiv[i][1][2])
                self.addtoG0(genbus_idx + N2, genbus_idx,
                             self.Init_mac_Gequiv[i][2][0])
                self.addtoG0(genbus_idx + N2, genbus_idx + N1,
                             self.Init_mac_Gequiv[i][2][1])
                self.addtoG0(genbus_idx + N2, genbus_idx + N2,
                             self.Init_mac_Gequiv[i][2][2])
            ## End if i_gentrip != i
        ## End if i_gentrip

        return


    def MergeIbrG(self, pfd, dyd, ts, i_ibrtrip = []):
        damptrap = 10 # better to be consistent with line 151 in lib_numba.py
        ws = pfd.ws

        N1 = len(pfd.bus_num)
        N2 = N1 * 2
        N3 = N1 * 3
        Nibr = dyd.ibr_epri_n

        Nnet = len(self.Init_net_G0_data) - 12*Nibr
        Nbch = len(self.Init_net_coe0) - 6*Nibr

        self.Init_net_Vt = np.zeros(3*N1 + 3*Nibr, dtype=np.complex128)

        # get complex bus voltages
        Init_net_VtA = self.Init_net_Vt[:N1]
        Init_net_VtB = self.Init_net_Vt[N1:N2]
        Init_net_VtC = self.Init_net_Vt[N2:N3]
        Init_ibr_VinA = self.Init_net_Vt[N3:N3+Nibr]
        Init_ibr_VinB = self.Init_net_Vt[N3+Nibr:N3+2*Nibr]
        Init_ibr_VinC = self.Init_net_Vt[N3+2*Nibr:N3+3*Nibr]


        for i in range(N1):

            Vt_temp = complex(pfd.bus_Vm[i] * np.cos(pfd.bus_Va[i]),
                            pfd.bus_Vm[i] * np.sin(pfd.bus_Va[i]))

            Init_net_VtA[i] = Vt_temp
            Init_net_VtB[i] = Vt_temp * complex(-0.5, -0.5*np.sqrt(3.0))
            Init_net_VtC[i] = Vt_temp * complex(-0.5, 0.5*np.sqrt(3.0))

        ## End for

        for i in range(Nibr):

            ibrbus = dyd.ibr_epri_bus[i]
            ibrid = dyd.ibr_epri_id[i]
            # if len(ibrid)==1:
            #     ibrid = ibrid + ' '

            ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0]
            ibrid_idx = np.where(pfd.ibr_id[ibrbus_idx] == ibrid)[0][0]
            ibrbus_idx = ibrbus_idx[ibrid_idx]

            # ibrbus = dyd.ibr_epri_bus[i]
            # ibrbus_idx = np.where(pfd.ibr_bus == ibrbus)[0][0]

            bus_idx = np.where(pfd.bus_num == ibrbus)[0][0]

            Vm = pfd.bus_Vm[bus_idx]
            Va = pfd.bus_Va[bus_idx]

            Vt = complex(Vm * np.cos(Va), Vm * np.sin(Va))
            St = complex(pfd.ibr_MW[ibrbus_idx], pfd.ibr_Mvar[ibrbus_idx]) / pfd.ibr_MVA_base[ibrbus_idx]
            It = np.conjugate(St/Vt)

            Vc = Vt + It*complex(dyd.ibr_epri_Rchoke[i], dyd.ibr_epri_Lchoke[i])
            Ic = Vc * complex(0, dyd.ibr_epri_Cfilt[i]) + Vc / dyd.ibr_epri_Rdamp[i]
            Iin = It + Ic
            Vin = Vc + Iin*complex(dyd.ibr_epri_Rchoke[i], dyd.ibr_epri_Lchoke[i])

            Init_ibr_VinA[i] = Vc
            Init_ibr_VinB[i] = Vc * complex(-0.5, -0.5*np.sqrt(3.0))
            Init_ibr_VinC[i] = Vc * complex(-0.5, 0.5*np.sqrt(3.0))

        ## End for

        for i in range(Nibr):
            ibrbus_idx = int(np.where(pfd.bus_num == dyd.ibr_epri_bus[i])[0])
            newbus_idx = N3 + i


            Fidx = ibrbus_idx
            Tidx = newbus_idx
            R = dyd.ibr_epri_Rchoke[i] * pfd.basemva / dyd.ibr_epri_basemva[i]
            X = dyd.ibr_epri_Lchoke[i] * pfd.basemva / dyd.ibr_epri_basemva[i]


            
            if X>0:
                L =  X / ws
                Rp = damptrap * (20.0 / 3.0 * 2.0 * L / ts)
                Rp_inv = 1.0 / Rp

                Req = (1 + R * (ts / 2.0 / L + Rp_inv)) / (ts / 2.0 / L + Rp_inv)
                icf = (1 - R * (ts / 2.0 / L - Rp_inv)) / (1 + R * (ts / 2.0 / L + Rp_inv))
                Gv1 = (ts / 2.0 / L - Rp_inv) / (1 + R * (ts / 2.0 / L + Rp_inv))
            elif X<0:
                CL = - 1 / X / ws
                Req = R + ts / 2 / CL
                icf = (2*R*CL - ts) / (2*R*CL + ts)

            C = dyd.ibr_epri_Cfilt[i] / ws / pfd.basemva * dyd.ibr_epri_basemva[i]
            Rc = ts / 2.0 / C 
            Rd = dyd.ibr_epri_Rdamp[i] * pfd.basemva / dyd.ibr_epri_basemva[i]

            self.Init_ibr_epri_Req[i] = Req
            self.Init_ibr_epri_icf[i] = icf
            self.Init_ibr_epri_Gv1[i] = Gv1

            # R-L branch
            idx = Nnet + 12*i
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx, Fidx, Fidx, 1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+1, Tidx, Tidx, 2 / Req + 1 / Rd + 1/ Rc)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+2, Fidx, Tidx, -1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+3, Tidx, Fidx, -1 / Req)

            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+4, Fidx+N1, Fidx+N1, 1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+5, Tidx+Nibr, Tidx+Nibr, 2 / Req + 1 / Rd + 1/ Rc)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+6, Fidx+N1, Tidx+Nibr, -1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+7, Tidx+Nibr, Fidx+N1, -1 / Req)

            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+8, Fidx+N2, Fidx+N2, 1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+9, Tidx+2*Nibr, Tidx+2*Nibr, 2 / Req + 1 / Rd + 1/ Rc)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+10, Fidx+N2, Tidx+2*Nibr, -1 / Req)
            numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+11, Tidx+2*Nibr, Fidx+N2, -1 / Req)

            iA_temp = (self.Init_net_Vt[Fidx] - self.Init_net_Vt[Tidx]) / complex(R, X)
            iB_temp = (self.Init_net_Vt[Fidx + N1] - self.Init_net_Vt[Tidx + Nibr]) / complex(R, X)
            iC_temp = (self.Init_net_Vt[Fidx + N2] - self.Init_net_Vt[Tidx + 2*Nibr]) / complex(R, X)


            coe_idx = Nbch + 6*i
            self.Init_net_coe0[coe_idx,:]   = np.array([Fidx, Tidx, Req, icf, Gv1, R, X, 0.0, iA_temp])
            self.Init_net_coe0[coe_idx+1,:] = np.array([Fidx+N1, Tidx + Nibr, Req, icf, Gv1, R, X, 0.0, iB_temp])
            self.Init_net_coe0[coe_idx+2,:] = np.array([Fidx+N2, Tidx + 2*Nibr, Req, icf, Gv1, R, X, 0.0, iC_temp])

            # C//R branch
            # numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx, Fidx, Fidx, 1 / Req)
            # numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+1, Tidx, Tidx, 1 / Req)
            # numba_set_coo(self.Init_net_G0_rows, self.Init_net_G0_cols, self.Init_net_G0_data, idx+2, Fidx, Tidx, -1 / Req)

            iA_temp = self.Init_net_Vt[Tidx] * (complex(0, ws * C) + 1/Rd)
            iB_temp = self.Init_net_Vt[Tidx + Nibr] * (complex(0, ws * C) + 1/Rd)
            iC_temp = self.Init_net_Vt[Tidx + 2*Nibr] * (complex(0, ws * C) + 1/Rd)

            self.Init_net_coe0[coe_idx + 3, :] = np.array([Tidx, -1, 1 / (1/Rd + 1/Rc), -1, 1/Rd - 1/Rc, 0.0, 0.0, C, iA_temp])
            self.Init_net_coe0[coe_idx + 4, :] = np.array([Tidx + Nibr, -1, 1 / (1/Rd + 1/Rc), -1, 1/Rd - 1/Rc, 0.0, 0.0, C, iB_temp])
            self.Init_net_coe0[coe_idx + 5, :] = np.array([Tidx + 2*Nibr, -1, 1 / (1/Rd + 1/Rc), -1, 1/Rd - 1/Rc, 0.0, 0.0, C, iC_temp])

        ## End for

        return


    def InitIhis(self):
        # calculate pre and his terms of branch current
        self.Init_net_V = np.real(self.Init_net_Vt)
        self.Init_brch_Ipre = np.real(self.Init_net_coe0[:, 8])
        self.Init_node_Ihis = np.zeros(self.Init_net_N)
        self.Init_brch_Ihis = np.zeros(len(self.Init_brch_Ipre))

        #### BEGIN FOOR LOOP ####
        for i in range(len(self.Init_brch_Ipre)):
            Fidx = int(self.Init_net_coe0[i, 0].real)
            Tidx = int(self.Init_net_coe0[i, 1].real)
            if Tidx == -1:
                if self.Init_net_coe0[i, 2] == 0:
                    continue
                brch_Ihis_temp = self.Init_net_coe0[i, 3] * self.Init_brch_Ipre[i] + self.Init_net_coe0[i, 4] * (self.Init_net_V[Fidx])
            else:
                brch_Ihis_temp = self.Init_net_coe0[i, 3] * self.Init_brch_Ipre[i] + self.Init_net_coe0[i, 4] * (self.Init_net_V[Fidx] - self.Init_net_V[Tidx])
                self.Init_node_Ihis[Tidx] += brch_Ihis_temp.real

            self.Init_brch_Ihis[i] = brch_Ihis_temp.real
            self.Init_node_Ihis[Fidx] -= brch_Ihis_temp.real
        #### END FOR LOOP ####

        return


    def addtoG0(self, row, col, addedvalue):
        found_flag = 0
        for i in range(len(self.Init_net_G0_data)):
            if (self.Init_net_G0_rows[i] == row) & (self.Init_net_G0_cols[i] == col):
                found_flag = 1
                self.Init_net_G0_data[i] += addedvalue
                return

        if found_flag == 0:
            self.Init_net_G0_data = np.append(self.Init_net_G0_data, addedvalue)
            self.Init_net_G0_rows = np.append(self.Init_net_G0_rows, row)
            self.Init_net_G0_cols = np.append(self.Init_net_G0_cols, col)

        return

    # def append_to_G0_data(self, rows, cols, values):

    #     self.Init_net_G0_rows = np.hstack((self.Init_net_G0_rows, np.array(rows)))
    #     self.Init_net_G0_cols = np.hstack((self.Init_net_G0_cols, np.array(cols)))
    #     self.Init_net_G0_data = np.hstack((self.Init_net_G0_data, np.array(values)))

    #     return

    def resize_G0_data(self, nentries_to_add):

        # Resize admittance matrix data to allow for generator data
        # NOTE: These will be used to create scipy.sparse.coo_matrix which
        # interprets duplicate entries as values to be added

        nexisting = self.Init_net_G0_rows.size

        assert(self.Init_net_G0_rows.size == self.Init_net_G0_cols.size == self.Init_net_G0_data.size)

        nentries = nexisting + nentries_to_add

        G0_rows = self.Init_net_G0_rows
        self.Init_net_G0_rows = np.zeros(nentries, dtype=np.int64)
        self.Init_net_G0_rows[:nexisting] = G0_rows[:]

        G0_cols = self.Init_net_G0_cols
        self.Init_net_G0_cols = np.zeros(nentries, dtype=np.int64)
        self.Init_net_G0_cols[:nexisting] = G0_cols[:]

        G0_data = self.Init_net_G0_data
        self.Init_net_G0_data = np.zeros(nentries, dtype=np.float64)
        self.Init_net_G0_data[:nexisting] = G0_data[:]

        return nexisting

    def CalcGnGinv(self, mode):
        # calculate the inverse of net conductance matrix
        self.Init_net_G0 = sp.coo_matrix((self.Init_net_G0_data,
                                          (self.Init_net_G0_rows, self.Init_net_G0_cols)),
                                            shape=(self.Init_net_N, self.Init_net_N)
                                            ).tolil()
        if mode == 'inv':
            self.Init_net_G0_inv = la.inv(self.Init_net_G0.tocsc())
            # self.Init_net_G0_inv = np.linalg.inv(self.Init_net_G0.toarray())
        elif mode == 'lu':

            # # TODO: The singular issue below is cause by zero IBR source impedance and zero shunt in the sole branch 
            # # connecting IBR with the grid. This issue should not be there when a detailed IBR model with LCL filter, say EPRI's simple IBR model. 
            # # A cheap and dirty solution for now is to add a small artificial G component to the diagonal entries of matrix G0.
            # # =========================Cheap and dirty solution============================
            # # Added on 4/12/2023
            # self.Init_net_G0 = sp.csr_matrix(self.Init_net_G0)
            # for i in range(self.Init_net_N):
            #     self.Init_net_G0[i,i] = self.Init_net_G0[i,i] + 0.01           
            # # =========================End of the cheap and dirty solution=================

            self.Init_net_G0_lu = la.splu(self.Init_net_G0.tocsc())
        elif mode == 'bbd':
            pass
        else:
            raise ValueError('Unrecognized mode: {}'.format(mode))

        self.admittance_mode = mode

        return


    def CombineX(self, pfd, dyd):

        xi = 0

        # machine states

        # GENROU
        dyd.gen_genrou_xi_st = xi
        for i in range(len(pfd.gen_bus)):
            self.Init_x = np.append(self.Init_x, self.Init_mac_dt[i])  # 1
            self.Init_x = np.append(self.Init_x, 1.0*pfd.ws)  # 2
            self.Init_x = np.append(self.Init_x, self.Init_mac_id[i])  # 3
            self.Init_x = np.append(self.Init_x, self.Init_mac_iq[i])  # 4
            self.Init_x = np.append(self.Init_x, self.Init_mac_ifd[i])  # 5
            self.Init_x = np.append(self.Init_x, self.Init_mac_i1d[i])  # 6
            self.Init_x = np.append(self.Init_x, self.Init_mac_i1q[i])  # 7
            self.Init_x = np.append(self.Init_x, self.Init_mac_i2q[i])  # 8
            self.Init_x = np.append(self.Init_x, self.Init_mac_ed[i])  # 9
            self.Init_x = np.append(self.Init_x, self.Init_mac_eq[i])  # 10
            self.Init_x = np.append(self.Init_x, self.Init_mac_psyd[i])  # 11
            self.Init_x = np.append(self.Init_x, self.Init_mac_psyq[i])  # 12
            self.Init_x = np.append(self.Init_x, self.Init_mac_psyfd[i])  # 13
            self.Init_x = np.append(self.Init_x, self.Init_mac_psy1q[i])  # 14
            self.Init_x = np.append(self.Init_x, self.Init_mac_psy1d[i])  # 15
            self.Init_x = np.append(self.Init_x, self.Init_mac_psy2q[i])  # 16
            self.Init_x = np.append(self.Init_x, self.Init_mac_te[i])  # 17
            self.Init_x = np.append(self.Init_x, self.Init_mac_qe[i])  # 18

            xi = xi + dyd.gen_genrou_odr

        # SEXS exciter model
        dyd.exc_sexs_xi_st = xi
        for i in range(dyd.exc_sexs_n):
            self.Init_x = np.append(self.Init_x, self.Init_mac_v1[i])  # 1
            self.Init_x = np.append(self.Init_x, self.Init_mac_EFD[i])  # 2

            xi = xi + dyd.exc_sexs_odr

        # TGOV1 governor model
        dyd.gov_tgov1_xi_st = xi
        for i in range(dyd.gov_tgov1_n):
            self.Init_x = np.append(self.Init_x, self.Init_tgov1_p1[i])    # 1
            self.Init_x = np.append(self.Init_x, self.Init_tgov1_p2[i])    # 2

            self.Init_x = np.append(self.Init_x, self.Init_mac_pm[int(dyd.gov_tgov1_idx[i])])    # 3

            xi = xi + dyd.gov_tgov1_odr

        # HYGOV governor model
        dyd.gov_hygov_xi_st = xi
        for i in range(dyd.gov_hygov_n):
            self.Init_x = np.append(self.Init_x, self.Init_hygov_xe[i])  # 1
            self.Init_x = np.append(self.Init_x, self.Init_hygov_xc[i])  # 2
            self.Init_x = np.append(self.Init_x, self.Init_hygov_xg[i])  # 3
            self.Init_x = np.append(self.Init_x, self.Init_hygov_xq[i])  # 4

            self.Init_x = np.append(self.Init_x, self.Init_mac_pm[int(dyd.gov_hygov_idx[i])])  # 5

            xi = xi + dyd.gov_hygov_odr

        # GAST governor model
        dyd.gov_gast_xi_st = xi
        for i in range(dyd.gov_gast_n):
            self.Init_x = np.append(self.Init_x, self.Init_gast_p1[i])  # 1
            self.Init_x = np.append(self.Init_x, self.Init_gast_p2[i])  # 2
            self.Init_x = np.append(self.Init_x, self.Init_gast_p3[i])  # 3

            self.Init_x = np.append(self.Init_x, self.Init_mac_pm[int(dyd.gov_gast_idx[i])])  # 4

            xi = xi + dyd.gov_gast_odr

        # IEEEST
        dyd.pss_ieeest_xi_st = xi
        for i in range(dyd.pss_ieeest_n):
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y1[i])  # 1
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y2[i])  # 2
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y3[i])  # 3
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y4[i])  # 4
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y5[i])  # 5
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y6[i])  # 6
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_y7[i])  # 7
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_x1[i])  # 8
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_x2[i])  # 9
            self.Init_x = np.append(self.Init_x, self.Init_ieeest_vs[i])  # 10

            xi = xi + dyd.pss_ieeest_odr

        # self.Init_gen_N = 22
        for i in range(dyd.ibr_wecc_n):
            # regca
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_s0[i])  # 1
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_s1[i])  # 2
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_s2[i])  # 3
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_Vmp[i])  # 4
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_Vap[i])  # 5
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_i1[i])  # 6
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_i2[i])  # 7
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_regca_ip2rr[i])  # 8

            # reecb
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s0[i])  # 9
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s1[i])  # 10
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s2[i])  # 11
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s3[i])  # 12
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s4[i])  # 13
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_s5[i])  # 14
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_Ipcmd[i])  # 15
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_Iqcmd[i])  # 16
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_Pref[i])  # 17
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_Qext[i])  # 18
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_q2vPI[i])  # 19
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_reecb_v2iPI[i])  # 20

            # repca
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s0[i])  # 21
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s1[i])  # 22
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s2[i])  # 23
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s3[i])  # 24
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s4[i])  # 25
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s5[i])  # 26
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_s6[i])  # 27
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_Vref[i])  # 28
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_Qref[i])  # 29
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_Freq_ref[i])  # 30
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_Plant_pref[i])  # 31
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_LineMW[i])  # 32
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_LineMvar[i])  # 33
            self.Init_x_ibr = np.append(self.Init_x_ibr, np.abs(self.Init_ibr_repca_LineMVA[i]))  # 34
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_QVdbout[i])  # 35
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_fdbout[i])  # 36
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_vq2qPI[i])  # 37
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_p2pPI[i])  # 38
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_Freq_ref[i])  # 39 Vf
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_LineMW[i])  # 40 Pe
            self.Init_x_ibr = np.append(self.Init_x_ibr, self.Init_ibr_repca_LineMvar[i])  # 41 Qe
        
        # EPRI IBR model
        for i in range(dyd.ibr_epri_n):
            # selected states, inputs and outputs
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Ea[i])  # 1 
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Eb[i])  # 2 
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Ec[i])  # 3 
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Idref[i])  # 4 Idref
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Idref[i])  # 5 Id
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Iqref[i])  # 6 Iqref
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Iqref[i])  # 7 Iq
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Vd[i])  # 8
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Vq[i])  # 9 
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, 60.0)  # 10 
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Pref[i])  # 11
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Qref[i])  # 12
            self.Init_x_ibr_epri = np.append(self.Init_x_ibr_epri, self.Init_ibr_epri_Vref[i])  # 13

        # bus measurement
        for i in range(len(pfd.bus_num)):
            # volt freq and angle by PLL
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_pll_ze[i])   # 1 ze
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_pll_de[i])   # 2 de
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_pll_we[i])   # 3 we

            # volt mag measurement
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_vt[i])       # 4 vt
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_vtm[i])      # 5 vtm
            self.Init_x_bus = np.append(self.Init_x_bus, self.Init_dvtm[i])     # 6 dvtm

        # load
        for i in range(len(pfd.load_bus)):
            # volt freq and angle by PLL
            self.Init_x_load = np.append(self.Init_x_load, self.Init_ZL_mag[i])  # 1 ZL_mag
            self.Init_x_load = np.append(self.Init_x_load, self.Init_ZL_ang[i])  # 2 ZL_ang
            self.Init_x_load = np.append(self.Init_x_load, self.Init_PL[i])  # 3 PL
            self.Init_x_load = np.append(self.Init_x_load, self.Init_QL[i])  # 4 QL


