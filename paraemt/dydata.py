
import math
import numpy as np
import xlrd
# xlrd.xlsx.ensure_elementtree_imported(False, None)
# xlrd.xlsx.Element_has_iter = True

class DyData():

    # Maps string governor model names to integers
    gov_model_map = {
        'GAST':0,
        'HYGOV':1,
        'TGOV1':2,
    }

    def __init__(self):
        ## types
        self.gen_type = np.asarray([])
        self.exc_type = np.asarray([])
        self.gov_type = np.asarray([])
        self.pss_type = np.asarray([])

        self.gen_Ra = np.asarray([])  # pu on machine MVA base
        self.gen_X0 = np.asarray([])  # pu on machine MVA base


        ## gen
        self.gen_n = 0

        # GENROU
        self.gen_genrou_bus = np.asarray([])
        self.gen_genrou_id = np.asarray([])
        self.gen_genrou_Td0p = np.asarray([])
        self.gen_genrou_Td0pp = np.asarray([])
        self.gen_genrou_Tq0p = np.asarray([])
        self.gen_genrou_Tq0pp = np.asarray([])
        self.gen_H = np.asarray([])  # pu on machine MVA base
        self.gen_D = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xd = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xq = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xdp = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xqp = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xdpp = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_Xl = np.asarray([])  # pu on machine MVA base
        self.gen_genrou_S10 = np.asarray([])
        self.gen_genrou_S12 = np.asarray([])
        self.gen_genrou_idx = np.asarray([])
        self.gen_genrou_n = 0
        self.gen_genrou_xi_st = 0
        self.gen_genrou_odr = 18


        ## exc
        self.exc_n = 0

        # SEXS
        self.exc_sexs_bus = np.asarray([])
        self.exc_sexs_id = np.asarray([])
        self.exc_sexs_TA_o_TB = np.asarray([])
        self.exc_sexs_TA = np.asarray([])
        self.exc_sexs_TB = np.asarray([])
        self.exc_sexs_K = np.asarray([])
        self.exc_sexs_TE = np.asarray([])
        self.exc_sexs_Emin = np.asarray([])  # pu on EFD base
        self.exc_sexs_Emax = np.asarray([])  # pu on EFD base
        self.exc_sexs_idx = np.asarray([])
        self.exc_sexs_n = 0
        self.exc_sexs_xi_st = 0
        self.exc_sexs_odr = 2

        ## gov
        self.gov_n = 0

        # TGOV1
        self.gov_tgov1_bus = np.asarray([])
        self.gov_tgov1_id = np.asarray([])
        self.gov_tgov1_R = np.asarray([])  # pu on machine MVA base
        self.gov_tgov1_T1 = np.asarray([])
        self.gov_tgov1_Vmax = np.asarray([])  # pu on machine MVA base
        self.gov_tgov1_Vmin = np.asarray([])  # pu on machine MVA base
        self.gov_tgov1_T2 = np.asarray([])
        self.gov_tgov1_T3 = np.asarray([])
        self.gov_tgov1_Dt = np.asarray([])  # pu on machine MVA base
        self.gov_tgov1_idx = np.asarray([])
        self.gov_tgov1_n = 0
        self.gov_tgov1_xi_st = 0
        self.gov_tgov1_odr = 3

        # HYGOV
        self.gov_hygov_bus = np.asarray([])
        self.gov_hygov_id = np.asarray([])
        self.gov_hygov_R = np.asarray([])  # pu on machine MVA base
        self.gov_hygov_r = np.asarray([])  # pu on machine MVA base
        self.gov_hygov_Tr = np.asarray([])
        self.gov_hygov_Tf = np.asarray([])
        self.gov_hygov_Tg = np.asarray([])
        self.gov_hygov_VELM = np.asarray([])
        self.gov_hygov_GMAX = np.asarray([])
        self.gov_hygov_GMIN = np.asarray([])
        self.gov_hygov_TW = np.asarray([])
        self.gov_hygov_At = np.asarray([])
        self.gov_hygov_Dturb = np.asarray([])  # pu on machine MVA base
        self.gov_hygov_qNL = np.asarray([])
        self.gov_hygov_idx = np.asarray([])
        self.gov_hygov_n = 0
        self.gov_hygov_xi_st = 0
        self.gov_hygov_odr = 5

        # GAST
        self.gov_gast_bus = np.asarray([])
        self.gov_gast_id = np.asarray([])
        self.gov_gast_R = np.asarray([])
        self.gov_gast_T1 = np.asarray([])
        self.gov_gast_T2 = np.asarray([])
        self.gov_gast_T3 = np.asarray([])
        self.gov_gast_LdLmt = np.asarray([])
        self.gov_gast_KT = np.asarray([])
        self.gov_gast_VMAX = np.asarray([])
        self.gov_gast_VMIN = np.asarray([])
        self.gov_gast_Dturb = np.asarray([])
        self.gov_gast_idx = np.asarray([])
        self.gov_gast_n = 0
        self.gov_gast_xi_st = 0
        self.gov_gast_odr = 4

        ## pss
        self.pss_n = 0

        # IEEEST
        self.pss_ieeest_bus = np.asarray([])
        self.pss_ieeest_id = np.asarray([])
        self.pss_ieeest_A1 = np.asarray([])
        self.pss_ieeest_A2 = np.asarray([])
        self.pss_ieeest_A3 = np.asarray([])
        self.pss_ieeest_A4 = np.asarray([])
        self.pss_ieeest_A5 = np.asarray([])
        self.pss_ieeest_A6 = np.asarray([])
        self.pss_ieeest_T1 = np.asarray([])
        self.pss_ieeest_T2 = np.asarray([])
        self.pss_ieeest_T3 = np.asarray([])
        self.pss_ieeest_T4 = np.asarray([])
        self.pss_ieeest_T5 = np.asarray([])
        self.pss_ieeest_T6 = np.asarray([])
        self.pss_ieeest_KS = np.asarray([])
        self.pss_ieeest_LSMAX = np.asarray([])
        self.pss_ieeest_LSMIN = np.asarray([])
        self.pss_ieeest_VCU = np.asarray([])
        self.pss_ieeest_VCL = np.asarray([])
        self.pss_ieeest_idx = np.asarray([])
        self.pss_ieeest_n = 0
        self.pss_ieeest_xi_st = 0
        self.pss_ieeest_odr = 10

        self.ec_Lad = np.asarray([])
        self.ec_Laq = np.asarray([])
        self.ec_Ll = np.asarray([])
        self.ec_Lffd = np.asarray([])
        self.ec_L11d = np.asarray([])
        self.ec_L11q = np.asarray([])
        self.ec_L22q = np.asarray([])
        self.ec_Lf1d = np.asarray([])

        self.ec_Ld = np.asarray([])
        self.ec_Lq = np.asarray([])
        self.ec_L0 = np.asarray([])

        self.ec_Ra = np.asarray([])
        self.ec_Rfd = np.asarray([])
        self.ec_R1d = np.asarray([])
        self.ec_R1q = np.asarray([])
        self.ec_R2q = np.asarray([])

        self.ec_Lfd = np.asarray([])
        self.ec_L1d = np.asarray([])
        self.ec_L1q = np.asarray([])
        self.ec_L2q = np.asarray([])

        self.base_es = np.asarray([])
        self.base_is = np.asarray([])
        self.base_Is = np.asarray([])
        self.base_Zs = np.asarray([])
        self.base_Ls = np.asarray([])
        self.base_ifd = np.asarray([])
        self.base_efd = np.asarray([])
        self.base_Zfd = np.asarray([])
        self.base_Lfd = np.asarray([])

        ## IBR WECC parameters
        self.ibr_wecc_n = 0
        self.ibr_wecc_idx = np.asarray([])
        self.ibr_wecc_odr = 41

        self.ibr_kVbase = np.asarray([])
        self.ibr_MVAbase = np.asarray([])
        self.ibr_fbase = np.asarray([])
        self.ibr_Ibase = np.asarray([])

        self.ibr_regca_bus = np.asarray([])
        self.ibr_regca_id = np.asarray([])
        self.ibr_regca_LVPLsw = np.asarray([])
        self.ibr_regca_Tg = np.asarray([])
        self.ibr_regca_Rrpwr = np.asarray([])
        self.ibr_regca_Brkpt = np.asarray([])
        self.ibr_regca_Zerox = np.asarray([])
        self.ibr_regca_Lvpl1 = np.asarray([])
        self.ibr_regca_Volim = np.asarray([])
        self.ibr_regca_Lvpnt1 = np.asarray([])
        self.ibr_regca_Lvpnt0 = np.asarray([])
        self.ibr_regca_Iolim = np.asarray([])
        self.ibr_regca_Tfltr = np.asarray([])
        self.ibr_regca_Khv = np.asarray([])
        self.ibr_regca_Iqrmax = np.asarray([])
        self.ibr_regca_Iqrmin = np.asarray([])
        self.ibr_regca_Accel = np.asarray([])

        self.ibr_reecb_bus = np.asarray([])
        self.ibr_reecb_id = np.asarray([])
        self.ibr_reecb_PFFLAG = np.asarray([])
        self.ibr_reecb_VFLAG = np.asarray([])
        self.ibr_reecb_QFLAG = np.asarray([])
        self.ibr_reecb_PQFLAG = np.asarray([])
        self.ibr_reecb_Vdip = np.asarray([])
        self.ibr_reecb_Vup = np.asarray([])
        self.ibr_reecb_Trv = np.asarray([])
        self.ibr_reecb_dbd1 = np.asarray([])
        self.ibr_reecb_dbd2 = np.asarray([])
        self.ibr_reecb_Kqv = np.asarray([])
        self.ibr_reecb_Iqhl = np.asarray([])
        self.ibr_reecb_Iqll = np.asarray([])
        self.ibr_reecb_Vref0 = np.asarray([])
        self.ibr_reecb_Tp = np.asarray([])
        self.ibr_reecb_Qmax = np.asarray([])
        self.ibr_reecb_Qmin = np.asarray([])
        self.ibr_reecb_Vmax = np.asarray([])
        self.ibr_reecb_Vmin = np.asarray([])
        self.ibr_reecb_Kqp = np.asarray([])
        self.ibr_reecb_Kqi = np.asarray([])
        self.ibr_reecb_Kvp = np.asarray([])
        self.ibr_reecb_Kvi = np.asarray([])
        self.ibr_reecb_Tiq = np.asarray([])
        self.ibr_reecb_dPmax = np.asarray([])
        self.ibr_reecb_dPmin = np.asarray([])
        self.ibr_reecb_Pmax = np.asarray([])
        self.ibr_reecb_Pmin = np.asarray([])
        self.ibr_reecb_Imax = np.asarray([])
        self.ibr_reecb_Tpord = np.asarray([])

        self.ibr_repca_bus = np.asarray([])
        self.ibr_repca_id = np.asarray([])
        self.ibr_repca_remote_bus = np.asarray([])
        self.ibr_repca_branch_From_bus = np.asarray([])
        self.ibr_repca_branch_To_bus = np.asarray([])
        self.ibr_repca_branch_id = np.asarray([])
        self.ibr_repca_VCFlag = np.asarray([])
        self.ibr_repca_RefFlag = np.asarray([])
        self.ibr_repca_FFlag = np.asarray([])
        self.ibr_repca_Tfltr = np.asarray([])
        self.ibr_repca_Kp = np.asarray([])
        self.ibr_repca_Ki = np.asarray([])
        self.ibr_repca_Tft = np.asarray([])
        self.ibr_repca_Tfv = np.asarray([])
        self.ibr_repca_Vfrz = np.asarray([])
        self.ibr_repca_Rc = np.asarray([])
        self.ibr_repca_Xc = np.asarray([])
        self.ibr_repca_Kc = np.asarray([])
        self.ibr_repca_emax = np.asarray([])
        self.ibr_repca_emin = np.asarray([])
        self.ibr_repca_dbd1 = np.asarray([])
        self.ibr_repca_dbd2 = np.asarray([])
        self.ibr_repca_Qmax = np.asarray([])
        self.ibr_repca_Qmin = np.asarray([])
        self.ibr_repca_Kpg = np.asarray([])
        self.ibr_repca_Kig = np.asarray([])
        self.ibr_repca_Tp = np.asarray([])
        self.ibr_repca_fdbd1 = np.asarray([])
        self.ibr_repca_fdbd2 = np.asarray([])
        self.ibr_repca_femax = np.asarray([])
        self.ibr_repca_femin = np.asarray([])
        self.ibr_repca_Pmax = np.asarray([])
        self.ibr_repca_Pmin = np.asarray([])
        self.ibr_repca_Tg = np.asarray([])
        self.ibr_repca_Ddn = np.asarray([])
        self.ibr_repca_Dup = np.asarray([])

        # IBR EPRI
        self.ibr_epri_n = 0
        self.ibr_epri_odr = 13
        self.ibr_epri_idx = np.asarray([])
        self.ibr_epri_basemva = np.asarray([])
        self.ibr_epri_basekV = np.asarray([])

        self.ibr_epri_bus = np.asarray([])
        self.ibr_epri_id = np.asarray([])
        self.ibr_epri_Vdcbase = np.asarray([])
        self.ibr_epri_KpI = np.asarray([])
        self.ibr_epri_KiI = np.asarray([])
        self.ibr_epri_KpPLL = np.asarray([])
        self.ibr_epri_KiPLL = np.asarray([])
        self.ibr_epri_KpP = np.asarray([])
        self.ibr_epri_KiP = np.asarray([])
        self.ibr_epri_KpQ = np.asarray([])
        self.ibr_epri_KiQ = np.asarray([])
        self.ibr_epri_Imax = np.asarray([])
        self.ibr_epri_Pqflag = np.asarray([])
        self.ibr_epri_Vdip = np.asarray([])
        self.ibr_epri_Vup = np.asarray([])
        self.ibr_epri_Rchoke = np.asarray([])
        self.ibr_epri_Lchoke = np.asarray([])
        self.ibr_epri_Cfilt = np.asarray([])
        self.ibr_epri_Rdamp = np.asarray([])

        # IBR
        self.ibr_n = 0

        # PLL for bus freq/ang measurement
        self.pll_bus = np.asarray([])
        self.pll_ke = np.asarray([])
        self.pll_te = np.asarray([])
        self.bus_odr = 6

        # bus volt magnitude measurement
        self.vm_bus = np.asarray([])
        self.vm_te = np.asarray([])

        # measurement method
        self.mea_bus = np.asarray([])
        self.mea_method = np.asarray([])

        # load
        self.load_odr = 4

        # total order
        self.nx_ttl = 0
        self.nibr_ttl = 0


    def getdata(self, file_dydata, pfd, N):
        # detailed machine model
        dyn_data = xlrd.open_workbook(file_dydata)

        # gen
        ngen = int(len(pfd.gen_bus) / N)
        gen_data = dyn_data.sheet_by_index(0)
        self.gen_n = gen_data.ncols - 1
        if ngen > self.gen_n:
            print('Error: More generators in pf data than dyn data!!\n')
        elif self.gen_n > ngen:
            print('Warning: More generators in dyn data than pf data!!\n')
        for i in range(self.gen_n):
            flag = 0
            typei = str(gen_data.cell_value(2, i + 1))
            self.gen_type = np.append(self.gen_type, typei)
            if typei == 'GENROU':
                flag = 1
                self.gen_genrou_idx = np.append(self.gen_genrou_idx, i)
                self.gen_genrou_n = self.gen_genrou_n + 1
                nn = 1
                self.gen_genrou_bus = np.append(self.gen_genrou_bus, float(gen_data.cell_value(0, i + 1)))
                self.gen_genrou_id = np.append(self.gen_genrou_bus, gen_data.cell_value(1, i + 1))
                self.gen_genrou_Td0p = np.append(self.gen_genrou_Td0p, float(gen_data.cell_value(2 + nn, i + 1)))
                self.gen_genrou_Td0pp = np.append(self.gen_genrou_Td0pp, float(gen_data.cell_value(3 + nn, i + 1)))
                self.gen_genrou_Tq0p = np.append(self.gen_genrou_Tq0p, float(gen_data.cell_value(4 + nn, i + 1)))
                self.gen_genrou_Tq0pp = np.append(self.gen_genrou_Tq0pp, float(gen_data.cell_value(5 + nn, i + 1)))
                self.gen_H = np.append(self.gen_H, float(gen_data.cell_value(6 + nn, i + 1)))
                self.gen_D = np.append(self.gen_D, float(gen_data.cell_value(7 + nn, i + 1)))
                self.gen_genrou_Xd = np.append(self.gen_genrou_Xd, float(gen_data.cell_value(8 + nn, i + 1)))
                self.gen_genrou_Xq = np.append(self.gen_genrou_Xq, float(gen_data.cell_value(9 + nn, i + 1)))
                self.gen_genrou_Xdp = np.append(self.gen_genrou_Xdp, float(gen_data.cell_value(10 + nn, i + 1)))
                self.gen_genrou_Xqp = np.append(self.gen_genrou_Xqp, float(gen_data.cell_value(11 + nn, i + 1)))
                self.gen_genrou_Xdpp = np.append(self.gen_genrou_Xdpp, float(gen_data.cell_value(12 + nn, i + 1)))
                self.gen_genrou_Xl = np.append(self.gen_genrou_Xl, float(gen_data.cell_value(13 + nn, i + 1)))
                self.gen_genrou_S10 = np.append(self.gen_genrou_S10, float(gen_data.cell_value(14 + nn, i + 1)))
                self.gen_genrou_S12 = np.append(self.gen_genrou_S12, float(gen_data.cell_value(15 + nn, i + 1)))
                self.gen_Ra = np.append(self.gen_Ra, float(gen_data.cell_value(16 + nn, i + 1)))
                self.gen_X0 = np.append(self.gen_X0, float(gen_data.cell_value(13 + nn, i + 1)))

            if flag == 0:
                print('ERROR: Machine model not supported:')
                print(typei)
                print('\n')

        # exc
        exc_data = dyn_data.sheet_by_index(1)
        self.exc_n = exc_data.ncols - 1
        for i in range(self.exc_n):
            flag = 0
            typei = str(exc_data.cell_value(2, i + 1))
            self.exc_type = np.append(self.exc_type, typei)
            if typei == 'SEXS':
                flag = 1
                self.exc_sexs_idx = np.append(self.exc_sexs_idx, i)
                self.exc_sexs_n = self.exc_sexs_n + 1
                nn = 1
                self.exc_sexs_bus = np.append(self.exc_sexs_bus, float(exc_data.cell_value(0, i + 1)))
                self.exc_sexs_id = np.append(self.exc_sexs_id, exc_data.cell_value(1, i + 1))
                self.exc_sexs_TA_o_TB = np.append(self.exc_sexs_TA_o_TB, float(exc_data.cell_value(2 + nn, i + 1)))
                self.exc_sexs_TB = np.append(self.exc_sexs_TB, float(exc_data.cell_value(3 + nn, i + 1)))
                self.exc_sexs_K = np.append(self.exc_sexs_K, float(exc_data.cell_value(4 + nn, i + 1)))
                self.exc_sexs_TE = np.append(self.exc_sexs_TE, float(exc_data.cell_value(5 + nn, i + 1)))
                self.exc_sexs_Emin = np.append(self.exc_sexs_Emin, float(exc_data.cell_value(6 + nn, i + 1)))
                self.exc_sexs_Emax = np.append(self.exc_sexs_Emax, float(exc_data.cell_value(7 + nn, i + 1)))
                self.exc_sexs_TA = np.append(self.exc_sexs_TA, self.exc_sexs_TB[i] * self.exc_sexs_TA_o_TB[i])

            if flag == 0:
                print('ERROR: Exciter model not supported:')
                print(typei)
                print('\n')

        # gov
        gov_data = dyn_data.sheet_by_index(2)
        self.gov_n = gov_data.ncols - 1
        self.gov_type = np.empty(self.gov_n, dtype=int) #[""]*self.gov_n
        for i in range(self.gov_n):
            flag = 0
            typei = str(gov_data.cell_value(2, i + 1))
            if typei == 'TGOV1':
                flag = 1
                self.gov_tgov1_n = self.gov_tgov1_n + 1
                self.gov_tgov1_bus = np.append(self.gov_tgov1_bus, int(gov_data.cell_value(0, i + 1)))
                tempid = str(gov_data.cell_value(1, i + 1))
                # if len(str(gov_data.cell_value(1, i + 1))) == 1:    #Min
                #     tempid = tempid + ' ' 
                self.gov_tgov1_id = np.append(self.gov_tgov1_id, tempid)
                idx1 = np.where(pfd.gen_bus == self.gov_tgov1_bus[-1])[0]
                idx2 = np.where(pfd.gen_id[idx1] == tempid)[0][0]
                self.gov_tgov1_idx = np.append(self.gov_tgov1_idx, int(idx1[idx2]))
                self.gov_type[int(idx1[idx2])] = DyData.gov_model_map[typei]
                self.gov_tgov1_R = np.append(self.gov_tgov1_R, float(gov_data.cell_value(3, i + 1)))
                self.gov_tgov1_T1 = np.append(self.gov_tgov1_T1, float(gov_data.cell_value(4, i + 1)))
                self.gov_tgov1_Vmax = np.append(self.gov_tgov1_Vmax, float(gov_data.cell_value(5, i + 1)))
                self.gov_tgov1_Vmin = np.append(self.gov_tgov1_Vmin, float(gov_data.cell_value(6, i + 1)))
                self.gov_tgov1_T2 = np.append(self.gov_tgov1_T2, float(gov_data.cell_value(7, i + 1)))
                self.gov_tgov1_T3 = np.append(self.gov_tgov1_T3, float(gov_data.cell_value(8, i + 1)))
                self.gov_tgov1_Dt = np.append(self.gov_tgov1_Dt, float(gov_data.cell_value(9, i + 1)))

            if typei == 'HYGOV':
                flag = 1
                self.gov_hygov_n = self.gov_hygov_n + 1
                self.gov_hygov_bus = np.append(self.gov_hygov_bus, int(gov_data.cell_value(0, i + 1)))
                tempid = str(gov_data.cell_value(1, i + 1))
                # if len(str(gov_data.cell_value(1, i + 1))) == 1:  # Min
                #     tempid = tempid + ' '
                self.gov_hygov_id = np.append(self.gov_hygov_id, tempid)
                idx1 = np.where(pfd.gen_bus == self.gov_hygov_bus[-1])[0]
                idx2 = np.where(pfd.gen_id[idx1] == tempid)[0][0]
                self.gov_hygov_idx = np.append(self.gov_hygov_idx, int(idx1[idx2]))
                self.gov_type[int(idx1[idx2])] = DyData.gov_model_map[typei]
                self.gov_hygov_R = np.append(self.gov_hygov_R, float(gov_data.cell_value(3, i + 1)))
                self.gov_hygov_r = np.append(self.gov_hygov_r, float(gov_data.cell_value(4, i + 1)))
                self.gov_hygov_Tr = np.append(self.gov_hygov_Tr, float(gov_data.cell_value(5, i + 1)))
                self.gov_hygov_Tf = np.append(self.gov_hygov_Tf, float(gov_data.cell_value(6, i + 1)))
                self.gov_hygov_Tg = np.append(self.gov_hygov_Tg, float(gov_data.cell_value(7, i + 1)))
                self.gov_hygov_VELM = np.append(self.gov_hygov_VELM, float(gov_data.cell_value(8, i + 1)))
                self.gov_hygov_GMAX = np.append(self.gov_hygov_GMAX, float(gov_data.cell_value(9, i + 1)))
                self.gov_hygov_GMIN = np.append(self.gov_hygov_GMIN, float(gov_data.cell_value(10, i + 1)))
                self.gov_hygov_TW = np.append(self.gov_hygov_TW, float(gov_data.cell_value(11, i + 1)))
                self.gov_hygov_At = np.append(self.gov_hygov_At, float(gov_data.cell_value(12, i + 1)))
                self.gov_hygov_Dturb = np.append(self.gov_hygov_Dturb, float(gov_data.cell_value(13, i + 1)))
                self.gov_hygov_qNL = np.append(self.gov_hygov_qNL, float(gov_data.cell_value(14, i + 1)))

            if typei == 'GAST':
                flag = 1
                self.gov_gast_n = self.gov_gast_n + 1
                self.gov_gast_bus = np.append(self.gov_gast_bus, int(gov_data.cell_value(0, i + 1)))
                tempid = str(gov_data.cell_value(1, i + 1))
                tempid = tempid.replace("'"," ")
                if len(tempid)==3:    
                    if (tempid[1] == '.') and (tempid[2]=='0'):
                        tempid = tempid[0]
                # if len(tempid) == 1:    #Min
                #     tempid = tempid + " "
                self.gov_gast_id = np.append(self.gov_gast_id, tempid)
                idx1 = np.where(pfd.gen_bus == self.gov_gast_bus[-1])[0]
                idx2 = np.where(pfd.gen_id[idx1] == tempid)[0][0]
                self.gov_gast_idx = np.append(self.gov_gast_idx, int(idx1[idx2]))
                self.gov_type[int(idx1[idx2])] = DyData.gov_model_map[typei]
                self.gov_gast_R = np.append(self.gov_gast_R, float(gov_data.cell_value(3, i + 1)))
                self.gov_gast_T1 = np.append(self.gov_gast_T1, float(gov_data.cell_value(4, i + 1)))
                self.gov_gast_T2 = np.append(self.gov_gast_T2, float(gov_data.cell_value(5, i + 1)))
                self.gov_gast_T3 = np.append(self.gov_gast_T3, float(gov_data.cell_value(6, i + 1)))
                self.gov_gast_LdLmt = np.append(self.gov_gast_LdLmt, float(gov_data.cell_value(7, i + 1)))
                self.gov_gast_KT = np.append(self.gov_gast_KT, float(gov_data.cell_value(8, i + 1)))
                self.gov_gast_VMAX = np.append(self.gov_gast_VMAX, float(gov_data.cell_value(9, i + 1)))
                self.gov_gast_VMIN = np.append(self.gov_gast_VMIN, float(gov_data.cell_value(10, i + 1)))
                self.gov_gast_Dturb = np.append(self.gov_gast_Dturb, float(gov_data.cell_value(11, i + 1)))

            if flag == 0:
                print('ERROR: Governor model not supported:')
                print(typei)
                print('\n')

        # pss
        pss_data = dyn_data.sheet_by_index(3)
        self.pss_n = pss_data.ncols - 1
        for i in range(self.pss_n):
            flag = 0
            typei = str(pss_data.cell_value(2, i + 1))
            self.pss_type = np.append(self.pss_type, typei)

            if typei == 'IEEEST':
                flag = 1
                self.pss_ieeest_n = self.pss_ieeest_n + 1
                self.pss_ieeest_bus = np.append(self.pss_ieeest_bus, float(pss_data.cell_value(0, i + 1)))
                tempid = str(pss_data.cell_value(1, i + 1))
                tempid = tempid.replace("'", " ")
                # if len(tempid) == 1:
                #     tempid = tempid + " "
                self.pss_ieeest_id = np.append(self.pss_ieeest_id, tempid)
                idx1 = np.where(pfd.gen_bus == self.pss_ieeest_bus[i])[0]
                idx2 = np.where(pfd.gen_id[idx1] == self.pss_ieeest_id[i])[0][0]
                self.pss_ieeest_idx = np.append(self.pss_ieeest_idx, idx1[idx2])
                self.pss_ieeest_A1 = np.append(self.pss_ieeest_A1, float(pss_data.cell_value(3, i + 1)))
                self.pss_ieeest_A2 = np.append(self.pss_ieeest_A2, float(pss_data.cell_value(4, i + 1)))
                self.pss_ieeest_A3 = np.append(self.pss_ieeest_A3, float(pss_data.cell_value(5, i + 1)))
                self.pss_ieeest_A4 = np.append(self.pss_ieeest_A4, float(pss_data.cell_value(6, i + 1)))
                self.pss_ieeest_A5 = np.append(self.pss_ieeest_A5, float(pss_data.cell_value(7, i + 1)))
                self.pss_ieeest_A6 = np.append(self.pss_ieeest_A6, float(pss_data.cell_value(8, i + 1)))
                self.pss_ieeest_T1 = np.append(self.pss_ieeest_T1, float(pss_data.cell_value(9, i + 1)))
                self.pss_ieeest_T2 = np.append(self.pss_ieeest_T2, float(pss_data.cell_value(10, i + 1)))
                self.pss_ieeest_T3 = np.append(self.pss_ieeest_T3, float(pss_data.cell_value(11, i + 1)))
                self.pss_ieeest_T4 = np.append(self.pss_ieeest_T4, float(pss_data.cell_value(12, i + 1)))
                self.pss_ieeest_T5 = np.append(self.pss_ieeest_T5, float(pss_data.cell_value(13, i + 1)))
                self.pss_ieeest_T6 = np.append(self.pss_ieeest_T6, float(pss_data.cell_value(14, i + 1)))
                self.pss_ieeest_KS = np.append(self.pss_ieeest_KS, float(pss_data.cell_value(15, i + 1)))
                self.pss_ieeest_LSMAX = np.append(self.pss_ieeest_LSMAX, float(pss_data.cell_value(16, i + 1)))
                self.pss_ieeest_LSMIN = np.append(self.pss_ieeest_LSMIN, float(pss_data.cell_value(17, i + 1)))
                self.pss_ieeest_VCU = np.append(self.pss_ieeest_VCU, float(pss_data.cell_value(18, i + 1)))
                self.pss_ieeest_VCL = np.append(self.pss_ieeest_VCL, float(pss_data.cell_value(19, i + 1)))

        # ibr wecc
        regca_data = dyn_data.sheet_by_index(4)
        self.ibr_wecc_n = regca_data.ncols - 1
        for i in range(self.ibr_wecc_n):
            self.ibr_regca_bus = np.append(self.ibr_regca_bus, float(regca_data.cell_value(0, i + 1)))
            # self.ibr_regca_id = np.append(self.ibr_regca_id, regca_data.cell_value(1, i + 1))
            tempid = str(regca_data.cell_value(1, i + 1))
            tempid = tempid.replace("'", " ")
            # if len(tempid) == 1:
            #     tempid = tempid + " "
            self.ibr_regca_id = np.append(self.ibr_regca_id, tempid)
            idx1 = np.where(pfd.ibr_bus == self.ibr_regca_bus[i])[0]
            idx2 = np.where(pfd.ibr_id[idx1] == self.ibr_regca_id[i])[0][0]
            self.ibr_wecc_idx = np.append(self.ibr_wecc_idx, idx1[idx2])
            self.ibr_regca_LVPLsw = np.append(self.ibr_regca_LVPLsw, float(regca_data.cell_value(2, i + 1)))
            self.ibr_regca_Tg = np.append(self.ibr_regca_Tg, float(regca_data.cell_value(3, i + 1)))
            self.ibr_regca_Rrpwr = np.append(self.ibr_regca_Rrpwr, float(regca_data.cell_value(4, i + 1)))
            self.ibr_regca_Brkpt = np.append(self.ibr_regca_Brkpt, float(regca_data.cell_value(5, i + 1)))
            self.ibr_regca_Zerox = np.append(self.ibr_regca_Zerox, float(regca_data.cell_value(6, i + 1)))
            self.ibr_regca_Lvpl1 = np.append(self.ibr_regca_Lvpl1, float(regca_data.cell_value(7, i + 1)))
            self.ibr_regca_Volim = np.append(self.ibr_regca_Volim, float(regca_data.cell_value(8, i + 1)))
            self.ibr_regca_Lvpnt1 = np.append(self.ibr_regca_Lvpnt1, float(regca_data.cell_value(9, i + 1)))
            self.ibr_regca_Lvpnt0 = np.append(self.ibr_regca_Lvpnt0, float(regca_data.cell_value(10, i + 1)))
            self.ibr_regca_Iolim = np.append(self.ibr_regca_Iolim, float(regca_data.cell_value(11, i + 1)))
            self.ibr_regca_Tfltr = np.append(self.ibr_regca_Tfltr, float(regca_data.cell_value(12, i + 1)))
            self.ibr_regca_Khv = np.append(self.ibr_regca_Khv, float(regca_data.cell_value(13, i + 1)))
            self.ibr_regca_Iqrmax = np.append(self.ibr_regca_Iqrmax, float(regca_data.cell_value(14, i + 1)))
            self.ibr_regca_Iqrmin = np.append(self.ibr_regca_Iqrmin, float(regca_data.cell_value(15, i + 1)))
            self.ibr_regca_Accel = np.append(self.ibr_regca_Accel, float(regca_data.cell_value(16, i + 1)))
            self.ibr_fbase = np.append(self.ibr_fbase, float(regca_data.cell_value(17, i + 1)))
            self.ibr_MVAbase = np.append(self.ibr_MVAbase, float(regca_data.cell_value(18, i + 1))) # need to maintain consistence in MVAbase between pfd and dyd

            ibrbus = pfd.ibr_bus[i]
            ibrbus_idx = np.where(pfd.bus_num == ibrbus)
            self.ibr_kVbase = np.append(self.ibr_kVbase, pfd.bus_basekV[ibrbus_idx])

        reecb_data = dyn_data.sheet_by_index(5)
        for i in range(reecb_data.ncols - 1):
            self.ibr_reecb_bus = np.append(self.ibr_reecb_bus, float(reecb_data.cell_value(0, i + 1)))
            tempid = str(reecb_data.cell_value(1, i + 1))
            tempid = tempid.replace("'", " ")
            # if len(tempid) == 1:
            #     tempid = tempid + " "
            self.ibr_reecb_id = np.append(self.ibr_reecb_id, tempid)
            # self.ibr_reecb_id = np.append(self.ibr_reecb_id, reecb_data.cell_value(1, i + 1))
            self.ibr_reecb_PFFLAG = np.append(self.ibr_reecb_PFFLAG, float(reecb_data.cell_value(2, i + 1)))
            self.ibr_reecb_VFLAG = np.append(self.ibr_reecb_VFLAG, float(reecb_data.cell_value(3, i + 1)))
            self.ibr_reecb_QFLAG = np.append(self.ibr_reecb_QFLAG, float(reecb_data.cell_value(4, i + 1)))
            self.ibr_reecb_PQFLAG = np.append(self.ibr_reecb_PQFLAG, float(reecb_data.cell_value(5, i + 1)))
            self.ibr_reecb_Vdip = np.append(self.ibr_reecb_Vdip, float(reecb_data.cell_value(6, i + 1)))
            self.ibr_reecb_Vup = np.append(self.ibr_reecb_Vup, float(reecb_data.cell_value(7, i + 1)))
            self.ibr_reecb_Trv = np.append(self.ibr_reecb_Trv, float(reecb_data.cell_value(8, i + 1)))
            self.ibr_reecb_dbd1 = np.append(self.ibr_reecb_dbd1, float(reecb_data.cell_value(9, i + 1)))
            self.ibr_reecb_dbd2 = np.append(self.ibr_reecb_dbd2, float(reecb_data.cell_value(10, i + 1)))
            self.ibr_reecb_Kqv = np.append(self.ibr_reecb_Kqv, float(reecb_data.cell_value(11, i + 1)))
            self.ibr_reecb_Iqhl = np.append(self.ibr_reecb_Iqhl, float(reecb_data.cell_value(12, i + 1)))
            self.ibr_reecb_Iqll = np.append(self.ibr_reecb_Iqll, float(reecb_data.cell_value(13, i + 1)))
            self.ibr_reecb_Vref0 = np.append(self.ibr_reecb_Vref0, float(reecb_data.cell_value(14, i + 1)))
            self.ibr_reecb_Tp = np.append(self.ibr_reecb_Tp, float(reecb_data.cell_value(15, i + 1)))
            self.ibr_reecb_Qmax = np.append(self.ibr_reecb_Qmax, float(reecb_data.cell_value(16, i + 1)))
            self.ibr_reecb_Qmin = np.append(self.ibr_reecb_Qmin, float(reecb_data.cell_value(17, i + 1)))
            self.ibr_reecb_Vmax = np.append(self.ibr_reecb_Vmax, float(reecb_data.cell_value(18, i + 1)))
            self.ibr_reecb_Vmin = np.append(self.ibr_reecb_Vmin, float(reecb_data.cell_value(19, i + 1)))
            self.ibr_reecb_Kqp = np.append(self.ibr_reecb_Kqp, float(reecb_data.cell_value(20, i + 1)))
            self.ibr_reecb_Kqi = np.append(self.ibr_reecb_Kqi, float(reecb_data.cell_value(21, i + 1)))
            self.ibr_reecb_Kvp = np.append(self.ibr_reecb_Kvp, float(reecb_data.cell_value(22, i + 1)))
            self.ibr_reecb_Kvi = np.append(self.ibr_reecb_Kvi, float(reecb_data.cell_value(23, i + 1)))
            self.ibr_reecb_Tiq = np.append(self.ibr_reecb_Tiq, float(reecb_data.cell_value(24, i + 1)))
            self.ibr_reecb_dPmax = np.append(self.ibr_reecb_dPmax, float(reecb_data.cell_value(25, i + 1)))
            self.ibr_reecb_dPmin = np.append(self.ibr_reecb_dPmin, float(reecb_data.cell_value(26, i + 1)))
            self.ibr_reecb_Pmax = np.append(self.ibr_reecb_Pmax, float(reecb_data.cell_value(27, i + 1)))
            self.ibr_reecb_Pmin = np.append(self.ibr_reecb_Pmin, float(reecb_data.cell_value(28, i + 1)))
            self.ibr_reecb_Imax = np.append(self.ibr_reecb_Imax, float(reecb_data.cell_value(29, i + 1)))
            self.ibr_reecb_Tpord = np.append(self.ibr_reecb_Tpord, float(reecb_data.cell_value(30, i + 1)))

        repca_data = dyn_data.sheet_by_index(6)
        for i in range(repca_data.ncols - 1):
            self.ibr_repca_bus = np.append(self.ibr_repca_bus, float(repca_data.cell_value(0, i + 1)))
            tempid = str(repca_data.cell_value(1, i + 1))
            tempid = tempid.replace("'", " ")
            # if len(tempid) == 1:
            #     tempid = tempid + " "
            self.ibr_repca_id = np.append(self.ibr_repca_id, tempid)
            # self.ibr_repca_id = np.append(self.ibr_repca_id, repca_data.cell_value(1, i + 1))
            self.ibr_repca_remote_bus = np.append(self.ibr_repca_remote_bus, float(repca_data.cell_value(2, i + 1)))
            self.ibr_repca_branch_From_bus = np.append(self.ibr_repca_branch_From_bus,
                                                       float(repca_data.cell_value(3, i + 1)))
            self.ibr_repca_branch_To_bus = np.append(self.ibr_repca_branch_To_bus,
                                                     float(repca_data.cell_value(4, i + 1)))
            self.ibr_repca_branch_id = np.append(self.ibr_repca_branch_id, repca_data.cell_value(5, i + 1))
            self.ibr_repca_VCFlag = np.append(self.ibr_repca_VCFlag, float(repca_data.cell_value(6, i + 1)))
            self.ibr_repca_RefFlag = np.append(self.ibr_repca_RefFlag, float(repca_data.cell_value(7, i + 1)))
            self.ibr_repca_FFlag = np.append(self.ibr_repca_FFlag, float(repca_data.cell_value(8, i + 1)))
            self.ibr_repca_Tfltr = np.append(self.ibr_repca_Tfltr, float(repca_data.cell_value(9, i + 1)))
            self.ibr_repca_Kp = np.append(self.ibr_repca_Kp, float(repca_data.cell_value(10, i + 1)))
            self.ibr_repca_Ki = np.append(self.ibr_repca_Ki, float(repca_data.cell_value(11, i + 1)))
            self.ibr_repca_Tft = np.append(self.ibr_repca_Tft, float(repca_data.cell_value(12, i + 1)))
            self.ibr_repca_Tfv = np.append(self.ibr_repca_Tfv, float(repca_data.cell_value(13, i + 1)))
            self.ibr_repca_Vfrz = np.append(self.ibr_repca_Vfrz, float(repca_data.cell_value(14, i + 1)))
            self.ibr_repca_Rc = np.append(self.ibr_repca_Rc, float(repca_data.cell_value(15, i + 1)))
            self.ibr_repca_Xc = np.append(self.ibr_repca_Xc, float(repca_data.cell_value(16, i + 1)))
            self.ibr_repca_Kc = np.append(self.ibr_repca_Kc, float(repca_data.cell_value(17, i + 1)))
            self.ibr_repca_emax = np.append(self.ibr_repca_emax, float(repca_data.cell_value(18, i + 1)))
            self.ibr_repca_emin = np.append(self.ibr_repca_emin, float(repca_data.cell_value(19, i + 1)))
            self.ibr_repca_dbd1 = np.append(self.ibr_repca_dbd1, float(repca_data.cell_value(20, i + 1)))
            self.ibr_repca_dbd2 = np.append(self.ibr_repca_dbd2, float(repca_data.cell_value(21, i + 1)))
            self.ibr_repca_Qmax = np.append(self.ibr_repca_Qmax, float(repca_data.cell_value(22, i + 1)))
            self.ibr_repca_Qmin = np.append(self.ibr_repca_Qmin, float(repca_data.cell_value(23, i + 1)))
            self.ibr_repca_Kpg = np.append(self.ibr_repca_Kpg, float(repca_data.cell_value(24, i + 1)))
            self.ibr_repca_Kig = np.append(self.ibr_repca_Kig, float(repca_data.cell_value(25, i + 1)))
            self.ibr_repca_Tp = np.append(self.ibr_repca_Tp, float(repca_data.cell_value(26, i + 1)))
            self.ibr_repca_fdbd1 = np.append(self.ibr_repca_fdbd1, float(repca_data.cell_value(27, i + 1)))
            self.ibr_repca_fdbd2 = np.append(self.ibr_repca_fdbd2, float(repca_data.cell_value(28, i + 1)))
            self.ibr_repca_femax = np.append(self.ibr_repca_femax, float(repca_data.cell_value(29, i + 1)))
            self.ibr_repca_femin = np.append(self.ibr_repca_femin, float(repca_data.cell_value(30, i + 1)))
            self.ibr_repca_Pmax = np.append(self.ibr_repca_Pmax, float(repca_data.cell_value(31, i + 1)))
            self.ibr_repca_Pmin = np.append(self.ibr_repca_Pmin, float(repca_data.cell_value(32, i + 1)))
            self.ibr_repca_Tg = np.append(self.ibr_repca_Tg, float(repca_data.cell_value(33, i + 1)))
            self.ibr_repca_Ddn = np.append(self.ibr_repca_Ddn, float(repca_data.cell_value(34, i + 1)))
            self.ibr_repca_Dup = np.append(self.ibr_repca_Dup, float(repca_data.cell_value(35, i + 1)))

        # PLL for bus freq/angle measurement
        pll_data = dyn_data.sheet_by_index(7)
        for i in range(pll_data.ncols - 1):
            self.pll_bus = np.append(self.pll_bus, float(pll_data.cell_value(0, i + 1)))
            self.pll_ke = np.append(self.pll_ke, float(pll_data.cell_value(1, i + 1)))
            self.pll_te = np.append(self.pll_te, float(pll_data.cell_value(2, i + 1)))

        # volt mag measurement
        vm_data = dyn_data.sheet_by_index(8)
        for i in range(vm_data.ncols - 1):
            self.vm_bus = np.append(self.vm_bus, float(vm_data.cell_value(0, i + 1)))
            self.vm_te = np.append(self.vm_te, float(vm_data.cell_value(1, i + 1)))

        # measurement methods
        mea_data = dyn_data.sheet_by_index(9)
        for i in range(mea_data.ncols - 1):
            self.mea_bus = np.append(self.mea_bus, float(mea_data.cell_value(0, i + 1)))
            self.mea_method = np.append(self.mea_method, float(mea_data.cell_value(1, i + 1)))
        

        # IBR EPRI
        ibr_epri_data = dyn_data.sheet_by_index(10)
        self.ibr_epri_n = ibr_epri_data.ncols - 1
        for i in range(ibr_epri_data.ncols - 1):
            self.ibr_epri_bus = np.append(self.ibr_epri_bus, float(ibr_epri_data.cell_value(0, i + 1)))
            tempid = str(ibr_epri_data.cell_value(1, i + 1))
            tempid = tempid.replace("'", " ")
            # if len(tempid) == 1:
            #     tempid = tempid + " "
            self.ibr_epri_id = np.append(self.ibr_epri_id, tempid)
            # self.ibr_epri_id = np.append(self.ibr_epri_id, ibr_epri_data.cell_value(1, i + 1))
            idx1 = np.where(pfd.ibr_bus == self.ibr_epri_bus[i])[0]
            idx2 = np.where(pfd.ibr_id[idx1] == self.ibr_epri_id[i])[0][0]
            self.ibr_epri_idx = np.append(self.ibr_epri_idx, idx1[idx2])
            self.ibr_epri_Vdcbase = np.append(self.ibr_epri_Vdcbase, float(ibr_epri_data.cell_value(2, i + 1)))
            self.ibr_epri_KpI = np.append(self.ibr_epri_KpI, float(ibr_epri_data.cell_value(3, i + 1)))
            self.ibr_epri_KiI = np.append(self.ibr_epri_KiI, float(ibr_epri_data.cell_value(4, i + 1)))
            self.ibr_epri_KpPLL = np.append(self.ibr_epri_KpPLL, float(ibr_epri_data.cell_value(5, i + 1)))
            self.ibr_epri_KiPLL = np.append(self.ibr_epri_KiPLL, float(ibr_epri_data.cell_value(6, i + 1)))
            self.ibr_epri_KpP = np.append(self.ibr_epri_KpP, float(ibr_epri_data.cell_value(7, i + 1)))
            self.ibr_epri_KiP = np.append(self.ibr_epri_KiP, float(ibr_epri_data.cell_value(8, i + 1)))
            self.ibr_epri_KpQ = np.append(self.ibr_epri_KpQ, float(ibr_epri_data.cell_value(9, i + 1)))
            self.ibr_epri_KiQ = np.append(self.ibr_epri_KiQ, float(ibr_epri_data.cell_value(10, i + 1)))
            self.ibr_epri_Imax = np.append(self.ibr_epri_Imax, float(ibr_epri_data.cell_value(11, i + 1)))
            self.ibr_epri_Pqflag = np.append(self.ibr_epri_Pqflag, float(ibr_epri_data.cell_value(12, i + 1)))
            self.ibr_epri_Vdip = np.append(self.ibr_epri_Vdip, float(ibr_epri_data.cell_value(13, i + 1)))
            self.ibr_epri_Vup = np.append(self.ibr_epri_Vup, float(ibr_epri_data.cell_value(14, i + 1)))
            self.ibr_epri_Rchoke = np.append(self.ibr_epri_Rchoke, float(ibr_epri_data.cell_value(15, i + 1)))
            self.ibr_epri_Lchoke = np.append(self.ibr_epri_Lchoke, float(ibr_epri_data.cell_value(16, i + 1)))
            self.ibr_epri_Cfilt = np.append(self.ibr_epri_Cfilt, float(ibr_epri_data.cell_value(17, i + 1)))
            self.ibr_epri_Rdamp = np.append(self.ibr_epri_Rdamp, float(ibr_epri_data.cell_value(18, i + 1)))

            if abs(self.ibr_epri_Vdcbase[-1])<1e-5:
                bus_idx = np.where(pfd.bus_num == self.ibr_epri_bus[-1])[0][0]
                self.ibr_epri_Vdcbase[-1] = pfd.bus_basekV[bus_idx]*2.5

            ibr_idx = np.where(pfd.ibr_bus == self.ibr_epri_bus[i])[0][0]
            self.ibr_epri_basemva = np.append(self.ibr_epri_basemva, pfd.ibr_MVA_base[ibr_idx])

            bus_idx = np.where(pfd.bus_num == self.ibr_epri_bus[i])[0][0]
            self.ibr_epri_basekV = np.append(self.ibr_epri_basekV, pfd.bus_basekV[bus_idx])
        

        self.ibr_n = self.ibr_wecc_n + self.ibr_epri_n

        self.nx_ttl = self.nx_ttl + self.gen_genrou_odr * self.gen_genrou_n
        self.nx_ttl = self.nx_ttl + self.exc_sexs_odr * self.exc_sexs_n
        self.nx_ttl = self.nx_ttl + self.gov_tgov1_odr * self.gov_tgov1_n
        self.nx_ttl = self.nx_ttl + self.gov_hygov_odr * self.gov_hygov_n
        self.nx_ttl = self.nx_ttl + self.gov_gast_odr * self.gov_gast_n
        self.nx_ttl = self.nx_ttl + self.pss_ieeest_odr * self.pss_ieeest_n
        self.nibr_ttl = self.nibr_ttl + self.ibr_wecc_odr * self.ibr_wecc_n



    def spreaddyd(self, pfd, dyd, N):
        dyd_dict = dyd.__dict__
        my_dyd = DyData()
        for x in dyd_dict.keys():
            newx = []
            if type(dyd_dict[x]) != np.ndarray:
                if isinstance(dyd_dict[x], int):
                    if x[-2:] == '_n':
                        newx = dyd_dict[x] * N
                        setattr(my_dyd, x, newx)
                    elif (x[-4:] == '_odr') or (x[-4:] == '_ttl'):
                        newx = dyd_dict[x]
                        setattr(my_dyd, x, newx)
                    elif x[-6:] == '_xi_st':
                        newx = 0    # to be defined in CombineX
                        setattr(my_dyd, x, newx) 
                    else:
                        pass
                else:
                    newx = dyd_dict[x] * N
                    setattr(my_dyd, x, newx)
                    # print('Warning: should not see this when spreading dyn data!!')
            else:
                if ((x == 'exe_sexs_bus') or
                    (x == 'gen_genrou_bus') or
                    (x == 'gov_gast_bus') or
                    (x == 'gov_hygov_bus') or
                    (x == 'gov_tgov1_bus') or
                    (x == 'pss_ieeest_bus') or
                    (x == 'pll_bus') or
                    (x == 'vm_bus')
                    ):

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * (max(pfd.bus_num)+1) / N
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'gen_genrou_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'exc_sexs_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'gov_gast_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'gov_hygov_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'gov_tgov1_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                elif x == 'pss_ieeest_idx':

                    for i in range(N):
                        tempx = dyd_dict[x]
                        tempnewx = dyd_dict[x] + i * dyd.gen_n
                        newx = np.concatenate((newx, tempnewx))

                else:

                    for i in range(N):
                        newx = np.concatenate((newx, dyd_dict[x]))

                setattr(my_dyd, x, newx)

        return my_dyd


    def ToEquiCirData(self, pfd, dyd):
        # IBR base calc
        for i in range(len(pfd.ibr_bus)):
            ibrbus = pfd.ibr_bus[i]
            ibrbus_idx = np.where(pfd.bus_num == ibrbus)[0][0]

            base_es_temp = pfd.bus_basekV[ibrbus_idx] * math.sqrt(2.0 / 3.0) * 1000.0
            base_is_temp = pfd.ibr_MVA_base[i] * 1000000.0 / (base_es_temp * 3.0 / 2.0)
            self.ibr_Ibase = np.append(self.ibr_Ibase, base_is_temp / math.sqrt(2.0))

        # Convert generator data to equivalent circuit data

        # base calculation
        for i in range(len(pfd.gen_bus)):
            genbus = pfd.gen_bus[i]
            genbus_idx = np.where(pfd.bus_num == genbus)[0][0]

            base_es_temp = pfd.bus_basekV[genbus_idx] * math.sqrt(2.0 / 3.0) * 1000.0
            self.base_es = np.append(self.base_es, base_es_temp)
            base_is_temp = pfd.gen_MVA_base[i] * 1000000.0 / (base_es_temp * 3.0 / 2.0)
            self.base_is = np.append(self.base_is, base_is_temp)
            base_Is_temp = base_is_temp / math.sqrt(2.0)
            self.base_Is = np.append(self.base_Is, base_Is_temp)
            base_Zs_temp = base_es_temp / base_is_temp
            self.base_Zs = np.append(self.base_Zs, base_Zs_temp)
            base_Ls_temp = base_Zs_temp * 1000.0 / 2.0 / 60.0 / math.pi
            self.base_Ls = np.append(self.base_Ls, base_Ls_temp)

            self.ec_Lad = np.append(self.ec_Lad, self.gen_genrou_Xd[i] - self.gen_genrou_Xl[i])
            self.ec_Laq = np.append(self.ec_Laq, self.gen_genrou_Xq[i] - self.gen_genrou_Xl[i])
            self.ec_Ll = np.append(self.ec_Ll, self.gen_genrou_Xl[i])
            self.ec_Ld = np.append(self.ec_Ld, self.ec_Lad[i] + self.ec_Ll[i])
            self.ec_Lq = np.append(self.ec_Lq, self.ec_Laq[i] + self.ec_Ll[i])
            self.ec_Lfd = np.append(self.ec_Lfd,
                                    (self.gen_genrou_Xdp[i] - self.gen_genrou_Xl[i]) * self.ec_Lad[i] / (
                                            self.ec_Lad[i] - (self.gen_genrou_Xdp[i] - self.gen_genrou_Xl[i])))
            self.ec_L1q = np.append(self.ec_L1q,
                                    (self.gen_genrou_Xqp[i] - self.gen_genrou_Xl[i]) * self.ec_Laq[i] / (
                                            self.ec_Laq[i] - (self.gen_genrou_Xqp[i] - self.gen_genrou_Xl[i])))

            z = self.gen_genrou_Xdpp[i] - self.gen_genrou_Xl[i]
            y = self.ec_Lad[i] * self.ec_Lfd[i] / (self.ec_Lad[i] + self.ec_Lfd[i])
            self.ec_L1d = np.append(self.ec_L1d, y * z / (y - z))
            self.ec_R1d = np.append(self.ec_R1d, (y + self.ec_L1d[i]) / self.gen_genrou_Td0pp[i] / pfd.ws) 

            z = self.gen_genrou_Xdpp[i] - self.gen_genrou_Xl[i]
            y = self.ec_Laq[i] * self.ec_L1q[i] / (self.ec_Laq[i] + self.ec_L1q[i])
            self.ec_L2q = np.append(self.ec_L2q, y * z / (y - z))
            self.ec_R2q = np.append(self.ec_R2q, (y + self.ec_L2q[i]) / self.gen_genrou_Tq0pp[i]/ pfd.ws ) 

            self.ec_Rfd = np.append(self.ec_Rfd, (self.ec_Lad[i] + self.ec_Lfd[i]) / self.gen_genrou_Td0p[i] / pfd.ws) 
            self.ec_R1q = np.append(self.ec_R1q, (self.ec_Laq[i] + self.ec_L1q[i]) / self.gen_genrou_Tq0p[i] / pfd.ws) 

            self.ec_Ra = np.append(self.ec_Ra, self.gen_Ra[i])
            self.ec_L0 = np.append(self.ec_L0, self.gen_X0[i])
            self.ec_Lf1d = np.append(self.ec_Lf1d, self.ec_Lad[i])

            self.ec_Lffd = np.append(self.ec_Lffd,
                                     self.ec_Lad[i] * self.ec_Lad[i] / (
                                             self.gen_genrou_Xd[i] - self.gen_genrou_Xdp[i]))
            self.ec_L11d = np.append(self.ec_L11d,
                                     self.ec_Lad[i] * self.ec_Lad[i] / (
                                             self.gen_genrou_Xd[i] - self.gen_genrou_Xdpp[i]))
            self.ec_L11q = np.append(self.ec_L11q,
                                     self.ec_Laq[i] * self.ec_Laq[i] / (
                                             self.gen_genrou_Xq[i] - self.gen_genrou_Xqp[i]))
            self.ec_L22q = np.append(self.ec_L22q,
                                     self.ec_Laq[i] * self.ec_Laq[i] / (
                                             self.gen_genrou_Xq[i] - self.gen_genrou_Xdpp[i]))


    def extract_emt_area(self, del_gen_idx, del_ibr_idx, emt_buses, pfd):
        for i in del_gen_idx:
            self.base_Is = np.delete(self.base_Is, i)
            # self.base_Lfd = np.delete(self.base_Lfd, i)
            self.base_Ls = np.delete(self.base_Ls, i)
            # self.base_Zfd = np.delete(self.base_Zfd, i)
            self.base_Zs = np.delete(self.base_Zs, i)
            # self.base_efd = np.delete(self.base_efd, i)
            self.base_es = np.delete(self.base_es, i)
            # self.base_ifd = np.delete(self.base_ifd, i)
            self.base_is = np.delete(self.base_is, i)

            self.ec_L0 = np.delete(self.ec_L0, i)
            self.ec_L11d = np.delete(self.ec_L11d, i)
            self.ec_L11q = np.delete(self.ec_L11q, i)
            self.ec_L1d = np.delete(self.ec_L1d, i)
            self.ec_L1q = np.delete(self.ec_L1q, i)
            self.ec_L22q = np.delete(self.ec_L22q, i)
            self.ec_L2q = np.delete(self.ec_L2q, i)
            self.ec_Lad = np.delete(self.ec_Lad, i)
            self.ec_Laq = np.delete(self.ec_Laq, i)
            self.ec_Ld = np.delete(self.ec_Ld, i)
            self.ec_Lf1d = np.delete(self.ec_Lf1d, i)
            self.ec_Lfd = np.delete(self.ec_Lfd, i)
            self.ec_Lffd = np.delete(self.ec_Lffd, i)
            self.ec_Ll = np.delete(self.ec_Ll, i)
            self.ec_Lq = np.delete(self.ec_Lq, i)
            self.ec_R1d = np.delete(self.ec_R1d, i)
            self.ec_R1q = np.delete(self.ec_R1q, i)
            self.ec_R2q = np.delete(self.ec_R2q, i)
            self.ec_Ra = np.delete(self.ec_Ra, i)
            self.ec_Rfd = np.delete(self.ec_Rfd, i)

            self.exc_type = np.delete(self.exc_type, i)
            self.exc_sexs_Emax = np.delete(self.exc_sexs_Emax, i)
            self.exc_sexs_Emin = np.delete(self.exc_sexs_Emin, i)
            self.exc_sexs_K = np.delete(self.exc_sexs_K, i)
            self.exc_sexs_TA = np.delete(self.exc_sexs_TA, i)
            self.exc_sexs_TA_o_TB = np.delete(self.exc_sexs_TA_o_TB, i)
            self.exc_sexs_TB = np.delete(self.exc_sexs_TB, i)
            self.exc_sexs_TE = np.delete(self.exc_sexs_TE, i)
            self.exc_sexs_bus = np.delete(self.exc_sexs_bus, i)
            self.exc_sexs_id = np.delete(self.exc_sexs_id, i)
            for j in range(i, len(self.exc_sexs_idx), 1):
                self.exc_sexs_idx[j] = self.exc_sexs_idx[j] - 1

            self.exc_sexs_idx = np.delete(self.exc_sexs_idx, i)
            self.exc_sexs_n = self.exc_sexs_n - 1
            self.exc_n = self.exc_n - 1

            self.gen_D = np.delete(self.gen_D, i)
            self.gen_H = np.delete(self.gen_H, i)
            self.gen_Ra = np.delete(self.gen_Ra, i)
            self.gen_X0 = np.delete(self.gen_X0, i)
            self.gen_type = np.delete(self.gen_type, i)
            self.gen_n = self.gen_n - 1

            self.gen_genrou_S10 = np.delete(self.gen_genrou_S10, i)
            self.gen_genrou_S12 = np.delete(self.gen_genrou_S12, i)
            self.gen_genrou_Td0p = np.delete(self.gen_genrou_Td0p, i)
            self.gen_genrou_Td0pp = np.delete(self.gen_genrou_Td0pp, i)
            self.gen_genrou_Tq0p = np.delete(self.gen_genrou_Tq0p, i)
            self.gen_genrou_Tq0pp = np.delete(self.gen_genrou_Tq0pp, i)
            self.gen_genrou_Xd = np.delete(self.gen_genrou_Xd, i)
            self.gen_genrou_Xdp = np.delete(self.gen_genrou_Xdp, i)
            self.gen_genrou_Xdpp = np.delete(self.gen_genrou_Xdpp, i)
            self.gen_genrou_Xl = np.delete(self.gen_genrou_Xl, i)
            self.gen_genrou_Xq = np.delete(self.gen_genrou_Xq, i)
            self.gen_genrou_Xqp = np.delete(self.gen_genrou_Xqp, i)
            self.gen_genrou_bus = np.delete(self.gen_genrou_bus, i)
            self.gen_genrou_id = np.delete(self.gen_genrou_id, i)
            self.gen_genrou_idx = np.delete(self.gen_genrou_idx, i)
            self.gen_genrou_n = self.gen_genrou_n - 1
            
            self.gov_type = np.delete(self.gov_type, i)
            if self.gov_tgov1_n>0:
                if i in self.gov_tgov1_idx:
                    j = np.where(self.gov_tgov1_idx==i)[0][0]
                    self.gov_tgov1_Dt = np.delete(self.gov_tgov1_Dt, j)
                    self.gov_tgov1_R = np.delete(self.gov_tgov1_R, j)
                    self.gov_tgov1_T1 = np.delete(self.gov_tgov1_T1, j)
                    self.gov_tgov1_T2 = np.delete(self.gov_tgov1_T2, j)
                    self.gov_tgov1_T3 = np.delete(self.gov_tgov1_T3, j)
                    self.gov_tgov1_Vmax = np.delete(self.gov_tgov1_Vmax, j)
                    self.gov_tgov1_Vmin = np.delete(self.gov_tgov1_Vmin, j)
                    self.gov_tgov1_bus = np.delete(self.gov_tgov1_bus, j)
                    self.gov_tgov1_id = np.delete(self.gov_tgov1_id, j)
                    # for k in range(j, len(self.gov_tgov1_idx), 1):
                    #     self.gov_tgov1_idx[k] = self.gov_tgov1_idx[k] - 1
                    # self.gov_tgov1_idx = np.delete(self.gov_tgov1_idx, j)
                    self.gov_tgov1_n = self.gov_tgov1_n - 1
                    self.gov_n = self.gov_n - 1
            
            if self.gov_hygov_n>0:
                if i in self.gov_hygov_idx:
                    j = np.where(self.gov_hygov_idx==i)[0][0]
                    self.gov_hygov_At = np.delete(self.gov_hygov_At, j)
                    self.gov_hygov_Dturb = np.delete(self.gov_hygov_Dturb, j)
                    self.gov_hygov_GMAX = np.delete(self.gov_hygov_GMAX, j)
                    self.gov_hygov_GMIN = np.delete(self.gov_hygov_GMIN, j)
                    self.gov_hygov_R = np.delete(self.gov_hygov_R, j)
                    self.gov_hygov_TW = np.delete(self.gov_hygov_TW, j)
                    self.gov_hygov_Tf = np.delete(self.gov_hygov_Tf, j)
                    self.gov_hygov_Tg = np.delete(self.gov_hygov_Tg, j)
                    self.gov_hygov_Tr = np.delete(self.gov_hygov_Tr, j)
                    self.gov_hygov_VELM = np.delete(self.gov_hygov_VELM, j)
                    self.gov_hygov_bus = np.delete(self.gov_hygov_bus, j)
                    self.gov_hygov_id = np.delete(self.gov_hygov_id, j)
                    self.gov_hygov_qNL = np.delete(self.gov_hygov_qNL, j)
                    self.gov_hygov_r = np.delete(self.gov_hygov_r, j)
                    # for k in range(j, len(self.gov_hygov_idx), 1):
                    #     self.gov_hygov_idx[k] = self.gov_hygov_idx[k] - 1
                    # self.gov_hygov_idx = np.delete(self.gov_hygov_idx, j)
                    self.gov_hygov_n = self.gov_hygov_n - 1
                    self.gov_n = self.gov_n - 1

            if self.gov_gast_n>0:
                if i in self.gov_gast_idx:
                    j = np.where(self.gov_gast_idx==i)[0][0]
                    self.gov_gast_Dturb = np.delete(self.gov_gast_Dturb, j)
                    self.gov_gast_KT = np.delete(self.gov_gast_KT, j)
                    self.gov_gast_LdLmt = np.delete(self.gov_gast_LdLmt, j)
                    self.gov_gast_R = np.delete(self.gov_gast_R, j)
                    self.gov_gast_T1 = np.delete(self.gov_gast_T1, j)
                    self.gov_gast_T2 = np.delete(self.gov_gast_T2, j)
                    self.gov_gast_T3 = np.delete(self.gov_gast_T3, j)
                    self.gov_gast_VMAX = np.delete(self.gov_gast_VMAX, j)
                    self.gov_gast_VMIN = np.delete(self.gov_gast_VMIN, j)
                    self.gov_gast_bus = np.delete(self.gov_gast_bus, j)
                    self.gov_gast_id = np.delete(self.gov_gast_id, j)
                    # for k in range(j, len(self.gov_gast_idx), 1):
                    #     self.gov_gast_idx[k] = self.gov_gast_idx[k] - 1
                    # self.gov_gast_idx = np.delete(self.gov_gast_idx, j)
                    self.gov_gast_n = self.gov_gast_n - 1
                    self.gov_n = self.gov_n - 1
            
            if self.pss_ieeest_n>0:
                if i in self.pss_ieeest_idx:
                    j = np.where(self.pss_ieeest_idx==i)[0][0]
                    self.pss_ieeest_A1 = np.delete(self.pss_ieeest_A1, j)
                    self.pss_ieeest_A2 = np.delete(self.pss_ieeest_A2, j)
                    self.pss_ieeest_A3 = np.delete(self.pss_ieeest_A3, j)
                    self.pss_ieeest_A4 = np.delete(self.pss_ieeest_A4, j)
                    self.pss_ieeest_A5 = np.delete(self.pss_ieeest_A5, j)
                    self.pss_ieeest_A6 = np.delete(self.pss_ieeest_A6, j)
                    self.pss_ieeest_KS = np.delete(self.pss_ieeest_KS, j)
                    self.pss_ieeest_LSMAX = np.delete(self.pss_ieeest_LSMAX, j)
                    self.pss_ieeest_LSMIN = np.delete(self.pss_ieeest_LSMIN, j)
                    self.pss_ieeest_T1 = np.delete(self.pss_ieeest_T1, j)
                    self.pss_ieeest_T2 = np.delete(self.pss_ieeest_T2, j)
                    self.pss_ieeest_T3 = np.delete(self.pss_ieeest_T3, j)
                    self.pss_ieeest_T4 = np.delete(self.pss_ieeest_T4, j)
                    self.pss_ieeest_T5 = np.delete(self.pss_ieeest_T5, j)
                    self.pss_ieeest_T6 = np.delete(self.pss_ieeest_T6, j)
                    self.pss_ieeest_VCL = np.delete(self.pss_ieeest_VCL, j)
                    self.pss_ieeest_VCU = np.delete(self.pss_ieeest_VCU, j)
                    self.pss_ieeest_bus = np.delete(self.pss_ieeest_bus, j)
                    self.pss_ieeest_id = np.delete(self.pss_ieeest_id, j)

                    for k in range(j, len(self.pss_ieeest_idx), 1):
                        self.pss_ieeest_idx[k] = self.pss_ieeest_idx[k] - 1
                    self.pss_ieeest_idx = np.delete(self.pss_ieeest_idx, j)
                    self.pss_ieeest_n = self.pss_ieeest_n - 1
                    self.pss_n = self.pss_n - 1


        # update gov_xx_idx
        i_tgov1 = 0
        i_hygov = 0
        i_gast = 0
        self.gov_tgov1_idx = np.array([])
        self.gov_hygov_idx = np.array([])
        self.gov_gast_idx = np.array([])
        for i in range(self.gov_n):
            flag = 0
            typei = self.gov_type[i]
            if typei == self.gov_model_map['TGOV1']:
                flag = 1
                idx1 = np.where(pfd.gen_bus == self.gov_tgov1_bus[i_tgov1])[0]
                idx2 = np.where(pfd.gen_id[idx1] == self.gov_tgov1_id[i_tgov1])[0][0]
                self.gov_tgov1_idx = np.append(self.gov_tgov1_idx, int(idx1[idx2]))
                i_tgov1 = i_tgov1 + 1

            if typei == self.gov_model_map['HYGOV']:
                flag = 1
                idx1 = np.where(pfd.gen_bus == self.gov_hygov_bus[i_hygov])[0]
                idx2 = np.where(pfd.gen_id[idx1] == self.gov_hygov_id[i_hygov])[0][0]
                self.gov_hygov_idx = np.append(self.gov_hygov_idx, int(idx1[idx2]))
                i_hygov = i_hygov + 1

            if typei == self.gov_model_map['GAST']:
                flag = 1
                idx1 = np.where(pfd.gen_bus == self.gov_gast_bus[i_gast])[0]
                idx2 = np.where(pfd.gen_id[idx1] == self.gov_gast_id[i_gast])[0][0]
                self.gov_gast_idx = np.append(self.gov_gast_idx, int(idx1[idx2]))
                i_gast = i_gast + 1
                

            if flag == 0:
                print('ERROR: Governor model not supported:')
                print(typei)
                print('\n')

            
        
        
        for i in del_ibr_idx:
            self.ibr_Ibase = np.delete(self.ibr_Ibase, i)
            self.ibr_MVAbase = np.delete(self.ibr_MVAbase, i)
            self.ibr_fbase = np.delete(self.ibr_fbase, i)
            self.ibr_kVbase = np.delete(self.ibr_kVbase, i)
            if self.ibr_epri_n>0:
                if i in self.ibr_epri_idx:
                    j = np.where(self.ibr_epri_idx==i)[0][0]
                    self.ibr_epri_Cfilt = np.delete(self.ibr_epri_Cfilt, j)
                    self.ibr_epri_Imax = np.delete(self.ibr_epri_Imax, j)
                    self.ibr_epri_KiI = np.delete(self.ibr_epri_KiI, j)
                    self.ibr_epri_KiP = np.delete(self.ibr_epri_KiP, j)
                    self.ibr_epri_KiPLL = np.delete(self.ibr_epri_KiPLL, j)
                    self.ibr_epri_KiQ = np.delete(self.ibr_epri_KiQ, j)
                    self.ibr_epri_KpI = np.delete(self.ibr_epri_KpI, j)
                    self.ibr_epri_KpP = np.delete(self.ibr_epri_KpP, j)
                    self.ibr_epri_KpPLL = np.delete(self.ibr_epri_KpPLL, j)
                    self.ibr_epri_KpQ = np.delete(self.ibr_epri_KpQ, j)
                    self.ibr_epri_Lchoke = np.delete(self.ibr_epri_Lchoke, j)
                    self.ibr_epri_Pqflag = np.delete(self.ibr_epri_Pqflag, j)
                    self.ibr_epri_Rchoke = np.delete(self.ibr_epri_Rchoke, j)
                    self.ibr_epri_Rdamp = np.delete(self.ibr_epri_Rdamp, j)
                    self.ibr_epri_Vdcbase = np.delete(self.ibr_epri_Vdcbase, j)
                    self.ibr_epri_Vdip = np.delete(self.ibr_epri_Vdip, j)
                    self.ibr_epri_Vup = np.delete(self.ibr_epri_Vup, j)
                    self.ibr_epri_basekV = np.delete(self.ibr_epri_basekV, j)
                    self.ibr_epri_basemva = np.delete(self.ibr_epri_basemva, j)
                    self.ibr_epri_bus = np.delete(self.ibr_epri_bus, j)
                    self.ibr_epri_id = np.delete(self.ibr_epri_id, j)

                    for k in range(j, len(self.ibr_epri_idx), 1):
                        self.ibr_epri_idx[k] = self.ibr_epri_idx[k] - 1
                    self.ibr_epri_idx = np.delete(self.ibr_epri_idx, j)
                    self.ibr_epri_n = self.ibr_epri_n - 1
                    self.ibr_n = self.ibr_n - 1

            if self.ibr_wecc_n>0:
                if i in self.ibr_wecc_idx:
                    j = np.where(self.ibr_wecc_idx==i)[0][0]
                    self.ibr_regca_Accel = np.delete(self.ibr_regca_Accel, j)
                    self.ibr_regca_Brkpt = np.delete(self.ibr_regca_Brkpt, j)
                    self.ibr_regca_Iolim = np.delete(self.ibr_regca_Iolim, j)
                    self.ibr_regca_Iqrmax = np.delete(self.ibr_regca_Iqrmax, j)
                    self.ibr_regca_Iqrmin = np.delete(self.ibr_regca_Iqrmin, j)
                    self.ibr_regca_Khv = np.delete(self.ibr_regca_Khv, j)
                    self.ibr_regca_LVPLsw = np.delete(self.ibr_regca_LVPLsw, j)
                    self.ibr_regca_Lvpl1 = np.delete(self.ibr_regca_Lvpl1, j)
                    self.ibr_regca_Lvpnt0 = np.delete(self.ibr_regca_Lvpnt0, j)
                    self.ibr_regca_Lvpnt1 = np.delete(self.ibr_regca_Lvpnt1, j)
                    self.ibr_regca_Rrpwr = np.delete(self.ibr_regca_Rrpwr, j)
                    self.ibr_regca_Tfltr = np.delete(self.ibr_regca_Tfltr, j)
                    self.ibr_regca_Tg = np.delete(self.ibr_regca_Tg, j)
                    self.ibr_regca_Volim = np.delete(self.ibr_regca_Volim, j)
                    self.ibr_regca_Zerox = np.delete(self.ibr_regca_Zerox, j)
                    self.ibr_regca_bus = np.delete(self.ibr_regca_bus, j)
                    self.ibr_regca_id = np.delete(self.ibr_regca_id, j)

                    self.ibr_reecb_Imax = np.delete(self.ibr_reecb_Imax, j)
                    self.ibr_reecb_Iqhl = np.delete(self.ibr_reecb_Iqhl, j)
                    self.ibr_reecb_Iqll = np.delete(self.ibr_reecb_Iqll, j)
                    self.ibr_reecb_Kqi = np.delete(self.ibr_reecb_Kqi, j)
                    self.ibr_reecb_Kqp = np.delete(self.ibr_reecb_Kqp, j)
                    self.ibr_reecb_Kqv = np.delete(self.ibr_reecb_Kqv, j)
                    self.ibr_reecb_Kvi = np.delete(self.ibr_reecb_Kvi, j)
                    self.ibr_reecb_Kvp = np.delete(self.ibr_reecb_Kvp, j)
                    self.ibr_reecb_PFFLAG = np.delete(self.ibr_reecb_PFFLAG, j)
                    self.ibr_reecb_PQFLAG = np.delete(self.ibr_reecb_PQFLAG, j)
                    self.ibr_reecb_Pmax = np.delete(self.ibr_reecb_Pmax, j)
                    self.ibr_reecb_Pmin = np.delete(self.ibr_reecb_Pmin, j)
                    self.ibr_reecb_QFLAG = np.delete(self.ibr_reecb_QFLAG, j)
                    self.ibr_reecb_Qmax = np.delete(self.ibr_reecb_Qmax, j)
                    self.ibr_reecb_Qmin = np.delete(self.ibr_reecb_Qmin, j)
                    self.ibr_reecb_Tiq = np.delete(self.ibr_reecb_Tiq, j)
                    self.ibr_reecb_Tp = np.delete(self.ibr_reecb_Tp, j)
                    self.ibr_reecb_Tpord = np.delete(self.ibr_reecb_Tpord, j)
                    self.ibr_reecb_Trv = np.delete(self.ibr_reecb_Trv, j)
                    self.ibr_reecb_VFLAG = np.delete(self.ibr_reecb_VFLAG, j)
                    self.ibr_reecb_Vdip = np.delete(self.ibr_reecb_Vdip, j)
                    self.ibr_reecb_Vmax = np.delete(self.ibr_reecb_Vmax, j)
                    self.ibr_reecb_Vmin = np.delete(self.ibr_reecb_Vmin, j)
                    self.ibr_reecb_Vref0 = np.delete(self.ibr_reecb_Vref0, j)
                    self.ibr_reecb_Vup = np.delete(self.ibr_reecb_Vup, j)
                    self.ibr_reecb_bus = np.delete(self.ibr_reecb_bus, j)
                    self.ibr_reecb_dPmax = np.delete(self.ibr_reecb_dPmax, j)
                    self.ibr_reecb_dPmin = np.delete(self.ibr_reecb_dPmin, j)
                    self.ibr_reecb_dbd1 = np.delete(self.ibr_reecb_dbd1, j)
                    self.ibr_reecb_dbd2 = np.delete(self.ibr_reecb_dbd2, j)
                    self.ibr_reecb_id = np.delete(self.ibr_reecb_id, j)

                    self.ibr_repca_Ddn = np.delete(self.ibr_repca_Ddn, j)
                    self.ibr_repca_Dup = np.delete(self.ibr_repca_Dup, j)
                    self.ibr_repca_FFlag = np.delete(self.ibr_repca_FFlag, j)
                    self.ibr_repca_Kc = np.delete(self.ibr_repca_Kc, j)
                    self.ibr_repca_Ki = np.delete(self.ibr_repca_Ki, j)
                    self.ibr_repca_Kig = np.delete(self.ibr_repca_Kig, j)
                    self.ibr_repca_Kp = np.delete(self.ibr_repca_Kp, j)
                    self.ibr_repca_Kpg = np.delete(self.ibr_repca_Kpg, j)
                    self.ibr_repca_Pmax = np.delete(self.ibr_repca_Pmax, j)
                    self.ibr_repca_Pmin = np.delete(self.ibr_repca_Pmin, j)
                    self.ibr_repca_Qmax = np.delete(self.ibr_repca_Qmax, j)
                    self.ibr_repca_Qmin = np.delete(self.ibr_repca_Qmin, j)
                    self.ibr_repca_Rc = np.delete(self.ibr_repca_Rc, j)
                    self.ibr_repca_RefFlag = np.delete(self.ibr_repca_RefFlag, j)
                    self.ibr_repca_Tfltr = np.delete(self.ibr_repca_Tfltr, j)
                    self.ibr_repca_Tft = np.delete(self.ibr_repca_Tft, j)
                    self.ibr_repca_Tfv = np.delete(self.ibr_repca_Tfv, j)
                    self.ibr_repca_Tg = np.delete(self.ibr_repca_Tg, j)
                    self.ibr_repca_Tp = np.delete(self.ibr_repca_Tp, j)
                    self.ibr_repca_VCFlag = np.delete(self.ibr_repca_VCFlag, j)
                    self.ibr_repca_Vfrz = np.delete(self.ibr_repca_Vfrz, j)
                    self.ibr_repca_Xc = np.delete(self.ibr_repca_Xc, j)
                    self.ibr_repca_branch_From_bus = np.delete(self.ibr_repca_branch_From_bus, j)
                    self.ibr_repca_branch_To_bus = np.delete(self.ibr_repca_branch_To_bus, j)
                    self.ibr_repca_branch_id = np.delete(self.ibr_repca_branch_id, j)
                    self.ibr_repca_bus = np.delete(self.ibr_repca_bus, j)
                    self.ibr_repca_dbd1 = np.delete(self.ibr_repca_dbd1, j)
                    self.ibr_repca_dbd2 = np.delete(self.ibr_repca_dbd2, j)
                    self.ibr_repca_emax = np.delete(self.ibr_repca_emax, j)
                    self.ibr_repca_emin = np.delete(self.ibr_repca_emin, j)
                    self.ibr_repca_fdbd1 = np.delete(self.ibr_repca_fdbd1, j)
                    self.ibr_repca_fdbd2 = np.delete(self.ibr_repca_fdbd2, j)
                    self.ibr_repca_femax = np.delete(self.ibr_repca_femax, j)
                    self.ibr_repca_femin = np.delete(self.ibr_repca_femin, j)
                    self.ibr_repca_id = np.delete(self.ibr_repca_id, j)
                    self.ibr_repca_remote_bus = np.delete(self.ibr_repca_remote_bus, j)


                    for k in range(j, len(self.ibr_wecc_idx), 1):
                        self.ibr_wecc_idx[k] = self.ibr_wecc_idx[k] - 1
                    self.ibr_wecc_idx = np.delete(self.ibr_wecc_idx, j)
                    self.ibr_wecc_n = self.ibr_wecc_n - 1
                    self.ibr_n = self.ibr_n - 1
        

            

        # measured bus
        nmea = len(self.mea_bus)
        for i in range(nmea):
            j = nmea - i - 1
            if self.mea_bus[j] not in emt_buses:
                self.mea_bus = np.delete(self.mea_bus, j)
                self.mea_method = np.delete(self.mea_method, j)
        # pll bus
        npll = len(self.pll_bus)
        for i in range(npll):
            j = npll - i - 1
            if self.pll_bus[j] not in emt_buses:
                self.pll_bus = np.delete(self.pll_bus, j)
                self.pll_ke = np.delete(self.pll_ke, j)
                self.pll_te = np.delete(self.pll_te, j)    
        # vm bus
        nvm = len(self.vm_bus)
        for i in range(nvm):
            j = nvm - i - 1
            if self.vm_bus[j] not in emt_buses:
                self.vm_bus = np.delete(self.vm_bus, j)
                self.vm_te = np.delete(self.vm_te, j)     
