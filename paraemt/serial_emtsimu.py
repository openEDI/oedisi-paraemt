
import numpy as np
import time
import pandas as pd

from paraemt.emtsimu import EmtSimu
from paraemt.psutils import initialize_emt, modify_system

class SerialEmtSimu(EmtSimu):

    @staticmethod
    def initialize_from_snp(input_snp, netMod):
        emt = super(SerialEmtSimu, SerialEmtSimu).initialize_from_snp(input_snp, netMod)
        emt.init_ibr_epri()
        return emt

    def __init__(self,
                 workingfolder='',
                 systemN=1,
                 EMT_N=0,
                 N_row=1,
                 N_col=1,
                 ts=50e-6,
                 Tlen=0.2,
                 kts = round(1/60/4/50e-6),
                 stepk = round(1/60/4/50e-6) - 1,
                 save_rate=1,
                 netMod='lu',
                 loadmodel_option=1,
                 record4cosim = False,
                 playback_enable=True,
                 Gd = 100,
                 Go = 0.001,
                 ):

        t0 = time.time()

        if workingfolder != '':
            (pfd, ini, dyd, emt_zones) = initialize_emt(workingfolder,
                                             systemN,
                                             EMT_N,
                                             N_row,
                                             N_col,
                                             ts,
                                             Tlen,
                                             kts,
                                             stepk,
                                             save_rate,
                                             netMod,
                                             loadmodel_option,
                                             record4cosim,
                                             )
        else:
            raise RuntimeError("Must provide 'workingfolder' keyword containing necessary initialization data")
        ## End if

        super().__init__(systemN,
                         EMT_N,
                         len(pfd.gen_bus),
                         dyd.ibr_wecc_n,
                         len(pfd.bus_num),
                         len(pfd.load_bus),
                         dyd.ibr_epri_n,
                         save_rate,
                         )
        self.ts = ts  # second
        self.kts = kts
        self.stepk = stepk
        self.Tlen = Tlen  # second
        self.iphasor = 0
        self.vphasor = 0

        self.systemN = systemN
        self.EMT_N = EMT_N
        self.ini = ini
        self.pfd = pfd
        self.dyd = dyd
        self.emt_zones = emt_zones

        self.init_ibr_epri()

        self.preprocess(ini, pfd, dyd)

        bus_rec, current_rec, voltage_rec = self.Record4CoSim(record4cosim,
                                                              self.ini.Init_brch_Ipre,
                                                              self.Vsol,
                                                              0.0)
        self.bus_rec = bus_rec
        self.current_rec = {}
        self.current_rec[0] = current_rec
        self.voltage_rec = {}
        self.voltage_rec[0] = voltage_rec
        self.playback_enable = playback_enable
        self.playback_t_chn = 1
        self.playback_sig_chn = 2
        self.playback_tn = 0
        self.Gd = Gd
        self.Go = Go


        modify_system(EMT_N, pfd, ini, record4cosim, emt_zones, Gd, Go)

        t1 = time.time()

        self.init_time = t1 - t0

        return

    def presolveV(self):

        self.Vsol_1 = self.Vsol

        return


    def solveV(self, admittance_mode, Igs, Igi, node_Ihis, Il):

        # self.Vsol_1 = self.Vsol

        I_RHS = Igs + Igi + node_Ihis

        if self.loadmodel_option == 2 and Il is not None:
            I_RHS += Il

        if admittance_mode == 'inv':

            Vsol = self.Ginv @ I_RHS

        elif admittance_mode == 'lu':

            Vsol = self.Glu.solve(I_RHS)

        elif admittance_mode == 'bbd':

            # TODO: Make this a thing...
            # self.Vsol = self.Gbbd.schur_solve(self.I_RHS)

            pass

        else:

            raise ValueError('Unrecognized mode: {}'.format(admittance_mode))

        return Vsol


    def run_simulation(self, debug=False, show_progress=False, record4cosim=False, playback_voltphasor=False):

        dyd = self.dyd
        ini = self.ini
        pfd = self.pfd

        t_evnt = 0.0
        t_pred = 0.0
        t_upig = 0.0
        t_upir = 0.0
        t_uper = 0.0
        t_upil = 0.0
        t_rent = 0.0
        t_solve = 0.0
        t_busmea = 0.0
        t_upx = 0.0
        t_upxr = 0.0
        t_upxl = 0.0
        t_save = 0.0
        t_phsr = 0.0
        t_helc = 0.0
        t_upih = 0.0
        Nsteps = 0

        tn = 0
        tsave = 0
        ts = self.ts

        nbus =  len(pfd.bus_num)

        if debug:
            xpre = {0:self.x_pv_1.copy()}
            igs = {0:self.Igs.copy()}
            igi = {0:self.Igi.copy()}
            ige = {0:self.Igi_epri.copy()}
            il = {0:self.Il.copy()}
            ihis = {0:self.node_Ihis.copy()}
            # irhs = {}
            vma = {}
            vpn = {}
            ima = {}
            ipn = {}
        ## End if

        t0 = time.time()

        # self.force_numba_compilation(ts)

        t1 = time.time()

        while tn*ts < self.Tlen:

            tn += 1
            if show_progress:    # Min Deleted 01302024
                if tn>1:
                    if np.mod(tn,500)==0:
                        print("%.3f" % self.t[-1])
            print("**** t = {} ****".format(tn*ts))

            flag_reini = 0

            tl_0 = time.time()

            self.StepChange(dyd, ini, tn)  # configure step change in exc or gov references

            # hard-coded disturbance for benchmarking EPRI's IBR model on the small test system
            if (self.systemN == 10) | (self.systemN == 11) | (self.systemN == 12):
                # ================================================================
                # added on 2/21/2023
                # step changes in IBR's PQ ref
                if tn*ts>= 4.0:
                    self.ibr_epri[0].cExternalInputs[9] = 45.052
                    if tn*ts>= 8.0:
                        self.ibr_epri[0].cExternalInputs[10] = 0.2026
                # ================================================================

            if self.systemN == 17:
                # ================================================================
                # added on 6/28/2023
                # step change in IBR's P ref
                if tn*ts>= 6.0:
                    self.ibr_epri[0].cExternalInputs[9] = 61.2
                # ================================================================
            
            if (self.systemN == 18) and (len(self.ibr_epri)>0):
                # ================================================================
                # added on 9/25/2023
                # step change in IBR's P ref
                if tn*ts>= 15:
                    self.ibr_epri[0].cExternalInputs[9] = 75
                # ================================================================

            # perturbing a specific state
            if self.pert_cplt == 0:
                if tn*ts >= self.pert_t:
                    self.pert_cplt = 1

                    if self.pert_sg_ibr == 0:
                        self.x_pv_1[self.pert_idx] = self.x_pv_1[self.pert_idx] + self.pert_dx
                        self.x_pred[0][self.pert_idx] = self.x_pv_1[self.pert_idx]
                        self.x_pred[1][self.pert_idx] = self.x_pv_1[self.pert_idx]
                        self.x_pred[2][self.pert_idx] = self.x_pv_1[self.pert_idx]

                    if self.pert_sg_ibr == 1:
                        self.x_ibr_pv_1[self.pert_idx] = self.x_ibr_pv_1[self.pert_idx] + self.pert_dx

                    if self.pert_sg_ibr == 2:
                        self.x_ibr_epri_pv_1[self.pert_idx] = self.x_ibr_epri_pv_1[self.pert_idx] + self.pert_dx

            if ((self.flag_gentrip == 0 and self.flag_reinit == 1) or
                (tn*ts >= self.fault_t and (tn - 1)*ts < self.fault_t) or
                (tn*ts >= self.fault_t + self.fault_tlen and (tn - 1)*ts < self.fault_t + self.fault_tlen)
                ):
                flag_reini = 1
            ## End if

            if tn*ts < self.fault_t:
                self.Ginv = ini.Init_net_G0
                self.net_coe = ini.Init_net_coe0
                self.Glu = ini.Init_net_G0_lu
            elif (tn*ts >= self.fault_t) and (tn*ts < self.fault_t+self.fault_tlen):
                self.Ginv = ini.Init_net_G1
                self.net_coe = ini.Init_net_coe1
                self.Glu = ini.Init_net_G1_lu
            else:
                self.Ginv = ini.Init_net_G2
                self.net_coe = ini.Init_net_coe2
                self.Glu = ini.Init_net_G2_lu
            ## End if

            tl_1 = time.time()

            self.predictX(pfd, dyd, self.ts)

            tl_2 = time.time()

            self.updateIg(pfd, dyd, ini)

            tl_3 = time.time()

            self.updateIibr(pfd, dyd, ini)

            tl_4 = time.time()

            self.updateIibr_epri(pfd, dyd, ini, tn)

            tl_5 = time.time()

            self.updateIl(pfd, dyd, tn) # update current injection from load

            tl_6 = time.time()

            # ibrbase = pd.DataFrame(np.transpose(self.pfd.bus_name))
            # ibrbase.to_csv("EMTbusNAME.csv")
            # ibrbase = pd.DataFrame(np.transpose(self.pfd.bus_num))
            # ibrbase.to_csv("EMTbusNUMBER.csv")
            # ibrbase = pd.DataFrame(np.transpose(self.pfd.gen_id))
            # ibrbase.to_csv("EMTgenID.csv")
            # ibrbase = pd.DataFrame(np.transpose(self.pfd.gen_bus))
            # ibrbase.to_csv("EMTgenBUS.csv")
            # ibrbase = pd.DataFrame(np.transpose(self.pfd.ibr_id))
            # ibrbase.to_csv("EMTibrID.csv")
            # ibrbase = pd.DataFrame(np.transpose(self.pfd.ibr_bus))
            # ibrbase.to_csv("EMTibrBUS.csv")

            self.helics_receive(tn, record4cosim)

            tl_7 = time.time()

            self.GenTrip(pfd, dyd, ini, tn)  # configure generation trip

            tl_8 = time.time()

            # re-init
            if flag_reini==1:
                self.Re_Init(pfd, dyd, ini, tn)
            # End if

            tl_9 = time.time()

            if debug:
                xpre[tn] = self.x_pv_1.copy()
                igs[tn] = self.Igs.copy()
                igi[tn] = self.Igi.copy()
                ige[tn] = self.Igi_epri.copy()
                il[tn] = self.Il.copy()
                ihis[tn] = self.node_Ihis.copy()

            tl_10 = time.time()

            self.presolveV()
            self.Vsol = self.solveV(ini.admittance_mode,
                                    self.Igs,
                                    self.Igi + self.Igi_epri,
                                    self.node_Ihis,
                                    self.Il)

            tl_11 = time.time()

            # if playback_voltphasor == True:
            #     Vtemp = 0*self.Vsol
            #     for busi in range(nbus):
            #         busi_csv = np.where(self.tsat_bus_num == pfd.bus_num[busi])[0][0]

            #         # # interpolation
            #         # idx = np.where(self.tsat_t>=tn*ts)[0]
            #         # idx = idx[0]-1
            #         # alfa = (tn*ts-self.tsat_t[idx]) / (self.tsat_t[idx+1] - self.tsat_t[idx])

            #         # Vmtemp = (1-alfa)*self.tsat_vt[idx][busi_csv+1] + alfa*self.tsat_vt[idx+1][busi_csv+1]
            #         # Vatemp = (1-alfa)*self.tsat_va[idx][busi_csv+1] + alfa*self.tsat_va[idx+1][busi_csv+1]

            #         Vmtemp = self.tsat_vt[tn][busi_csv+1]
            #         Vatemp = self.tsat_va[tn][busi_csv+1]

            #         Vtemp[busi] = Vmtemp*np.cos(pfd.ws*tn*ts + Vatemp/180*np.pi)
            #         Vtemp[busi + nbus] = Vmtemp*np.cos(pfd.ws*tn*ts + Vatemp/180*np.pi - 2*np.pi/3)
            #         Vtemp[busi + 2*nbus] = Vmtemp*np.cos(pfd.ws*tn*ts + Vatemp/180*np.pi + 2*np.pi/3)
            #     self.Vsol = Vtemp



            self.BusMea(pfd, dyd, tn)  # bus measurement

            tl_12 = time.time()

            self.updateX(pfd, dyd, ini, tn, playback_voltphasor)

            tl_13 = time.time()

            self.updateXibr(pfd, dyd, ini)

            tl_14 = time.time()

            self.updateXl(pfd, dyd, tn)

            tl_15 = time.time()

            

            if debug and len(self.fft_vabc) >= self.fft_N:
                # print("Saving ffts at time ", tn)
                vma[tn] = self.fft_vma
                vpn[tn] = self.fft_vpn0
                ima[tn] = self.fft_ima
                ipn[tn] = self.fft_ipn0
            ## End if

            tl_16 = time.time()

            if (len(self.emt_zones)>0) and record4cosim==False: 
                self.update_phasor()
            # self.update_phasor()
            
            tl_17 = time.time()

            self.helics_publish()

            tl_18 = time.time()

            if tn*ts >= self.t_gentrip:
                pass

            self.updateIhis(ini)

            self.save(tn, record4cosim)  # save has to be placed after updateIhis to ensure time alignment for recorded signals for pseudo co-sim

            tl_19 = time.time()

            self.helics_update()

            tl_20 = time.time()

            t_evnt += tl_1 - tl_0
            t_pred += tl_2 - tl_1
            t_upig += tl_3 - tl_2
            t_upir += tl_4 - tl_3
            t_uper += tl_5 - tl_4
            t_upil += tl_6 - tl_5
            t_helc += tl_7 - tl_6
            t_evnt += tl_8 - tl_7
            t_rent += tl_9 - tl_8
            t_save += tl_10 - tl_9
            t_solve += tl_11 - tl_10
            t_busmea += tl_12 - tl_11
            t_upx += tl_13 - tl_12
            t_upxr += tl_14 - tl_13
            t_upxl += tl_15 - tl_14
            t_save += tl_16 - tl_15
            t_phsr += tl_17 - tl_16
            t_helc += tl_18 - tl_17
            t_upih += tl_19 - tl_18
            t_helc += tl_20 - tl_19

            Nsteps += 1

        #### END TIME LOOP ####

        t2 = time.time()

        numba_comp = t1 - t0
        loop = t2 - t1
        elapsed = numba_comp + loop + self.init_time
        timing_string = """**** Timing Info ****
    Dimension:   {:8d}
    Init:        {:10.2e} {:8.2%}
    Comp:        {:10.2e} {:8.2%}
    Loop:        {:10.2e} {:8.2%} {:8d} {:8.2e}
      Event:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      PredX:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdIG:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdIR:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdER:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdIL:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      ReInit:      {:10.2e} {:8.2%} {:8d} {:8.2e}
      Solve:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      BusMea:      {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdX:        {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdXR:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdXL:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      Save:        {:10.2e} {:8.2%} {:8d} {:8.2e}
      UpdIH:       {:10.2e} {:8.2%} {:8d} {:8.2e}
      Phasor:      {:10.2e} {:8.2%} {:8d} {:8.2e}
      Helics:      {:10.2e} {:8.2%} {:8d} {:8.2e}
    Total:       {:10.2e}

        """.format(self.ini.Init_net_G0_inv.shape[0],
                   self.init_time, self.init_time / elapsed,
                   numba_comp, numba_comp / elapsed,
                   loop, loop / elapsed, Nsteps, loop / Nsteps,
                   t_evnt, t_evnt / elapsed, Nsteps, t_evnt / Nsteps,
                   t_pred, t_pred / elapsed, Nsteps, t_pred / Nsteps,
                   t_upig, t_upig / elapsed, Nsteps, t_upig / Nsteps,
                   t_upir, t_upir / elapsed, Nsteps, t_upir / Nsteps,
                   t_uper, t_uper / elapsed, Nsteps, t_uper / Nsteps,
                   t_upil, t_upil / elapsed, Nsteps, t_upil / Nsteps,
                   t_rent, t_rent / elapsed, Nsteps, t_rent / Nsteps,
                   t_solve, t_solve / elapsed, Nsteps, t_solve / Nsteps,
                   t_busmea, t_busmea / elapsed, Nsteps, t_busmea / Nsteps,
                   t_upx, t_upx / elapsed, Nsteps, t_upx / Nsteps,
                   t_upxr, t_upxr / elapsed, Nsteps, t_upxr / Nsteps,
                   t_upxl, t_upxl / elapsed, Nsteps, t_upxl / Nsteps,
                   t_save, t_save / elapsed, Nsteps, t_save / Nsteps,
                   t_upih, t_upih / elapsed, Nsteps, t_upih / Nsteps,
                   t_phsr, t_phsr / elapsed, Nsteps, t_phsr / Nsteps,
                   t_helc, t_helc / elapsed, Nsteps, t_helc / Nsteps,
                   elapsed
                   )
        print(timing_string)

        if debug:
            self.dbg_xpre = xpre
            self.dbg_igs = igs
            self.dbg_igi = igi
            self.dbg_ige = ige
            self.dbg_il = il
            self.dbg_ihis = ihis
            self.dbg_vma = vma
            self.dbg_vpn = vpn
            self.dbg_ima = ima
            self.dbg_ipn = ipn
        ## End if

        return

    def calc_int_i(self, ii, vv, bb):

        n = bb.shape[1]
        nt = ii.shape[0]

        assert(ii.shape == vv.shape)
        assert(ii.shape == (nt, 3*n + 1))
        assert(bb.shape == (1,n))

        i_int = np.zeros((nt,3*n))

        Gd = self.Gd
        Go = self.Go
        G = np.array([[Gd, Go, Go], [Go, Gd, Go], [Go, Go, Gd]])

        for k in range(n):

            itemp = ii[:, 3*k+1:3*(k+1)+1]
            vtemp = vv[:, 3*k+1:3*(k+1)+1]

            i_int[:,3*k:3*(k+1)] = np.transpose(G @ np.transpose(vtemp)  + np.transpose(itemp))

        ## End for

        return i_int

