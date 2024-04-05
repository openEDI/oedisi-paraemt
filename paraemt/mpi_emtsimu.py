
import numpy as np
import time
import os
from paraemt.emtsimu import EmtSimu
from paraemt.mpi_bbd_matrix import schur_bbd_lu
from paraemt.mpi_utils import *
from paraemt.partitionutil import form_bbd
from paraemt.psutils import initialize_emt, modify_system

from mpi4py import MPI

if 'EPRI_IBR' in os.environ.keys():
    EPRI_IBR = int(os.environ['EPRI_IBR'])
else:
    EPRI_IBR = False

class MpiEmtSimu(EmtSimu):

    def __init__(self,
                 workingfolder='',
                 systemN=1,
                 N_row=1,
                 N_col=1,
                 ts=50e-6,
                 Tlen=0.2,
                 save_rate=1,
                 emt_buses=[],
                 emt_branches=[],
                 loadmodel_option=1,
                 flag_itfc_R=0,
                 ):

        t0 = time.time()

        if workingfolder != '':
            (pfd, ini, dyd) = initialize_emt(workingfolder,
                                             systemN,
                                             N_row,
                                             N_col,
                                             ts,
                                             Tlen,
                                             save_rate,
                                             emt_buses,
                                             emt_branches,
                                             'bbd',
                                             loadmodel_option,
                                             )
        else:
            raise RuntimeError("Must provide 'workingfolder' keyword")
        ## End if


        super().__init__(len(pfd.gen_bus),
                         dyd.ibr_wecc_n,
                         len(pfd.bus_num),
                         len(pfd.load_bus),
                         dyd.ibr_epri_n,
                         save_rate,
                         )
        self.ts = ts  # second
        self.Tlen = Tlen  # second

        self.systemN = systemN
        self.ini = ini
        self.pfd = pfd
        self.dyd = dyd

        self.brch_ipre_need_sync = False
        self.comm = None

        if EPRI_IBR:
            self._init_ibr_epri()

        self.preprocess(ini, pfd, dyd)

        modify_system(systemN, flag_itfc_R, dyd, ini)

        self.solve_sync_time = 0.0
        self.solve_reorder_time = 0.0
        self.solve_schur_time = 0.0
        # TODO: Track total mpi comm time
        self.mpi_total_time = 0.0

        t1 = time.time()

        self.init_time = t1 - t0

        return


    def form_bbd(self, G0, nparts):

        t0 = time.time()

        (self.BBD_0,
         self.index_order_0,
         self.inv_order_0) = form_bbd(self.ini.Init_net_G0,
                                      nparts)
        self.nparts = nparts

        # if self.fault_t and self.fault_t < self.Tlen:

        #     (self.BBD_1, self.index_order_1, self.inv_order_1) = form_bbd(ini.Init_net_G1,
        #                                                                   nparts)

        #     if self.fault_tripline != 0:

        #         (self.BBD_2, self.index_order_2, self.inv_order_2) = form_bbd(ini.Init_net_G2,
        #                                                                       nparts)

        #     else:

        #         self.BBD_2 = self.BBD_0
        #         self.index_order_2 = self.index_order_0
        #         self.inv_order_2 = self.inv_order_0

        #     ## End if

        # ## End if

        t1 = time.time()

        self.bbd_time = t1 - t0

        return


    def set_mpi_comm(self, comm):

        # WARNING: This function must be called after `preprocess` in EmtSimu parent object

        t0 = time.time()

        self.comm = comm.Dup()

        self.rank = self.comm.Get_rank()

        size = self.comm.Get_size()

        if size > 1:
            (self.gen_range, self.gen_counts) = self._divide_range(self.ngen, size)
            (self.ibr_range, self.ibr_counts) = self._divide_range(self.nibr, size)
            (self.ebr_range, self.ebr_counts) = self._divide_range(self.nibrepri, size)
            (self.brch_range, self.brch_counts) = self._divide_range(len(self.brch_Ihis), size)
            (self.bus_range, self.bus_counts) = self._divide_range(self.nbus, size)

            # print("Gen Range: ", self.gen_range[:,self.rank])
            # print("IBR Range: ", self.ibr_range[:,self.rank])
            # print("EPRI IBR Range: ", self.ebr_range[:,self.rank])
            # print("Brch Range: ", self.brch_range[:,self.rank])

            if self.loadmodel_option == 2:
                (self.load_range, self.load_counts) = self._divide_range(self.nload, size)
                # print("Load Range: ", self.load_range[:,self.rank])

        if self.rank != 0:
            self.node_Ihis[:] = 0.0

        t1 = time.time()

        self.mpi_setup_time = t1 - t0

        return


    def setup_c_libs(self):
        self.init_ibr_epri()
        return


    def compute_bbd_lu(self, drop_tol=-1.0, dense_corner=False):

        t0 = time.time()

        self.mpi_lu_0 = schur_bbd_lu(self.comm,
                                     self.BBD_0,
                                     drop_tol=drop_tol,
                                     dense_corner=dense_corner)

        self.mpi_lu_0.print_summary()

        # TODO: Need fault simulation?
        # if self.fault_t and self.fault_t < self.Tlen:

        #     self.mpi_lu_1 = schur_bbd_lu(self.comm,
        #                                  self.BBD_1,
        #                                  drop_tol=drop_tol,
        #                                  dense_corner=dense_corner)

        #     if self.fault_tripline != 0:
        #         self.mpi_lu_2 = schur_bbd_lu(self.comm,
        #                                      self.BBD_2,
        #                                      drop_tol=drop_tol,
        #                                      dense_corner=dense_corner)
        #     else:
        #         self.mpi_lu_2 = self.mpi_lu_0
        #     ## End if
        # ## End if

        t1 = time.time()

        self.bbd_lu_time = t1 - t0

        self.mpi_lu = self.mpi_lu_0
        self.index_order = self.index_order_0
        self.inv_order = self.inv_order_0

        return


    def helics_setup(self):
        if self.comm.Get_rank() == 0:
            super().helics_setup()
        else:
            pass
        return


    def helics_receive(self, tn):
        if self.comm.Get_rank() == 0:
            super().helics_receive(tn)
        else:
            pass
        return


    def helics_publish(self):
        if self.comm.Get_rank() == 0:
            super().helics_publish()
        else:
            pass
        return


    def helics_update(self):
        if self.comm.Get_rank() == 0:
            super().helics_update()
        else:
            pass
        return


    def presolveV(self):
        self.Vsol_1 = self.Vsol
        return


    def sync_I_RHS(self, Igs, Igi, node_Ihis, Il):

        self.I_RHS = Igs + node_Ihis + Igi

        if self.loadmodel_option == 2 and Il is not None:
            self.I_RHS += Il

        if self.comm != None:
            self.comm.Allreduce(MPI.IN_PLACE, self.I_RHS)

        return


    def solveV(self, admittance_mode, Igs, Igi, node_Ihis, Il):

        t0 = time.time()

        self.sync_I_RHS(Igs, Igi, node_Ihis, Il)

        t1 = time.time()

        tmpIRHS = self.I_RHS[self.index_order]

        t2 = time.time()

        tmpVsol = self.mpi_lu.schur_solve(tmpIRHS)

        t3 = time.time()

        Vsol = tmpVsol[self.inv_order]

        t4 = time.time()

        self.solve_sync_time += t1 - t0
        self.solve_reorder_time += t2 - t1 + t4 - t3
        self.solve_schur_time += t3 - t2

        return Vsol


    def BusMea(self, pfd, dyd, tn):

        rt = super().BusMea(pfd, dyd, tn)

        if self.comm != None:

            recv_buf = [self.x_bus_pv_1,
                        6*self.bus_counts,
                        6*self.bus_range[0,:],
                        MPI.DOUBLE
                        ]

            sidx = 6*self.bus_range[0,self.rank]
            eidx = 6*self.bus_range[1,self.rank]
            send_buf = self.x_bus_pv_1[sidx:eidx]

            self.comm.Allgatherv(send_buf, recv_buf)

        ## End if

        return rt


    def save(self, tn):

        if (self.comm is not None and
            self.comm.Get_size() > 1 and
            np.mod(tn, self.save_rate) == 0
            ):

            if self.ngen > 0:
                if self.rank == 0:
                    self.comm.Reduce(MPI.IN_PLACE, self.x_pv_1, root=0)
                else:
                    self.comm.Reduce(self.x_pv_1, None, root=0)
                ## End if
            ## End if

            if self.nibr > 0:

                recv_buf = [self.x_ibr_pv_1,
                            41*self.ibr_counts,
                            41*self.ibr_range[0,:],
                            MPI.DOUBLE
                            ]

                sidx = 41*self.ibr_range[0,self.rank]
                eidx = 41*self.ibr_range[1,self.rank]
                send_buf = self.x_ibr_pv_1[sidx:eidx]

                self.comm.Gatherv(send_buf, recv_buf, root=0)

            ## End if

            if self.nibrepri > 0:

                recv_buf = [self.x_ibr_epri_pv_1,
                            13*self.ebr_counts,
                            13*self.ebr_range[0,:],
                            MPI.DOUBLE
                            ]

                sidx = 13*self.ebr_range[0,self.rank]
                eidx = 13*self.ebr_range[1,self.rank]
                send_buf = self.x_ibr_epri_pv_1[sidx:eidx]

                self.comm.Gatherv(send_buf, recv_buf, root=0)

            ## End if

            if self.nload > 0 and self.loadmodel_option == 2:

                recv_buf = [self.x_load_pv_1,
                            4*self.load_counts,
                            4*self.load_range[0,:],
                            MPI.DOUBLE
                            ]
                sidx = 4*self.load_range[0,self.rank]
                eidx = 4*self.load_range[1,self.rank]
                send_buf = self.x_load_pv_1[sidx:eidx]

                self.comm.Gatherv(send_buf, recv_buf, root=0)
            ## End if

        ## End if

        return super().save(tn)


    def updateIibr_epri(self, pfd, dyd, ini, tn):

        if self.comm is not None and self.comm.Get_size() > 1:
            self.sync_brch_Ipre()
        ## End if

        return super().updateIibr_epri(pfd, dyd, ini, tn)


    def update_phasor(self):

        if self.compute_phasor == 1 and self.comm is not None and self.comm.Get_size() > 1:
            self.sync_brch_Ipre()
        ## End if

        return super().update_phasor()


    def updateIhis(self, ini):

        self.brch_ipre_need_sync = True

        return super().updateIhis(ini)


    def run_simulation(self, debug=False):

        rank = self.comm.Get_rank()

        pfd = self.pfd
        ini = self.ini
        dyd = self.dyd

        t_evnt = 0.0
        t_pred = 0.0
        t_upig = 0.0
        t_upir = 0.0
        t_uper = 0.0
        t_upil = 0.0
        t_plbk = 0.0
        t_solve = 0.0
        t_busmea = 0.0
        t_upx = 0.0
        t_upxr = 0.0
        t_upxl = 0.0
        t_save = 0.0
        t_phsr = 0.0
        t_upih = 0.0
        t_helc = 0.0
        t_rein = 0.0
        Nrein = 0
        Nsteps = 0

        if debug:
            xpre = {0:self.x_pv_1.copy()}
            igs = {0:self.Igs.copy()}
            igi = {0:self.Igi.copy()}
            ige = {0:self.Igi_epri.copy()}
            il = {0:self.Il.copy()}
            ihis = {0:self.node_Ihis.copy()}
            vma = {}
            vpn = {}
            ima = {}
            ipn = {}
        ## End if

        #### BEGIN TIME LOOP ####

        self.net_coe = ini.Init_net_coe0

        tn = 0
        ts = self.ts

        t_start = time.time()

        while tn * self.ts < self.Tlen:

            tn += 1
            flag_reini = 0

            # print("**** tn = {} ****".format(tn))

            tl_0 = time.time()

            self.StepChange(dyd, ini, tn)

            # Check if we need to reinitialize
            if ((self.flag_gentrip == 0 and self.flag_reinit == 1) or
                (tn*ts >= self.fault_t and (tn - 1)*ts < self.fault_t) or
                (tn*ts >= self.fault_t + self.fault_tlen and (tn - 1)*ts < self.fault_t + self.fault_tlen)
                ):
                flag_reini = 1


            ## TODO: Need to add fault stuff back?
            # # Get the network information for this time step
            # if tn*ts<self.fault_t:

            #     self.net_coe = ini.Init_net_coe0
            #     mpi_lu = mpi_lu_0
            #     index_order = index_order_0
            #     inv_order = inv_order_0

            # elif (tn*ts >= self.fault_t) and (tn*ts < self.fault_t+self.fault_tlen):

            #     self.net_coe = ini.Init_net_coe1
            #     mpi_lu = mpi_lu_1
            #     index_order = index_order_1
            #     inv_order = inv_order_1

            # else:

            #     self.net_coe = ini.Init_net_coe2
            #     mpi_lu = mpi_lu_2
            #     index_order = index_order_2
            #     inv_order = inv_order_2

            # ## END if ##

            tl_1 = time.time()

            self.predictX(pfd, dyd, self.ts)

            tl_2 = time.time()

            self.updateIg(pfd, dyd, ini)

            tl_3 = time.time()

            self.updateIibr(pfd, dyd, ini)

            tl_4 = time.time()

            self.updateIibr_epri(pfd, dyd, ini, tn)

            tl_5 = time.time()

            self.updateIl(pfd, dyd, tn)

            tl_6 = time.time()

            self.helics_receive(tn)

            tl_7 = time.time()

            # TODO: What if there is a branch fault and a gen trip?
            repartition = self.GenTrip(pfd, dyd, ini, tn)

            if repartition:

                # GenTrip changes admittance matrix. Need to repartition.
                if self.comm.Get_rank() == 0:
                    (BBD, index_order, inv_order) = form_bbd(self.ini.Init_net_G0,
                                                             self.nparts)
                else:
                    BBD = None
                    index_order = inv_order = None
                ## End if

                self.comm.Barrier()

                # GenTrip changes things in `ini` object. Redistribute.
                (self.ini, self.index_order, self.inv_order) = mpi_distribute_obj(
                    [ini, index_order, inv_order],
                    self.comm,
                    root=0)

                # Admittance matrix changed so need to redistribute and factor
                self.mpi_lu = schur_bbd_lu(self.comm,
                                           BBD,
                                           drop_tol=self.mpi_lu.drop_tol,
                                           dense_corner=self.mpi_lu.dense_corner)
                self.mpi_lu.print_summary()

            #### END IF ####

            tl_8 = time.time()

            if flag_reini == 1:
                self.Re_Init(pfd, dyd, ini)
                Nrein += 1
            #### END IF RE_INIT ####

            tl_9 = time.time()

            # self.sync_I_RHS()

            if debug:
                mpi_sync_and_save(xpre, self.x_pv_1.copy(), tn, self.comm, 0)
                mpi_sync_and_save(igs, self.Igs.copy(), tn, self.comm, 0)
                mpi_sync_and_save(igi, self.Igi.copy(), tn, self.comm, 0)
                mpi_sync_and_save(ige, self.Igi_epri.copy(), tn, self.comm, 0)
                mpi_sync_and_save(il, self.Il.copy(), tn, self.comm, 0)
                mpi_sync_and_save(ihis, self.node_Ihis.copy(), tn, self.comm, 0)
            ## End if

            tl_10 = time.time()

            self.presolveV()
            self.Vsol = self.solveV('bbd',
                                    self.Igs,
                                    self.Igi + self.Igi_epri,
                                    self.node_Ihis,
                                    self.Il)

            tl_11 = time.time()

            self.BusMea(pfd, dyd, tn)

            tl_12 = time.time()

            self.updateX(pfd, dyd, ini, tn)

            tl_13 = time.time()

            self.updateXibr(pfd, dyd, ini)

            tl_14 = time.time()

            self.updateXl(pfd, dyd, tn)

            tl_15 = time.time()

            self.save(tn)

            if debug and len(self.fft_vabc) >= self.fft_N:
                vma[tn] = self.fft_vma
                vpn[tn] = self.fft_vpn0
                ima[tn] = self.fft_ima
                ipn[tn] = self.fft_ipn0
            ## End if

            tl_16 = time.time()

            self.update_phasor()

            tl_17 = time.time()

            self.helics_publish()

            tl_18 = time.time()

            self.updateIhis(ini)

            # tp = "Rank: {:d}\nbrch_Ipre =".format(self.rank) + str(self.brch_Ipre)
            # print(tp)

            tl_19 = time.time()

            self.helics_update()

            tl_20 = time.time()

            Nsteps += 1

            t_evnt += tl_1 - tl_0
            t_pred += tl_2 - tl_1
            t_upig += tl_3 - tl_2
            t_upir += tl_4 - tl_3
            t_uper += tl_5 - tl_4
            t_upil += tl_6 - tl_5
            t_plbk += tl_7 - tl_6
            t_helc += tl_8 - tl_7
            t_rein += tl_9 - tl_8
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

            self.comm.Barrier()

        #### END TIME LOOP ####

        # if save_snapshot:
        #     if tn not in self.v:
        #         self.save(tn)
        #     if rank == 0:
        #         self.dump_res(pfd, dyd, ini,
        #                      save_snapshot_mode,
        #                      output_snp_ful,
        #                      output_snp_1pt,
        #                      output_res
        #                      )

        if rank == 0:
            t_stop = time.time()

            print("**** Solver Timing ****")
            self.mpi_lu.print_timing()

            # self.print_timing()

            if Nrein == 0:
                avg_re_init = 0.0
            else:
                avg_re_init = t_rein/Nrein
            ## End if

            t_sync = self.solve_sync_time
            t_reord = self.solve_reorder_time
            t_schur = self.solve_schur_time

            init = self.init_time
            bbd = self.bbd_time
            mpi_setup = self.mpi_setup_time
            nmb_comp = self.nmb_comp_time
            factor = self.bbd_lu_time
            loop = t_stop - t_start
            elapsed = init + bbd + mpi_setup + nmb_comp + factor + loop

            timing_string = """**** Timing Info ****
Dimension:   {:8d}    NNZ:   {:12d}
Init:        {:10.2e} {:8.2%}
BBD Form:    {:10.2e} {:8.2%}
MPI Setup:   {:10.2e} {:8.2%}
Nmb Comp:    {:10.2e} {:8.2%}
LU Factor:   {:10.2e} {:8.2%}
Loop:        {:10.2e} {:8.2%} {:8d} {:8.2e}
  Event:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  PredX:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdIG:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdIR:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdER:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdIL:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  ReInt:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  Solve:     {:10.2e} {:8.2%} {:8d} {:8.2e}
    SyncI:   {:10.2e} {:8.2%} {:8d} {:8.2e}
    Reord:   {:10.2e} {:8.2%} {:8d} {:8.2e}
    Schur:   {:10.2e} {:8.2%} {:8d} {:8.2e}
  BusMe:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdX:      {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdXR:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdXL:     {:10.2e} {:8.2%} {:8d} {:8.2e}
  Save:      {:10.2e} {:8.2%} {:8d} {:8.2e}
  Phasor:    {:10.2e} {:8.2%} {:8d} {:8.2e}
  Helics:    {:10.2e} {:8.2%} {:8d} {:8.2e}
  UpdIH:     {:10.2e} {:8.2%} {:8d} {:8.2e}
Total:       {:10.2e}

            """.format(self.BBD_0.shape[0], self.BBD_0.nnz,
                       init,         init/elapsed,
                       bbd,          bbd/elapsed,
                       mpi_setup,    mpi_setup/elapsed,
                       nmb_comp,     nmb_comp/elapsed,
                       factor,       factor/elapsed,
                       loop,         loop/elapsed,      Nsteps,       loop/Nsteps,
                       t_evnt,       t_evnt/elapsed,    Nsteps,       t_evnt/Nsteps,
                       t_pred,       t_pred/elapsed,    Nsteps,       t_pred/Nsteps,
                       t_upig,       t_upig/elapsed,    Nsteps,       t_upig/Nsteps,
                       t_upir,       t_upir/elapsed,    Nsteps,       t_upir/Nsteps,
                       t_uper,       t_uper/elapsed,    Nsteps,       t_uper/Nsteps,
                       t_upil,       t_upil/elapsed,    Nsteps,       t_upil/Nsteps,
                       t_rein,       t_rein/elapsed,    Nrein,        avg_re_init,
                       t_solve,      t_solve/elapsed,   Nsteps,       t_solve/Nsteps,
                       t_sync,       t_sync/elapsed,    Nsteps,       t_sync/Nsteps,
                       t_reord,      t_reord/elapsed,   Nsteps,       t_reord/Nsteps,
                       t_schur,      t_schur/elapsed,   Nsteps,       t_schur/Nsteps,
                       t_busmea,     t_busmea/elapsed,  Nsteps,       t_busmea/Nsteps,
                       t_upx,        t_upx/elapsed,     Nsteps,       t_upx/Nsteps,
                       t_upxr,       t_upxr/elapsed,    Nsteps,       t_upxr/Nsteps,
                       t_upxl,       t_upxl/elapsed,    Nsteps,       t_upxl/Nsteps,
                       t_save,       t_save/elapsed,    Nsteps,       t_save/Nsteps,
                       t_phsr,       t_phsr/elapsed,    Nsteps,       t_phsr/Nsteps,
                       t_helc,       t_helc/elapsed,    Nsteps,       t_helc/Nsteps,
                       t_upih,       t_upih/elapsed,    Nsteps,       t_upih/Nsteps,
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

        #### END IF RANK == 0 ####

        return


    def dump_res(self,
                 snapshot_mode,
                 output_snp_ful,
                 output_snp_1pt,
                 output_res,
                 ):

        self.comm = None
        self.mpi_lu = None
        self.index_order = None
        self.inv_order = None
        self.mpi_lu_0 = None
        self.index_order_0 = None
        self.inv_order_0 = None

        retval = super().dump_res(snapshot_mode,
                                  output_snp_ful,
                                  output_snp_1pt,
                                  output_res)

        return retval


    def mpi_print(self, to_print, root=0):
        return mpi_print(to_print, self.comm, root=root)


    def mpi_distribute_obj(self, objects, root=0):
        return mpi_distribute_obj(objects, self.comm, root=root)


    def mpi_sync_vect(self, vector, root):
        return mpi_sync_vect(vector, self.comm, root=root)


    def save_vect(self, storage, vector, key):
        return save_vect(self, storage, vector, key)


    def mpi_sync_and_save(self, storage, vector, key, root):
        return mpi_sync_and_save(storage, vector, key, self.comm, root=root)


    def sync_brch_Ipre(self):

        if self.brch_ipre_need_sync:

            recv_buf = [self.brch_Ipre,
                        self.brch_counts,
                        self.brch_range[0,:],
                        MPI.DOUBLE
                        ]

            sidx = self.brch_range[0,self.rank]
            eidx = self.brch_range[1,self.rank]
            send_buf = self.brch_Ipre[sidx:eidx]

            self.comm.Allgatherv(send_buf, recv_buf)

            self.brch_ipre_need_sync = False

        return


    def _divide_range(self, n, size):

        (num, rem) = divmod(n, size)

        mpi_counts = np.array([num]*size)
        mpi_counts[:rem] += 1

        mpi_range = np.zeros((2,size), dtype=int)
        bdrs = np.cumsum(mpi_counts)
        mpi_range[0,1:] = bdrs[:-1]
        mpi_range[1,:] = bdrs[:]

        return (mpi_range, mpi_counts)
