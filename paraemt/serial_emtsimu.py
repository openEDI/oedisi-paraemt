import numpy as np
import time

from paraemt.emtsimu import EmtSimu
from paraemt.psutils import initialize_emt_from_file, modify_system


class SerialEmtSimu(EmtSimu):
    @staticmethod
    def initialize_from_snp(input_snp, netMod): # This is used for OEDI task
        emt = super(SerialEmtSimu, SerialEmtSimu).initialize_from_snp(input_snp, netMod)
        emt.init_ibr_epri()
        return emt

    def __init__(
        self,
        pfd_name='',
        dyd_name='',
        systemN=1,
        EMT_N=0,
        N_row=1,
        N_col=1,
        ts=50e-6,
        Tlen=0.2,
        kts=round(1 / 60 / 4 / 50e-6),
        stepk=round(1 / 60 / 4 / 50e-6) - 1,
        save_rate=1,
        netMod="lu",
        loadmodel_option=1,
        record4cosim=False,
        playback_enable=True,
        workingfolder='',
        Gd=100,
        Go=0.001,
    ):
        t0 = time.time()

        (pfd, ini, dyd, emt_zones) = initialize_emt_from_file(
            workingfolder,
            pfd_name,
            dyd_name,
            N_row,
            N_col,
            ts,
            netMod,
            loadmodel_option,
            record4cosim,
        )

        super().__init__(
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

        bus_rec, current_rec, voltage_rec = self.Record4CoSim(
            record4cosim, self.ini.Init_brch_Ipre, self.Vsol, 0.0
        )
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
        self.workingfolder=workingfolder # Added, Min

        t1 = time.time()

        self.init_time = t1 - t0

    def presolveV(self):
        self.Vsol_1 = self.Vsol

    def solveV(self, admittance_mode, Igs, Igi, node_Ihis, Il):
        # self.Vsol_1 = self.Vsol

        I_RHS = Igs + Igi + node_Ihis

        if self.loadmodel_option == 2 and Il is not None:
            I_RHS += Il

        if admittance_mode == "inv":
            Vsol = self.Ginv @ I_RHS

        elif admittance_mode == "lu":
            Vsol = self.Glu.solve(I_RHS)

        elif admittance_mode == "bbd":
            # TODO: Make this a thing...
            # self.Vsol = self.Gbbd.schur_solve(self.I_RHS)

            pass

        else:
            raise ValueError("Unrecognized mode: {}".format(admittance_mode))

        return Vsol

    def calc_int_i(self, ii, vv, bb):
        n = bb.shape[1]
        nt = ii.shape[0]

        assert ii.shape == vv.shape
        assert ii.shape == (nt, 3 * n + 1)
        assert bb.shape == (1, n)

        i_int = np.zeros((nt, 3 * n))

        Gd = self.Gd
        Go = self.Go
        G = np.array([[Gd, Go, Go], [Go, Gd, Go], [Go, Go, Gd]])

        for k in range(n):
            itemp = ii[:, 3 * k + 1 : 3 * (k + 1) + 1]
            vtemp = vv[:, 3 * k + 1 : 3 * (k + 1) + 1]

            i_int[:, 3 * k : 3 * (k + 1)] = np.transpose(
                G @ np.transpose(vtemp) + np.transpose(itemp)
            )

        ## End for

        return i_int
