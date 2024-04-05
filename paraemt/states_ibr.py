
import numpy as np

# states class
class States_ibr():
    def __init__(self, nibr):
        # IBR
        self.nx_freq = np.zeros(nibr)

        # IBR - regca
        self.nx_regca_s0 = np.zeros(nibr)
        self.nx_regca_s1 = np.zeros(nibr)
        self.nx_regca_s2 = np.zeros(nibr)

        self.nx_regca_Vmp = np.zeros(nibr)
        self.nx_regca_Vap = np.zeros(nibr)
        self.nx_regca_i1 = np.zeros(nibr)
        self.nx_regca_i2 = np.zeros(nibr)
        self.nx_regca_ip2rr = np.zeros(nibr)

        # IBR - reecb
        self.nx_reecb_s0 = np.zeros(nibr)
        self.nx_reecb_s1 = np.zeros(nibr)
        self.nx_reecb_s2 = np.zeros(nibr)
        self.nx_reecb_s3 = np.zeros(nibr)
        self.nx_reecb_s4 = np.zeros(nibr)
        self.nx_reecb_s5 = np.zeros(nibr)

        self.nx_reecb_Ipcmd = np.zeros(nibr)
        self.nx_reecb_Iqcmd = np.zeros(nibr)
        self.nx_reecb_Pref = np.zeros(nibr)
        self.nx_reecb_Qext = np.zeros(nibr)
        self.nx_reecb_q2vPI = np.zeros(nibr)
        self.nx_reecb_v2iPI = np.zeros(nibr)

        # IBR - repca
        self.nx_repca_s0 = np.zeros(nibr)
        self.nx_repca_s1 = np.zeros(nibr)
        self.nx_repca_s2 = np.zeros(nibr)
        self.nx_repca_s3 = np.zeros(nibr)
        self.nx_repca_s4 = np.zeros(nibr)
        self.nx_repca_s5 = np.zeros(nibr)
        self.nx_repca_s6 = np.zeros(nibr)

        self.nx_repca_Vref = np.zeros(nibr)
        self.nx_repca_Qref = np.zeros(nibr)
        self.nx_repca_Freq_ref = np.zeros(nibr)
        self.nx_repca_Plant_pref = np.zeros(nibr)
        self.nx_repca_LineMW = np.zeros(nibr)
        self.nx_repca_LineMvar = np.zeros(nibr)
        self.nx_repca_LineMVA = np.zeros(nibr)
        self.nx_repca_QVdbout = np.zeros(nibr)
        self.nx_repca_fdbout = np.zeros(nibr)
        self.nx_repca_Pref_out = np.zeros(nibr)
        self.nx_repca_vq2qPI = np.zeros(nibr)
        self.nx_repca_p2pPI = np.zeros(nibr)

        # IBR - PLL
        self.nx_pll_ze = np.zeros(nibr)
        self.nx_pll_de = np.zeros(nibr)
        self.nx_pll_we = np.zeros(nibr)

        return
