
import numpy as np

# states class
class States():
    def __init__(self, ngen):
        self.pv_dt_1 = np.zeros(ngen)
        self.pv_dt_2 = np.zeros(ngen)
        self.pv_dt_3 = np.zeros(ngen)
        self.pd_dt = np.zeros(ngen)
        self.nx_dt = np.zeros(ngen)

        self.pv_w_1 = np.zeros(ngen)
        self.pv_w_2 = np.zeros(ngen)
        self.pv_w_3 = np.zeros(ngen)
        self.pd_w = np.zeros(ngen)
        self.nx_w = np.zeros(ngen)

        self.pv_id_1 = np.zeros(ngen)
        self.pv_id_2 = np.zeros(ngen)
        self.pv_id_3 = np.zeros(ngen)
        self.pd_id = np.zeros(ngen)
        self.nx_id = np.zeros(ngen)

        self.pv_iq_1 = np.zeros(ngen)
        self.pv_iq_2 = np.zeros(ngen)
        self.pv_iq_3 = np.zeros(ngen)
        self.pd_iq = np.zeros(ngen)
        self.nx_iq = np.zeros(ngen)

        self.pv_ifd_1 = np.zeros(ngen)
        self.pv_ifd_2 = np.zeros(ngen)
        self.pv_ifd_3 = np.zeros(ngen)
        self.nx_ifd = np.zeros(ngen)

        self.pv_i1d_1 = np.zeros(ngen)
        self.pv_i1d_2 = np.zeros(ngen)
        self.pv_i1d_3 = np.zeros(ngen)
        self.nx_i1d = np.zeros(ngen)

        self.pv_i1q_1 = np.zeros(ngen)
        self.pv_i1q_2 = np.zeros(ngen)
        self.pv_i1q_3 = np.zeros(ngen)
        self.nx_i1q = np.zeros(ngen)

        self.pv_i2q_1 = np.zeros(ngen)
        self.pv_i2q_2 = np.zeros(ngen)
        self.pv_i2q_3 = np.zeros(ngen)
        self.nx_i2q = np.zeros(ngen)

        self.pv_ed_1 = np.zeros(ngen)
        self.pv_ed_2 = np.zeros(ngen)
        self.pv_ed_3 = np.zeros(ngen)
        self.nx_ed = np.zeros(ngen)

        self.pv_eq_1 = np.zeros(ngen)
        self.pv_eq_2 = np.zeros(ngen)
        self.pv_eq_3 = np.zeros(ngen)
        self.nx_eq = np.zeros(ngen)

        self.pv_psyd_1 = np.zeros(ngen)
        self.pv_psyd_2 = np.zeros(ngen)
        self.pv_psyd_3 = np.zeros(ngen)
        self.nx_psyd = np.zeros(ngen)

        self.pv_psyq_1 = np.zeros(ngen)
        self.pv_psyq_2 = np.zeros(ngen)
        self.pv_psyq_3 = np.zeros(ngen)
        self.nx_psyq = np.zeros(ngen)

        self.pv_psyfd_1 = np.zeros(ngen)
        self.pv_psyfd_2 = np.zeros(ngen)
        self.pv_psyfd_3 = np.zeros(ngen)
        self.nx_psyfd = np.zeros(ngen)

        self.pv_psy1q_1 = np.zeros(ngen)
        self.pv_psy1q_2 = np.zeros(ngen)
        self.pv_psy1q_3 = np.zeros(ngen)
        self.nx_psy1q = np.zeros(ngen)

        self.pv_psy1d_1 = np.zeros(ngen)
        self.pv_psy1d_2 = np.zeros(ngen)
        self.pv_psy1d_3 = np.zeros(ngen)
        self.nx_psy1d = np.zeros(ngen)

        self.pv_psy2q_1 = np.zeros(ngen)
        self.pv_psy2q_2 = np.zeros(ngen)
        self.pv_psy2q_3 = np.zeros(ngen)
        self.nx_psy2q = np.zeros(ngen)

        self.pv_te_1 = np.zeros(ngen)
        self.pv_te_2 = np.zeros(ngen)
        self.pv_te_3 = np.zeros(ngen)
        self.nx_te = np.zeros(ngen)

        self.pv_i_d_1 = np.zeros((3,ngen))
        self.pv_i_d_2 = np.zeros(ngen)
        self.pv_i_d_3 = np.zeros(ngen)

        self.pv_i_q_1 = np.zeros((3,ngen))
        self.pv_i_q_2 = np.zeros(ngen)
        self.pv_i_q_3 = np.zeros(ngen)

        self.pv_u_d_1 = np.zeros(ngen)
        self.pv_u_d_2 = np.zeros(ngen)
        self.pv_u_d_3 = np.zeros(ngen)
        self.pd_u_d = np.zeros(ngen)

        self.pv_u_q_1 = np.zeros(ngen)
        self.pv_u_q_2 = np.zeros(ngen)
        self.pv_u_q_3 = np.zeros(ngen)
        self.pd_u_q = np.zeros(ngen)

        self.pv_his_d_1 = np.zeros(ngen)
        self.pv_his_fd_1 = np.zeros(ngen)
        self.pv_his_1d_1 = np.zeros(ngen)
        self.pv_his_q_1 = np.zeros(ngen)
        self.pv_his_1q_1 = np.zeros(ngen)
        self.pv_his_2q_1 = np.zeros(ngen)
        self.pv_his_red_d_1 = np.zeros(ngen)
        self.pv_his_red_q_1 = np.zeros(ngen)

        # EXC
        self.pv_EFD_1 = np.zeros(ngen)
        self.pv_EFD_2 = np.zeros(ngen)
        self.pv_EFD_3 = np.zeros(ngen)
        self.pd_EFD = np.zeros(ngen)
        self.nx_EFD = np.zeros(ngen)

        # SEXS
        self.pv_v1_1 = np.zeros(ngen)
        self.pv_v1_2 = np.zeros(ngen)
        self.pv_v1_3 = np.zeros(ngen)
        self.nx_v1 = np.zeros(ngen)

        # GOV
        self.pv_pm_1 = np.zeros(ngen)
        self.pv_pm_2 = np.zeros(ngen)
        self.pv_pm_3 = np.zeros(ngen)
        self.nx_pm = np.zeros(ngen)

        # TGOV1
        self.pv_p1_1 = np.zeros(ngen)
        self.pv_p1_2 = np.zeros(ngen)
        self.pv_p1_3 = np.zeros(ngen)
        self.nx_p1 = np.zeros(ngen)

        self.pv_p2_1 = np.zeros(ngen)
        self.pv_p2_2 = np.zeros(ngen)
        self.pv_p2_3 = np.zeros(ngen)
        self.nx_p2 = np.zeros(ngen)

        # HYGOV
        self.pv_xe_1 = np.zeros(ngen)
        self.pv_xe_2 = np.zeros(ngen)
        self.pv_xe_3 = np.zeros(ngen)

        self.pv_xc_1 = np.zeros(ngen)
        self.pv_xc_2 = np.zeros(ngen)
        self.pv_xc_3 = np.zeros(ngen)

        self.pv_xg_1 = np.zeros(ngen)
        self.pv_xg_2 = np.zeros(ngen)
        self.pv_xg_3 = np.zeros(ngen)

        self.pv_xq_1 = np.zeros(ngen)
        self.pv_xq_2 = np.zeros(ngen)
        self.pv_xq_3 = np.zeros(ngen)

        # GAST
        self.pv_p1_1 = np.zeros(ngen)
        self.pv_p1_2 = np.zeros(ngen)
        self.pv_p1_3 = np.zeros(ngen)

        self.pv_p2_1 = np.zeros(ngen)
        self.pv_p2_2 = np.zeros(ngen)
        self.pv_p2_3 = np.zeros(ngen)

        self.pv_p3_1 = np.zeros(ngen)
        self.pv_p3_2 = np.zeros(ngen)
        self.pv_p3_3 = np.zeros(ngen)

        # IEEEST
        self.pv_y1_1 = np.zeros(ngen)
        self.pv_y2_1 = np.zeros(ngen)
        self.pv_y3_1 = np.zeros(ngen)
        self.pv_y4_1 = np.zeros(ngen)
        self.pv_y5_1 = np.zeros(ngen)
        self.pv_y6_1 = np.zeros(ngen)
        self.pv_y7_1 = np.zeros(ngen)
        self.pv_x1_1 = np.zeros(ngen)
        self.pv_x2_1 = np.zeros(ngen)
        self.pv_vs_1 = np.zeros(ngen)

        return
