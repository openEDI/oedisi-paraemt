import numpy as np
import pandas as pd
import os
import time
import json
import logging
from paraemt.serial_emtsimu import SerialEmtSimu
from datetime import datetime

import helics as h
from oedisi.types.common import BrokerConfig
from pydantic import BaseModel

from ctypes import *  # For DLL
from paraemt.Python_C_DLL_wrapper import wrap_function, MODELINSTANCE, MODELINFO

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class ParaemtConfig(BaseModel):
    """Pydantic model for the static configuration of paraemt."""
    name: str
    # Configuration in JSON format
    pfd_file: str
    dyd_file: str
    systemN: int 
    SimMod: int  # 0 - Standard Initialization, 1 - Initialize from Snapshot
    save_snapshot: bool  # Save snapshot at end of run or not
    save_snapshot_mode: (
        int  # 0 - 1 point snapshot, 1 - whole simulation snapshot, 2 - both
    )
    DSrate: int  # down sampling rate, i.e. results saved every DSrate sim steps.
    compute_phasor: (
        int  # Save to CSVs intermediate calculation results--useful for debugging
    )
    dump_all: bool  # Save results to csv files
    save_results_csv: bool  # Display simulation progress
    show_progress: bool  # recording for pseudo co-sim
    record4cosim: bool  # playback or helics for co-sim: True: play recorded signals, False: read signals thru helics
    playback_enable: bool  # replay voltage phasors at all buses
    playback_voltphasor: bool
    EMT_N: int
    N_row: int  # haven't tested the mxn layout, so plz don't set N_row/N_col to other nums.
    N_col: int
    use_helics: bool
    ts: float  # time step, second, 50.05005005
    kts: int  # phasor step = kts * EMT steps
    stepk: int  # k-th EMT step within one phasor step, initialized at kts-1
    t_release_f: float
    loadmodel_option: int  # 1-const rlc, 2-const z
    netMod: str  # interface R for pseudo co-sim
    Gd: float
    Go: float
    Tlen: float  # total simulation time length, second
    # ctrl step change
    t_sc: float
    i_gen_sc: int
    flag_exc_gov: int  # 0 - exc, 1 - gov
    dsp: float
    flag_sc: int
    # gen trip
    t_gentrip: float
    i_gentrip: int  # 13: 1431-N, 15: 2032 G
    flag_gentrip: int
    flag_reinit: int

    # line fault
    # fault_t = 1000.0
    # fault_line_idx = 0
    # fault_tlen = 5/60 # 5 cycles
    # fault_dist = 0.01 # percntage
    # fault_type = 11
    # fault_r = [30, 30, 30, 0.001, 0.001, 0.001]
    # fault_tripline = 0


# def initialize_emt(config: ParaemtConfig):
#     # ==========================================================================
#     # Read the JOSON configuuration of EMT simulation
#     # ParaEMT initialization
#     shared_lib_path = os.getcwd() + "\\models\\ibrepri.dll"  # EPRI's IBR model
#     add_lib = CDLL(shared_lib_path)
#     Model_GetInfo = wrap_function(add_lib, "Model_GetInfo", POINTER(MODELINFO), None)
#     Model_Outputs = wrap_function(
#         add_lib, "Model_Outputs", c_int, [POINTER(MODELINSTANCE)]
#     )
#     info = Model_GetInfo()
#     num_in_ports, num_out_ports, num_param = (
#         info.contents.cNumInputPorts,
#         info.contents.cNumOutputPorts,
#         info.contents.cNumParameters,
#     )
#     num_int_states, num_float_states, num_double_states = (
#         info.contents.cNumIntStates,
#         info.contents.cNumFloatStates,
#         info.contents.cNumDoubleStates,
#     )

#     sim_info = """
#     ---- Sim Info ----
#     System: {}, {}
#     N Rows: {:d}
#     N Cols: {:d}
#     Time Step: {:.8e}
#     Time End: {:e}
#     Network Solve Mode: {:s}
#     """.format(
#         config.pdf_file,
#         config.dyd_file,
#         config.N_row,
#         config.N_col,
#         config.ts,
#         config.Tlen,
#         config.netMod,
#     )
#     print(sim_info)
#     # TODO: Test and debug snapshot stuff
#     input_snp = (
#         "sim_snp_S" + str(config.name) + "_" + str(int(config.ts * 1e6)) + "u_1pt.pkl"
#     )
#     t0 = time.time()
#     if config.SimMod == 0:
#         emt = SerialEmtSimu(
#             workingfolder=os.getcwd(),
#             systemN=config.systemN,
#             EMT_N=config.EMT_N,
#             N_row=config.N_row,
#             N_col=config.N_col,
#             ts=config.ts,
#             Tlen=config.Tlen,
#             kts=config.kts,
#             stepk=config.stepk,
#             save_rate=config.DSrate,
#             netMod=config.netMod,
#             loadmodel_option=config.loadmodel_option,
#             record4cosim=config.record4cosim,
#             playback_enable=config.playback_enable,
#             Gd=config.Gd,
#             Go=config.Go,
#         )
#     else:
#         print("Loading snapshot file: ", input_snp)
#         emt = SerialEmtSimu.initialize_from_snp(input_snp, config.netMod)
#     emt.compute_phasor = config.compute_phasor
#     emt.tsat_gen_omg = [0]
#     emt.tsat_gen_maci = [0]
#     emt.tsat_t = [0]
#     emt.Tlen = config.Tlen
#     # ctrl step change
#     emt.t_sc = config.t_sc
#     emt.i_gen_sc = config.i_gen_sc
#     emt.flag_exc_gov = config.flag_exc_gov
#     emt.dsp = config.dsp
#     emt.flag_sc = config.flag_sc
#     # gen trip
#     emt.t_gentrip = config.t_gentrip
#     emt.i_gentrip = config.i_gentrip
#     emt.flag_gentrip = config.flag_gentrip
#     emt.flag_reinit = config.flag_reinit
#     emt.t_release_f = config.t_release_f
#     emt.loadmodel_option = config.loadmodel_option
#     return emt


class ParaemtFederate:
    def __init__(
        self,
        config: ParaemtConfig,
        input_mapping: dict[str, str], # TODO component_definition? Necessary?
        broker_config: BrokerConfig,
    ):
        """Paraemt HELICS federate with oedisi

        Parameters
        ----------
        config : TestFederateConfig
            Static configuration including the federate name and input needed at startup.
        input_mapping : dict[str,str]
            Maps any dynamic inputs to helics subscription keys. In this example,
            there are none.
        broker_config : BrokerConfig
            Configures the helics broker ip and port. Default is 127.0.0.1:23404.
        """
        self.emt = self.initialize_emt(config)

        # Create HELICS Federate object that describes the federate properties
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = config.name  # Sets HELICS name
        fedinfo.core_type = (
            h.HELICS_CORE_TYPE_ZMQ
        )  # OEDISI simulations use ZMQ transport layer
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        # Maximum time resolution. Time unit may depend on simulation type.
        h.helicsFederateInfoSetTimeProperty(
            fedinfo,
            h.helics_property_time_delta,
            config.DSrate * config.ts,  # 10 microseconds
        )

        self.vfed = h.helicsCreateValueFederate(config.name, fedinfo)
        logger.info("Value federate created")

        # This should match the dynamic output in component_definition.json
        self.pub_Pinj = self.vfed.register_publication(
            "emt_Pinj", h.HELICS_DATA_TYPE_DOUBLE, ""
        )
        self.pub_Qinj = self.vfed.register_publication(
            "emt_Qinj", h.HELICS_DATA_TYPE_DOUBLE, ""
        )
        self.pub_V = self.vfed.register_publication(
            "emt_Vsol", h.HELICS_DATA_TYPE_STRING, ""
        )
    
    # TODO to confirm, moved here
    # def initialize_emt(config: ParaemtConfig):
    def initialize_emt(self, config): # TODO to confirm, Min added self
        # ==========================================================================
        # Read the JOSON configuuration of EMT simulation
        # ParaEMT initialization
        shared_lib_path = os.getcwd() + "\\models\\ibrepri.dll"  # EPRI's IBR model
        add_lib = CDLL(shared_lib_path)
        Model_GetInfo = wrap_function(add_lib, "Model_GetInfo", POINTER(MODELINFO), None)
        Model_Outputs = wrap_function(
            add_lib, "Model_Outputs", c_int, [POINTER(MODELINSTANCE)]
        )
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

        sim_info = """
        ---- Sim Info ----
        System: {}, {}
        N Rows: {:d}
        N Cols: {:d}
        Time Step: {:.8e}
        Time End: {:e}
        Network Solve Mode: {:s}
        """.format(
            config.pfd_file,
            config.dyd_file,
            config.N_row,
            config.N_col,
            config.ts,
            config.Tlen,
            config.netMod,
        )
        print(sim_info)
        # TODO: Test and debug snapshot stuff
        input_snp = (
            "sim_snp_S" + str(config.name) + "_" + str(int(config.ts * 1e6)) + "u_1pt.pkl"
        )
        if config.SimMod == 0:
            self.emt = SerialEmtSimu(
                workingfolder=os.getcwd(),
                pfd_name=config.pfd_file,
                dyd_name=config.dyd_file,
                systemN=config.systemN,
                EMT_N=config.EMT_N,
                N_row=config.N_row,
                N_col=config.N_col,
                ts=config.ts,
                Tlen=config.Tlen,
                kts=config.kts,
                stepk=config.stepk,
                save_rate=config.DSrate,
                netMod=config.netMod,
                loadmodel_option=config.loadmodel_option,
                record4cosim=config.record4cosim,
                playback_enable=config.playback_enable,
                Gd=config.Gd,
                Go=config.Go,
            )
        else:
            print("Loading snapshot file: ", input_snp)
            self.emt = SerialEmtSimu.initialize_from_snp(input_snp, config.netMod)
        self.save_rate=config.DSrate
        self.emt.compute_phasor = config.compute_phasor
        self.emt.tsat_gen_omg = [0]
        self.emt.tsat_gen_maci = [0]
        self.emt.tsat_t = [0]
        self.emt.Tlen = config.Tlen
        # ctrl step change
        self.emt.t_sc = config.t_sc
        self.emt.i_gen_sc = config.i_gen_sc
        self.emt.flag_exc_gov = config.flag_exc_gov
        self.emt.dsp = config.dsp
        self.emt.flag_sc = config.flag_sc
        # gen trip
        self.emt.t_gentrip = config.t_gentrip
        self.emt.i_gentrip = config.i_gentrip
        self.emt.flag_gentrip = config.flag_gentrip
        self.emt.flag_reinit = config.flag_reinit
        self.emt.t_release_f = config.t_release_f
        self.emt.loadmodel_option = config.loadmodel_option
        return self.emt

    def run(self, config):
        """Run HELICS simulation until completion."""
        # Starts HELICS simulation. Essentially says to broker "This simulation is ready"
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")
        # We request the time we want
        h.helicsFederateRequestTime(self.vfed, 0.0)

        tn = 0  # number of time time steps    Min added
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
        ts = config.ts
        t1 = time.time()
        while tn * ts < config.Tlen:
            # logger.info("start time: " + str(datetime.now()))
            # ==========================================================================
            # Run Paraself.emt, Only the time loop function
            tn += 1
            if config.show_progress:  # Print the simulation progress
                if tn > 1:
                    if np.mod(tn, 200) == 0: 
                        logger.info("start time: " + str(datetime.now()))  # TODO, confirm, move here okay? necessary?
                        print("%.4f" % self.emt.t[-1])
            tl_0 = time.time()
            flag_reini = 0
            self.emt.StepChange(self.emt.dyd, self.emt.ini, tn)  # configure step change in exc or gov references
            if (config.flag_gentrip == 0 and config.flag_reinit == 1):  # If the generator is tripped
                flag_reini = 1
            # TODO, add fault code here in future
            self.emt.Ginv = self.emt.ini.Init_net_G0  
            self.emt.net_coe = self.emt.ini.Init_net_coe0
            self.emt.Glu = self.emt.ini.Init_net_G0_lu

            tl_1 = time.time()
            self.emt.predictX(self.emt.pfd, self.emt.dyd, ts)
            tl_2 = time.time()
            self.emt.updateIg(self.emt.pfd, self.emt.dyd, self.emt.ini)
            tl_3 = time.time()
            self.emt.updateIibr(self.emt.pfd, self.emt.dyd, self.emt.ini)
            tl_4 = time.time()
            self.emt.updateIibr_epri(self.emt.pfd, self.emt.dyd, self.emt.ini, tn)
            tl_5 = time.time()
            self.emt.updateIl(self.emt.pfd, self.emt.dyd, tn)  # update current injection from load
            tl_6 = time.time()
            tl_7 = time.time()
            self.emt.GenTrip(self.emt.pfd, self.emt.dyd, self.emt.ini, tn)  # configure generation trip
            tl_8 = time.time()
            # re-init
            if flag_reini == 1:
                self.emt.Re_Init(self.emt.pfd, self.emt.dyd, self.emt.ini, tn)
            tl_9 = time.time()
            self.emt.presolveV()
            self.emt.Vsol = self.emt.solveV(
                self.emt.ini.admittance_mode,
                self.emt.Igs,
                self.emt.Igi + self.emt.Igi_epri,
                self.emt.node_Ihis,
                self.emt.Il,
            )
            tl_10 = time.time()
            self.emt.BusMea(self.emt.pfd, self.emt.dyd, tn)  # bus measurement
            tl_11 = time.time()
            self.emt.updateX(
                self.emt.pfd, self.emt.dyd, self.emt.ini, tn, config.playback_voltphasor
            )
            tl_12 = time.time()
            self.emt.updateXibr(self.emt.pfd, self.emt.dyd, self.emt.ini)
            tl_13 = time.time()
            self.emt.updateXl(self.emt.pfd, self.emt.dyd, tn)
            tl_14 = time.time()
            tl_15 = time.time()
            tl_16 = time.time()
            self.emt.updateIhis(self.emt.ini)
            tl_17 = time.time()
            tl_18 = time.time()
            if tn % self.save_rate == 0:
                # save has to be placed after updateIhis to ensure time alignment for recorded
                #     signals for pseudo co-sim
                self.emt.save(tn)
                self.pub_Pinj.publish(float(self.emt.Pinj))
                self.pub_Qinj.publish(float(self.emt.Qinj))
                # self.pub_example.publish(
                # If possible, use either basic types available like floats, ints, etc, or types provided
                # by the oedisi.types.data_types module.
                # Any indexing information should have appropriate labels if necessary.
                # VoltageArray(values=[0.0, 1.0, 2.0], ids=["node1", "node2", "node3"])
                #    self.emt.Vsol,  # Three phase voltage waveform
                #    self.emt.brch_Ipre,  # Three phase current waveform
                # )
                _ = h.helicsFederateRequestTime(self.vfed, tn * ts)

            tl_19 = time.time()

            t_evnt += tl_1 - tl_0
            t_pred += tl_2 - tl_1
            t_upig += tl_3 - tl_2
            t_upir += tl_4 - tl_3
            t_uper += tl_5 - tl_4
            t_upil += tl_6 - tl_5
            t_helc += tl_7 - tl_6
            t_evnt += tl_8 - tl_7
            t_rent += tl_9 - tl_8
            t_solve += tl_10 - tl_9
            t_busmea += tl_11 - tl_10
            t_upx += tl_12 - tl_11
            t_upxr += tl_13 - tl_12
            t_upxl += tl_14 - tl_13
            t_phsr += tl_15 - tl_14
            t_helc += tl_16 - tl_15
            t_upih += tl_17 - tl_16
            t_save += tl_18 - tl_17
            t_helc += tl_19 - tl_18
            Nsteps += 1

        # ==========================================================================
        # Save simulation results locally
        save_results_csv = config.save_results_csv
        if save_results_csv:
            print("Saving results...")
            df_v = pd.DataFrame(self.emt.v).T
            df_v.to_csv("paraemt.emt_v.csv")
            df_x = pd.DataFrame(self.emt.x).T
            df_x.to_csv("paraemt.emt_x.csv")
            df_ibr = pd.DataFrame(self.emt.x_ibr).T
            df_ibr.to_csv("paraemt.emt_ibr.csv")
            df_ebr = pd.DataFrame(self.emt.x_ibr_epri).T
            df_ebr.to_csv("paraemt.emt_ebr.csv")
            df_bus = pd.DataFrame(self.emt.x_bus).T
            df_bus.to_csv("paraemt.emt_bus.csv")
            df_load = pd.DataFrame(self.emt.x_load).T
            df_load.to_csv("paraemt.emt_load.csv")
        # Save simulation snapshot locally
        output_snp_ful = (
            "sim_snp_S" + str(config.systemN) + "_" + str(int(ts * 1e6)) + "u.pkl"
        )
        output_snp_1pt = (
            "sim_snp_S" + str(config.systemN) + "_" + str(int(ts * 1e6)) + "u_1pt.pkl"
        )
        output_res = (
            "sim_res_S" + str(config.systemN) + "_" + str(int(ts * 1e6)) + "u.pkl"
        )
        if config.save_snapshot:
            print("Saving Snapshot...")
            self.emt.dump_res(
                config.SimMod,
                config.save_snapshot_mode,
                output_snp_ful,
                output_snp_1pt,
                output_res,
            )
        # ==========================================================================

        # ==========================================================================
        # Print time cost information of self.emt simulation
        t2 = time.time()
        numba_comp = 0
        loop = t2 - t1
        elapsed = numba_comp + loop + self.emt.init_time

        names = [
            "Loop",
            "Event",
            "PredX",
            "UpdIG",
            "UpdIR",
            "UpdER",
            "UpdIL",
            "ReInit",
            "Solve",
            "BusMea",
            "UpdX",
            "UpdXR",
            "UpdXL",
            "Save",
            "UpdIH",
            "Phasor",
            "Helics",
        ]
        reported_variables = [
            loop,
            t_evnt,
            t_pred,
            t_upig,
            t_upir,
            t_uper,
            t_upil,
            t_rent,
            t_solve,
            t_busmea,
            t_upx,
            t_upxr,
            t_upxl,
            t_save,
            t_upih,
            t_phsr,
            t_helc,
        ]
        variable_timing_string = "\n".join(
            f"{name:13s}: {time:10.2e} {time / elapsed:8.2%} {Nsteps:8d} {time / Nsteps:8.2e}"
            for name, time in zip(names, reported_variables)
        )

        timing_string = f"""**** Timing Info ****

        Dimension:   {self.emt.ini.Init_net_G0_inv.shape[0]:8d}
        Init:        {self.emt.init_time:10.2e} {self.emt.init_time / elapsed:8.2%}
        Comp:        {numba_comp:10.2e} {numba_comp / elapsed:8.2%}
        {variable_timing_string}
        Total:       {elapsed:10.2e}
        """
        print(timing_string)
        # ==========================================================================
        self.destroy()

    def destroy(self):
        "Clears memory and frees memory from HELICS>"
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    """Creates and runs HELICS simulation."""

    # Static inputs are always defined in a static_inputs.json   
    with open("static_config.json") as f:  # Original code: with open("static_inputs.json") as f:
        config = ParaemtConfig(**json.load(f))

    # TODO,  do not have required file
    # Any HELICS subscriptions should use input_mapping.json
    # with open("input_mapping.json") as f:
    with open("component_definition.json") as f: # Revised by Min
        input_mapping = json.load(f)

    sfed = ParaemtFederate(config, input_mapping, broker_config)  # include emt_initialization here
    sfed.run(config)


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
