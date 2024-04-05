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
from oedisi.types.data_types import VoltageArray
from pydantic import BaseModel

from ctypes import *  # For DLL
from paraemt.Python_C_DLL_wrapper import wrap_function, MODELINSTANCE, MODELINFO
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

EMT_config = """
{
    # Important
    "systemN": 6,
    "DSrate": 40,                     # down sampling rate, i.e. results saved every DSrate sim steps.
    "ts": 50e-6,                      # time step, second
    "Tlen": 2,                        # total simulation time length, second
    "use_helics": False

    # Less important
    "SimMod": 0,                      # 0 - Standard Initialization, 1 - Initialize from Snapshot
    "save_snapshot": False,           # Save snapshot at end of run or not
    "save_snapshot_mode": 0,          # 0 - 1 point snapshot, 1 - whole simulation snapshot, 2 - both
    "compute_phasor": 1,
    "dump_all": False,                # Save to CSVs intermediate calculation results--useful for debugging
    "save_results_csv": True,         # Save results to csv files
    "show_progress": True,            # Display simulation progress
    "record4cosim": True,             # recording for pseudo co-sim
    "playback_enable": False,         # playback or helics for co-sim: True: play recorded signals, False: read signals thru helics
    "playback_voltphasor": False,     # replay voltage phasors at all buses
    "EMT_N": 1,
    "N_row": 1,                       # haven't tested the mxn layout, so plz don't set N_row/N_col to other nums.
    "N_col": 1,
    "kts": 333,                       # phasor step = kts * EMT steps
    "stepk": 332,                     # k-th EMT step within one phasor step, initialized at kts-1
    "t_release_f": 0,
    "loadmodel_option": 1,            # 1-const rlc, 2-const z
    "netMod": lu,
    "Gd": 100,                        # interface R for pseudo co-sim 
    "Go": 0.001,
    "t_sc": 2000,                     # ctrl step change time
    "i_gen_sc": 2,
    "flag_exc_gov": 0,                # 0 - exc, 1 - gov
    "dsp": -0.02,
    "flag_sc": 1,                     # defulat 1, do not change
    "t_gentrip": 1000,                # gen trip time
    "i_gentrip": 1,                   # gen idx
    "flag_gentrip": 1,
    "flag_reinit": 1
    }
}
"""
class TestFederateConfig(BaseModel):
    """Pydantic model for the static configuration of the federate."""
    #Configuration in JSON format
    SimMod: int                 # 0 - Standard Initialization, 1 - Initialize from Snapshot
    save_snapshot: bool         # Save snapshot at end of run or not
    save_snapshot_mode: int     # 0 - 1 point snapshot, 1 - whole simulation snapshot, 2 - both
    DSrate: int                 # down sampling rate, i.e. results saved every DSrate sim steps.
    compute_phasor: int         # Save to CSVs intermediate calculation results--useful for debugging
    dump_all: bool              # Save results to csv files
    save_results_csv: bool      # Display simulation progress
    show_progress: bool         # recording for pseudo co-sim
    record4cosim: bool          # playback or helics for co-sim: True: play recorded signals, False: read signals thru helics
    playback_enable: bool       # replay voltage phasors at all buses
    playback_voltphasor: bool
    systemN: int
    EMT_N: int
    N_row: int                  # haven't tested the mxn layout, so plz don't set N_row/N_col to other nums.
    N_col: int
    use_helics: bool
    ts: float                   # time step, second, 50.05005005
    kts: int                    # phasor step = kts * EMT steps
    stepk: int                  # k-th EMT step within one phasor step, initialized at kts-1
    t_release_f: float
    loadmodel_option: int       # 1-const rlc, 2-const z
    netMod: str                 # interface R for pseudo co-sim 
    Gd: float
    Go: float
    Tlen: float                 # total simulation time length, second
    # ctrl step change
    t_sc: float
    i_gen_sc: int
    flag_exc_gov: int           # 0 - exc, 1 - gov
    dsp: float
    flag_sc: int
    # gen trip
    t_gentrip: float
    i_gentrip: int              # 13: 1431-N, 15: 2032 G
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

    # systemN
    # 1: 2-gen
    # 2: 9-bus
    # 3: 39-bus
    # 4: 179-bus
    # 5: 240-bus
    # 6: 2-area
    # 7: 1-EMT zone in 2-area case
    # 8: 2-area case for debugging EPRI IBR model
    # 9: 2-area case for GridPack-ParaEMT interface demo
    # 10: IBR benchmark case (2-bus)
    # 11: IBR benchmark case (3-bus)
    # 12: IBR benchmark case (3-bus) with X_line = 1
    # 13: IBR benchmark case (3-bus, 2 IBRs)
    # 14: 240-bus with EPRI's IBR model
    # 15: 2-area (recording signals at 2 interfaces)
    # 16: 2-EMT zones in 2-area case
    # 17: IBR benchmark case (3-bus, low SCR by GenTrip)
    # 18: IBR benchmark new case 1
    # 19: IBR benchmark new case 2
    # 20: IBR benchmark new case 3
    # 21: 2gen case for correcting ParaEMT by Matlab
    # 22： modified 240-bus
    # 23: old 240-bus (37IBRs)

class TestFederate:
    def __init__(
        self,
        config: TestFederateConfig,
        input_mapping: dict[str, str],
        broker_config: BrokerConfig,
    ):
        """Template federate to get you started.

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
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, 0.000005)

        self.vfed = h.helicsCreateValueFederate(config.name, fedinfo)
        logger.info("Value federate created")

        # Register any subscriptions you may have
        # self.sub_example = self.vfed.register_subscription(
        #    input_mapping["subscription_name"], ""
        # )
        
        #==========================================================================
        # Read the JOSON configuuration of EMT simulation
        config = TestFederateConfig.parse_raw(EMT_config)  #EMT_settings 
        # ParaEMT initialization
        shared_lib_path = os.getcwd() + "\\models\\ibrepri.dll"  # EPRI's IBR model
        add_lib = CDLL(shared_lib_path)  
        Model_GetInfo = wrap_function(add_lib,'Model_GetInfo',POINTER(MODELINFO),None)
        Model_Outputs = wrap_function(add_lib,'Model_Outputs',c_int,[POINTER(MODELINSTANCE)])
        info = Model_GetInfo()
        num_in_ports,num_out_ports,num_param = info.contents.cNumInputPorts, info.contents.cNumOutputPorts,info.contents.cNumParameters
        num_int_states,num_float_states,num_double_states = info.contents.cNumIntStates,info.contents.cNumFloatStates,info.contents.cNumDoubleStates

        workingfolder = os.getcwd()  #TODO Joseph, deal with it
        os.chdir(workingfolder)

        SimMod= config.SimMod        # Import EMT simulation configuration              
        # save_snapshot= config.save_snapshot       
        # save_snapshot_mode= config.save_snapshot_mode     
        DSrate= config.DSrate                
        compute_phasor= config.compute_phasor       
        # dump_all= config.dump_all         
        # save_results_csv= config.save_results_csv    
        # show_progress= config.show_progress        
        record4cosim= config.record4cosim         
        playback_enable= config.playback_enable      
        # playback_voltphasor= config.playback_voltphasor
        systemN= config.systemN
        EMT_N= config.EMT_N
        N_row= config.N_row              
        N_col= config.N_col
        use_helics= config.use_helics
        ts= config.ts                 
        kts= config.kts                
        stepk= config.stepk                 
        t_release_f= config.t_release_f
        loadmodel_option= config.loadmodel_option      
        netMod= config.netMod                
        Gd= config.Gd
        Go= config.Go
        Tlen= config.Tlen  

        sim_info = """
        ---- Sim Info ----
        System: {:d}
        N Rows: {:d}
        N Cols: {:d}
        Time Step: {:.8e}
        Time End: {:e}
        Network Solve Mode: {:s}
        """.format(
            systemN,
            N_row,
            N_col,
            ts,
            Tlen,
            netMod,
        )
        print(sim_info)
        # TODO: Test and debug snapshot stuff
        input_snp = 'sim_snp_S' + str(systemN) + '_' + str(int(ts * 1e6)) + 'u_1pt.pkl'
        t0 = time.time()
        if SimMod == 0:
            emt = SerialEmtSimu(workingfolder=workingfolder,
                                systemN=systemN,
                                EMT_N = EMT_N,
                                N_row=N_row,
                                N_col=N_col,
                                ts=ts,
                                Tlen=Tlen,
                                kts=kts,
                                stepk=stepk,
                                save_rate=DSrate,
                                netMod=netMod,
                                loadmodel_option=loadmodel_option,
                                record4cosim = record4cosim,
                                playback_enable=playback_enable,
                                Gd = Gd,
                                Go = Go,
                                )
        else:
            print("Loading snapshot file: ", input_snp)
            emt = SerialEmtSimu.initialize_from_snp(input_snp, netMod)
        emt.compute_phasor = compute_phasor
        emt.tsat_gen_omg = [0]
        emt.tsat_gen_maci = [0]
        emt.tsat_t = [0]
        emt.Tlen = config.Tlen
        # ctrl step change
        emt.t_sc = config.t_sc
        emt.i_gen_sc = config.i_gen_sc
        emt.flag_exc_gov = config.flag_exc_gov  
        emt.dsp = config.dsp
        emt.flag_sc = config.flag_sc 
        # gen trip
        emt.t_gentrip = config.t_gentrip
        emt.i_gentrip = config.i_gentrip
        emt.flag_gentrip = config.flag_gentrip
        emt.flag_reinit = config.flag_reinit
        emt.t_release_f = t_release_f
        emt.loadmodel_option = loadmodel_option 
        if use_helics:
            emt.helics_setup()
        #==========================================================================

        # This should match the dynamic output in component_definition.json
        self.pub_example = self.vfed.register_publication(
            "appropriate_helics_pub", h.HELICS_DATA_TYPE_DOUBLE, ""
        )

    def run(self, emt, config):
        """Run HELICS simulation until completion."""
        # Starts HELICS simulation. Essentially says to broker "This simulation is ready"
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")
        # We request the time we want
        granted_time = h.helicsFederateRequestTime(self.vfed, 0.0)

        tn = 0 # number of time time steps    Min added
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
        while granted_time < config.Tlen:
        # while tn*ts< config.Tlen:  #TODO, Joseph
            logger.info("start time: " + str(datetime.now()))
            #==========================================================================
            # Run ParaEMT, Only the time loop function
            tn += 1
            if config.show_progress:    # Print the simulation progress
                if tn>1:
                    if np.mod(tn,500)==0:
                        print("%.3f" % emt.t[-1])
            tl_0 = time.time()
            flag_reini = 0
            emt.StepChange(emt.dyd, emt.ini, tn)  # configure step change in exc or gov references
            if ((config.flag_gentrip == 0 and config.flag_reinit == 1)):  # If the generator is tripped
                flag_reini = 1
            tl_1 = time.time()
            emt.predictX(emt.pfd, emt.dyd, emt.ts)
            tl_2 = time.time()
            emt.updateIg(emt.pfd, emt.dyd, emt.ini)
            tl_3 = time.time()
            emt.updateIibr(emt.pfd, emt.dyd, emt.ini)
            tl_4 = time.time()
            emt.updateIibr_epri(emt.pfd, emt.dyd, emt.ini, tn)
            tl_5 = time.time()
            emt.updateIl(emt.pfd, emt.dyd, tn) # update current injection from load
            tl_6 = time.time()
            emt.helics_receive(tn, config.record4cosim)
            tl_7 = time.time()
            emt.GenTrip(emt.pfd, emt.dyd, emt.ini, tn)  # configure generation trip
            tl_8 = time.time()
            # re-init
            if flag_reini==1:
                emt.Re_Init(emt.pfd, emt.dyd, emt.ini, tn)
            tl_9 = time.time()
            emt.presolveV()
            emt.Vsol = emt.solveV(emt.ini.admittance_mode,
                                    emt.Igs,
                                    emt.Igi + emt.Igi_epri,
                                    emt.node_Ihis,
                                    emt.Il)
            tl_10 = time.time()
            emt.BusMea(emt.pfd, emt.dyd, tn)  # bus measurement
            tl_11 = time.time()
            emt.updateX(emt.pfd, emt.dyd, emt.ini, tn, config.playback_voltphasor)
            tl_12 = time.time()
            emt.updateXibr(emt.pfd, emt.dyd, emt.ini)
            tl_13 = time.time()
            emt.updateXl(emt.pfd, emt.dyd, tn)
            tl_14 = time.time()
            if (len(emt.emt_zones)>0) and config.record4cosim==False: 
                emt.update_phasor()
            tl_15 = time.time()       
            emt.helics_publish()
            tl_16 = time.time()
            emt.updateIhis(emt.ini)
            tl_17 = time.time()
            emt.save(tn, config.record4cosim)  # save has to be placed after updateIhis to ensure time alignment for recorded signals for pseudo co-sim
            tl_18 = time.time()
            emt.helics_update()
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
            # end of run ParaEMT
            #==========================================================================
            self.pub_example.publish(
                # If possible, use either basic types available like floats, ints, etc, or types provided
                # by the oedisi.types.data_types module.
                # Any indexing information should have appropriate labels if necessary.
                # VoltageArray(values=[0.0, 1.0, 2.0], ids=["node1", "node2", "node3"])
                # TODO Joseph, confirm
                emt.Vsol,       # Three phase voltage waveform
                emt.brch_Ipre   # Three phase current waveform
            )

            granted_time = h.helicsFederateRequestTime(self.vfed, int(granted_time) + 1)
            logger.info("end time: " + str(datetime.now()))

        #==========================================================================
        # Save simulation results locally
        save_results_csv=config.save_results_csv
        if save_results_csv:
            print("Saving results...")
            df_v = pd.DataFrame(emt.v).T
            df_v.to_csv("paraemt_v.csv")
            df_x = pd.DataFrame(emt.x).T
            df_x.to_csv("paraemt_x.csv")
            df_ibr = pd.DataFrame(emt.x_ibr).T
            df_ibr.to_csv("paraemt_ibr.csv")
            df_ebr = pd.DataFrame(emt.x_ibr_epri).T
            df_ebr.to_csv("paraemt_ebr.csv")
            df_bus = pd.DataFrame(emt.x_bus).T
            df_bus.to_csv("paraemt_bus.csv")
            df_load = pd.DataFrame(emt.x_load).T
            df_load.to_csv("paraemt_load.csv")
        # Save simulation snapshot locally 
        output_snp_ful = 'sim_snp_S' + str(config.systemN) + '_' + str(int(ts * 1e6)) + 'u.pkl'
        output_snp_1pt = 'sim_snp_S' + str(config.systemN) + '_' + str(int(ts * 1e6)) + 'u_1pt.pkl'
        output_res = 'sim_res_S' + str(config.systemN) + '_' + str(int(ts * 1e6)) + 'u.pkl'  
        if config.save_snapshot:
            print("Saving Snapshot...")
            emt.dump_res(config.SimMod, config.save_snapshot_mode, output_snp_ful, output_snp_1pt, output_res)
        #==========================================================================

        #========================================================================== 
        # Print time cost information of EMT simulation
        t2 = time.time()
        numba_comp = 0
        loop = t2 - t1
        elapsed = numba_comp + loop + emt.init_time
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
        #========================================================================== 
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
    with open("static_inputs.json") as f:
        config = TestFederateConfig(**json.load(f))

    # Any HELICS subscriptions should use input_mapping.json
    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = TestFederate(config, input_mapping, broker_config)
    sfed.run()

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
