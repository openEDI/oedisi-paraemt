import sys
import numpy as np
import pandas as pd
import ast
import glob
import json
import random

from pkg_resources import yield_lines
from federateaccesspoint import federateagent
# from main_step1_sim import paraemt
# from main_step2_save import saveoutputsparaemt

def helics_get_json(fname):
    f = open(fname, "r")
    js = json.load(f)
    f.close()
    return js  

# class paraemtagent(federateagent):
#     """
#         Agent to handle poewrflow for the electrical network
#     """

#     def __init__(self, config_dict):
#         super(paraemtagent, self).__init__(
#             name=config_dict['name'], feeder_num=0, starttime=config_dict['starttime'], endtime=config_dict['stoptime'], agent_num=0, config_dict=config_dict)
#         self.config_dict = config_dict
#         self.idx = 0
#         self.recv = {}

#     def initialize_paraemt_simulation(self): 
#         self.peobj = paraemt(config_dict=self.config_dict['paraemt'])
#         self.deltat = self.peobj.ts
#         self.tn = self.peobj.tn

#     def run(self):
#         """ This must be implemented """
#         flag_reini = 0
#         tsave = 0
#         self.sync_time_helics(self.deltat)
#         self.initialize_paraemt_simulation()
#         self.Tlen = self.config_dict['paraemt']['Tlen']
#         self.peobj.tn = -1
#         while self.currenttime < (self.endtime - self.starttime ):
#             self.peobj.tn += 1
#             self.peobj.run1()
#             tmp = self.helics_get_all()
#             if tmp != []:
#                 self.process_subscription_event(tmp)
#             self.peobj.recv = self.recv
#             self.peobj.run2()
#             self.process_periodic_publication()
#             self.sync_time_helics(self.deltat)

#             self.idx += 1
#         self.peobj.emtfinalize()
#         saveoutputsparaemt()


#     def process_subscription_event(self, msg):
#         # process data from Gridpack. 
#         # Right now this data is coming from a surrogate Gridpack EMT, 
#         # which reads a csv and publishes periodically. 
        
#         keys_to_check = ["gridpack_i_9_10_real",
#             "gridpack_i_9_10_imag",
#             "gridpack_v_9_10_real",
#             "gridpack_v_9_10_imag"]
#         if self.get_currenttime() > 0.001 :
#             print(f"{self.name}, {self.get_currenttime()} subscribed to message {msg.values()}")
#             if(set(list(msg.keys())) .intersection(set(keys_to_check))):
#                 self.recv["i_9_10_phasor"] = complex(msg['gridpack_i_9_10_real'],msg['gridpack_i_9_10_imag'])
#                 self.recv["v_10_phasor"] = complex(msg["gridpack_v_9_10_real"], msg["gridpack_v_9_10_imag"])
#                 self.peobj.recv = self.recv
#         else:
#             print("Using default values from gridpack ", self.get_currenttime() )
#             self.recv["i_9_10_phasor"] = complex(- 9.713144, 10.52305)
#             self.recv["v_10_phasor"] = complex(0.70720997, - 0.683408834)
#             self.peobj.recv = self.recv

#     def process_periodic_publication(self):
#         # Periodically publish data to the surrpogate 
#         # Gridpack agent, that sucscribes and does nothing as of now. 
#         # "emt_Pinj_9_10",
#         # "emt_Qinj_9_10"
#         print("broadcast values : ",self.peobj.emt.Pinj, self.peobj.emt.Qinj)
#         for y in self.pub.keys(): 
#             if y == "emt_Pinj_9_10":
#                 # self.broadcast(self.pub[y], self.peobj.emt.Pinj)
#                 self.broadcast(y, self.peobj.emt.Pinj*-1.0)
#                 # print(f"{self.name}, {self.get_currenttime()} published:  [{y}:{self.peobj.emt.Pinj}]")
#             if y == "emt_Qinj_9_10":
#                 # self.broadcast(self.pub[y], self.peobj.emt.Qinj)
#                 self.broadcast(y, self.peobj.emt.Qinj*-1.0)
#                 # print(f"{self.name}, {self.get_currenttime()} published:  [{y}:{self.peobj.emt.Qinj}]")            

#     def process_endpoint_event(self, msg):
#         pass
#     def process_periodic_endpoint(self):
#         pass

class gridpackstandin(federateagent):
    """
        Agent for gridpack standin
    """

    def __init__(self, config_dict):
        super(gridpackstandin, self).__init__(
            name=config_dict['name'], feeder_num=0, starttime=config_dict['starttime'], endtime=config_dict['stoptime'], agent_num=0, config_dict=config_dict)
        self.config_dict = config_dict
        self.idx = 0
        self.logvolt = []

    def run(self):
        while self.currenttime < (self.endtime - self.starttime ):
            self.process_periodic_publication()
            tmp = self.helics_get_all()
            if tmp != []:
                self.process_subscription_event(tmp)
            self.sync_time_helics(self.deltat)

    def process_subscription_event(self, msg):
        # process data from Gridpack. 
        # Right now this data is coming from a surrogate Gridpack EMT, 
        # which reads a csv and publishes periodically. 
        print(f"{self.name}, {self.get_currenttime()} subscribed to message {msg}")
        ## Inject the values subscribed from EMT into GridPACK. 


    def process_periodic_publication(self):
        # Periodically publish data to the surrpogate 
        # Gridpack agent, that sucscribes and does nothing as of now.
        self.idx +=1  
        i_9_10_phasor = complex(-9.713144, 10.52305) # this default value here is expected at t=0.0
        v_10_phasor = complex(0.70720997, -0.683408834) # this default value here is expected at t=0.0

        ## Gridpack needs to create these values and publish to HELICS message bus. 
        ir = i_9_10_phasor.real
        ii = i_9_10_phasor.imag
        vr = v_10_phasor.real
        vi = v_10_phasor.imag
            # "gridpack_i_9_10_real",
            # "gridpack_i_9_10_imag",
            # "gridpack_v_9_10_real",
            # "gridpack_v_9_10_imag"
        for y in self.pub.keys(): 
            if y == "gridpack_i_9_10_real":
                pubval = ir
            if y == "gridpack_i_9_10_imag":
                pubval = ii
            if y == "gridpack_v_9_10_real":
                pubval = vr
            if y == "gridpack_v_9_10_imag":
                pubval = vi

            print("broadcast values : ",ir, ii, vr, vi)
            # self.broadcast(self.pub[y],pubval)
            self.broadcast(y,pubval)
            # print(f"{self.name}, {self.get_currenttime()}  published:  [{y}:{pubval}]")

    def process_endpoint_event(self, msg):
        pass
    def process_periodic_endpoint(self):
        pass    


def OrchestrateAgents(fname):
    js = config_dict = helics_get_json(fname)
    obj = globals()[js['Agent']](js)
    broker_port = js["helics"].get('broker_port', 32000)
    broker_ip = js["helics"].get('broker_ip', "127.0.0.1")
    obj.run_helics_setup(broker_ip=broker_ip, broker_port=broker_port)
    obj.enter_execution(function_targets=[],
                        function_arguments=[[]])
