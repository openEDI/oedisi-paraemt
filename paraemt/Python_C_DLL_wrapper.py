#Python 37
from ctypes import *
from sys import platform
import os

# Ensure is dll is created as 64 bit dll
shared_lib_path = os.getcwd() + "\\models\\ibrepri.dll"
if platform == "linux" or platform == "linux2":
    ext = ".so"
elif platform == "darwin":
    ext = ".dylib"
else:
    ext = ".dll"

shared_lib_path = os.path.join(os.getcwd(), "models", "ibrepri" + ext)

# Define Python ctype structures to mirror Structures in C
class INTERFACESIGNAL(Structure):
    _fields_ = [("cName", c_char_p),
                ("cDescription", c_char_p),
                ("cUnit", c_char_p),
                ("cDatatype", c_double),
                ("cWidth", c_int)]
class PARAMETERS(Structure):
    _fields_ = [("cName", c_char_p),
                ("cDescription", c_char_p),
                ("cUnit", c_char_p),
                ("cDatatype", c_double),
                ("cFixedValue", c_int),
                ("cDefaultValue", c_double),
                ("cMinValue", c_double),
                ("cMaxValue", c_double)]
class MODELINFO(Structure):
    _fields_ = [("cDLLInterfaceversion", c_char_p),
                ("cModelName", c_char_p),
                ("cModelVersion", c_char_p),
                ("cModelDescription", c_char_p),
                ("cGeneralInformation", c_char_p),
                ("cModelCreated", c_char_p),
                ("cModelCreator", c_char_p),
                ("cModelLastModifiedDate", c_char_p),
                ("cModelLastModifiedBy", c_char_p),
                ("cModelModifiedComment", c_char_p),
                ("cModelModifiedHistory", c_char_p),
                ("cFixedStepBasedSampleTime", c_double),
                ("cEMT_RMS_Mode", c_uint),
                ("cNumInputPorts", c_int),
                ("cInputPortsInfo", POINTER(INTERFACESIGNAL)),
                ("cNumOutputPorts", c_int),
                ("cOutputPortsInfo", POINTER(INTERFACESIGNAL)),
                ("cNumParameters", c_int),
                ("cParametersInfo", POINTER(PARAMETERS)),
                ("cNumIntStates", c_int),
                ("cNumFloatStates", c_int),
                ("cNumDoubleStates", c_int)]
class MODELINSTANCE(Structure):
    _fields_ = [("cExternalInputs", POINTER(c_double)),
                ("cExternalOutputs", POINTER(c_double)),
                ("cParameters", POINTER(c_double)),
                ("cTime", c_double),
                ("cLastError", c_char_p),
                ("cLastGeneral", c_char_p),
                ("cIntStates", POINTER(c_int)),
                ("cFloatStates", POINTER(c_float)),
                ("cDoubleStates", POINTER(c_double))]

# Define Python wrapping function to mirror C functions
def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


# # Load DLL
# add_lib = CDLL(shared_lib_path)   # Min

# # Get information regarding model
# Model_GetInfo = wrap_function(add_lib,'Model_GetInfo',POINTER(MODELINFO),None)
# info = Model_GetInfo()
# num_in_ports,num_out_ports,num_param = info.contents.cNumInputPorts, info.contents.cNumOutputPorts,info.contents.cNumParameters
# num_int_states,num_float_states,num_double_states = info.contents.cNumIntStates,info.contents.cNumFloatStates,info.contents.cNumDoubleStates
# print ("Model Name: ", info.contents.cModelName)
# print ("Model Version: ", info.contents.cModelVersion)
# print ("Model Description: ", info.contents.cModelDescription)
# print ("Model General Information: ", info.contents.cGeneralInformation)
# print ("Model Created: ", info.contents.cModelCreated)
# print ("Model Creator: ", info.contents.cModelCreator)
# print ("Model Last modified date: ", info.contents.cModelLastModifiedDate)
# print ("Model Last modified by: ", info.contents.cModelLastModifiedBy)
# print ("Model Modified comments: ", info.contents.cModelModifiedComment)
# print ("Model Modification history: ", info.contents.cModelModifiedHistory)
# print ("Model Fixed Sample Time: ", info.contents.cFixedStepBasedSampleTime)
# print ("Model EMT_RMS_Mode: ", info.contents.cEMT_RMS_Mode)
# print ("Model Number of Input Ports: ", info.contents.cNumInputPorts)
# print ("Model Number of Output Ports: ", info.contents.cNumOutputPorts)
# print ("Model Number of Parameters: ", info.contents.cNumParameters)
# print ("Model Number of Integer States: ", info.contents.cNumIntStates)
# print ("Model Number of Float States: ", info.contents.cNumFloatStates)
# print ("Model Number of Double States: ", info.contents.cNumDoubleStates)
# print ("End of Model information")

# # Set initial inputs, outputs, parameters, and states
# kVbase, IBR_MVA_base, fbase = 0.65, 100.0, 60.0
# Inputs = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,90.0,5.0,1.0]  #Should be updated at each time step
# Outputs = [0.00,0.0,0.00,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] #Updated by DLL
# Parameters = [kVbase,IBR_MVA_base,(2.5*kVbase),0.5,20.0,20.0,700.0,0.5,20.0,0.5,20.0,1.2,1.0,0.7,1.2,0.0015,0.15,0.0167,1000000.0] #Do not change
# IntSt = []
# FloatSt = []
# DoubleSt = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] #Updated by DLL

# # Define object of model. One object per instance of the model used in simulation
# modelinstance = MODELINSTANCE()
# modelinstance.cExternalInputs = (c_double*num_in_ports)(*Inputs)
# modelinstance.cExternalOutputs = (c_double*num_out_ports)(*Outputs)
# modelinstance.cParameters = (c_double*num_param)(*Parameters)
# modelinstance.cTime = 0.00000   # Should be updated at each time step
# modelinstance.cIntStates = (c_int*num_int_states)(*IntSt)
# modelinstance.cFloatStates = (c_float*num_float_states)(*FloatSt)
# modelinstance.cDoubleStates = (c_double*num_double_states)(*DoubleSt)

# # Call function to check parameter values. Not critical for initial test run of model use
# print ("Check Parameters")
# Model_CheckParameters = wrap_function(add_lib,'Model_CheckParameters',c_int,[POINTER(MODELINSTANCE)])
# return_int = Model_CheckParameters(modelinstance)

# # Call function to initialize states of the model. Needs appropriate code in the DLL
# print ("Model Initialization")
# Model_Initialize = wrap_function(add_lib,'Model_Initialize',c_int,[POINTER(MODELINSTANCE)])
# return_int = Model_Initialize(modelinstance)

# print ("Run rough open loop time domain simulation")
# # Rough test code to run C code DLL in open loop for two cycles of 60 Hz at 5us time step
# time = np.arange(0,(20.0/60.0),0.000005)
# Va, Vb, Vc, f = [],[],[],[]
# for t in time:

#     modelinstance.cTime = t # should be updated by the simulation time step

#     # Should be updated by the network solution based on model output from previous time step
#     Z_netw = (0.15)*kVbase*kVbase/IBR_MVA_base
#     modelinstance.cExternalInputs[0] = kVbase*np.sqrt(2.0/3.0)*math.sin(2*math.pi*fbase*t)
#     modelinstance.cExternalInputs[1] = kVbase*np.sqrt(2.0/3.0)*math.sin((2*math.pi*fbase*t)-(2*math.pi/3.0))
#     modelinstance.cExternalInputs[2] = kVbase*np.sqrt(2.0/3.0)*math.sin((2*math.pi*fbase*t)+(2*math.pi/3.0))

#     modelinstance.cExternalInputs[3] = (modelinstance.cExternalOutputs[0] - modelinstance.cExternalInputs[0])/(Z_netw)
#     modelinstance.cExternalInputs[4] = (modelinstance.cExternalOutputs[1] - modelinstance.cExternalInputs[1])/(Z_netw)
#     modelinstance.cExternalInputs[5] = (modelinstance.cExternalOutputs[2] - modelinstance.cExternalInputs[2])/(Z_netw)

#     modelinstance.cExternalInputs[6] = modelinstance.cExternalInputs[3] + modelinstance.cExternalInputs[0]*0.001
#     modelinstance.cExternalInputs[7] = modelinstance.cExternalInputs[4] + modelinstance.cExternalInputs[1]*0.001
#     modelinstance.cExternalInputs[8] = modelinstance.cExternalInputs[5] + modelinstance.cExternalInputs[2]*0.001

#     # Call main function from the DLL to update outputs based on updated inputs
#     Model_Outputs = wrap_function(add_lib,'Model_Outputs',c_int,[POINTER(MODELINSTANCE)])
#     a = Model_Outputs(modelinstance)

#     # Prepare channels for plotting
#     Va.append(modelinstance.cExternalInputs[0])
#     Vb.append(modelinstance.cExternalInputs[1])
#     Vc.append(modelinstance.cExternalInputs[2])

#     f.append(modelinstance.cExternalOutputs[9])

# fig, (ax1,ax2) = pl.subplots(nrows=2,ncols=1,sharex=True)
# ax1.plot(time,Va)
# ax1.plot(time,Vb)
# ax1.plot(time,Vc)
# ax2.plot(time,f)
# ax1.set_ylabel('Voltage (kV)')
# ax2.set_ylabel('Frequency (Hz)')
# ax2.set_xlabel('Time (s)')

# pl.show()

