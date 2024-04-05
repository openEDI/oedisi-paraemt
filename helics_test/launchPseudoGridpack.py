"""
Script to launch Pseudo-Gridpack agent
"""

from helicsinterface import OrchestrateAgents



# Path to pseudo gridpack configuration file
fname = "./gridpackstandinhelics.json"

# Launch the pseudo Gridpack agent
OrchestrateAgents(fname)




# import glob
# import multiprocessing
# from multiprocessing import Process
# print("CPU available", multiprocessing.cpu_count())

# # Launch the pseudo Gridpack agent
# flist = glob.glob("./gridpackstandinhelics.json")
# flist.sort()
# print(flist)
# allprocess = []

# # create parallel process for the agents
# for fname in flist:
#     p = Process(target=OrchestrateAgents, args=(fname,))
#     allprocess.append(p)

# # Launch agents in parallel
# if __name__ == '__main__':
#     for p in allprocess:
#         p.start()
#     for p in allprocess:
#         p.join()
