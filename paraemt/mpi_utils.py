
import numpy as np

from mpi4py import MPI

def mpi_print(to_print, comm, root=0):
    if comm.Get_rank() == root:
        print(to_print)
    else:
        pass
    return

def mpi_distribute_obj(objects, comm, root=0):
    dist = []
    for obj in objects:
        o = comm.bcast(obj, root=root)
        dist.append(o)
    return dist

def mpi_sync_vect(vector, comm, root=0):
    if comm.Get_rank() == root:
        synced = np.zeros(vector.shape)
    else:
        synced = None
    comm.Reduce(vector, synced, root=root)
    return synced

def save_vect(storage, vector, key):
    storage[key] = vector
    return

def mpi_sync_and_save(storage, vector, key, comm, root=0):
    to_save = mpi_sync_vect(vector, comm, root)
    save_vect(storage, to_save, key)
    return
