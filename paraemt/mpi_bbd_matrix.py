#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.sparse.linalg as la
import scipy.linalg as dla
import scipy.sparse as sp
import time

from paraemt.bbd_matrix import *

from mpi4py import MPI


def distribute_bbd_matrix(comm, Abbd):

    rank = comm.Get_rank()
    nrank = comm.Get_size()

    if rank == 0:
        block_dim = Abbd.block_dim
    else:
        block_dim = None

    block_dim = comm.bcast(block_dim, root=0)
    npart = block_dim - 1

    (n,r) = divmod(npart, nrank)
    rank_cnt = np.repeat([n], nrank)
    rank_cnt[:r] += 1
    rank_idx_range = np.zeros(rank_cnt.size + 1, dtype=int)
    rank_idx_range[1:] = rank_cnt.cumsum()

    if rank == 0:
        # size = comm.Get_size()
        blocks = [[] for k in range(nrank)]
        for i in range(Abbd.block_dim - 1):
            block_rank = np.argmax(rank_idx_range > i) - 1
            blocks[block_rank].append([i,
                                       Abbd.get_diag_block(i), # Aii
                                       Abbd.get_right_block(i), # Ain
                                       Abbd.get_lower_block(i) # Ani
                                       ])
        for rank in range(nrank):
            blocks[rank].append([Abbd.block_dim - 1,
                                 Abbd.get_diag_block(Abbd.block_dim - 1)
                                 ])
        block_dim = Abbd.block_dim
        block_sizes = Abbd.block_sizes
    else:
        blocks = None
        block_sizes = None

    blocks = comm.scatter(blocks, root=0)

    Ampi = bbd_matrix(block_dim, blocks)
    Ampi.block_sizes = comm.bcast(block_sizes, root=0)

    return (Ampi, rank_idx_range)


def schur_lu(comm, A_bbd, dense_corner=False):

    Lmpi = bbd_matrix(A_bbd.block_dim)
    Umpi = bbd_matrix(A_bbd.block_dim)

    for idx in A_bbd.diag_blocks.keys():
            
        Aii = A_bbd.diag_blocks[idx]
        Ain = A_bbd.right_blocks[idx]
        Ani = A_bbd.lower_blocks[idx]

        N = Aii.shape[0]

        lu = la.splu(Aii.tocsc(), permc_spec="NATURAL")
        Lmpi.diag_blocks[idx] = lu.L[lu.perm_r,:]
        Umpi.diag_blocks[idx] = lu.U

        Lmpi.lower_blocks[idx] = la.spsolve(lu.U.transpose().tocsc(),
                                            Ani.transpose().tocsc()
                                            ).transpose()
        Umpi.right_blocks[idx] = la.spsolve(Lmpi.diag_blocks[idx], Ain.tocsc())

    N = A_bbd.corner.shape[0]
    B = sp.csc_matrix((N,N))
    for idx in A_bbd.diag_blocks.keys():
        B += Lmpi.lower_blocks[idx] @ Umpi.right_blocks[idx]
    B = comm.allreduce(B, op=MPI.SUM)

    lu = la.splu(A_bbd.corner.tocsc() - B)
    Lmpi.corner = lu.L[lu.perm_r,:]
    Umpi.corner = lu.U[:,lu.perm_c]

    if dense_corner:
        Lmpi.corner = Lmpi.corner.todense()
        Umpi.corner = Umpi.corner.todense()

    Lmpi.complete = True
    Umpi.complete = True

    return (Lmpi, Umpi)



def drop_small(Acsc, tol):

    idx = np.abs(Acsc.data) < tol
    Acsc.data[idx] = 0.0
    Acsc.eliminate_zeros()
    return sum(idx)



class schur_bbd_lu:
    def __init__(self, comm, A_bbd, drop_tol=-1.0, dense_corner=False):

        self.comm = comm.Dup()

        (A_mpi, rank_idx_range) = distribute_bbd_matrix(self.comm, A_bbd)
        (self.L, self.U) = schur_lu(self.comm, A_mpi, dense_corner=dense_corner)

        if drop_tol > 0.0:
            ndrop = 0
            for k in self.L.diag_blocks.keys():
                ndrop += drop_small(self.L.diag_blocks[k], drop_tol)
                ndrop += drop_small(self.L.lower_blocks[k], drop_tol)
                ndrop += drop_small(self.U.diag_blocks[k], drop_tol)
                ndrop += drop_small(self.U.right_blocks[k], drop_tol)

            print("Dropped {} values in non-corner blocks".format(ndrop))

            ndrop = drop_small(self.L.corner, drop_tol)
            ndrop += drop_small(self.U.corner, drop_tol)

            print("Dropped {} values in corner blocks".format(ndrop))


        self.block_sizes = A_mpi.block_sizes

        rank = self.comm.Get_rank()
        self.rank_size = np.zeros(self.comm.Get_size(), dtype=int)
        for r in range(self.rank_size.size):
            for idx in range(rank_idx_range[r], rank_idx_range[r+1]):
                self.rank_size[r] += self.block_sizes[idx]

        self.rank_start_idx_dense = np.cumsum(self.rank_size) - self.rank_size
        self.my_dimension = self.rank_size[rank]
        self.my_idx_range = rank_idx_range[rank:rank+2].copy()

        self.drop_tol = drop_tol
        self.dense_corner = dense_corner

        self.t_formbv = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0

        self.tf_yloop = 0.0
        self.tf_allreduce = 0.0
        self.tf_csolve = 0.0

        self.tb_csolve = 0.0
        self.tb_loop = 0.0
        self.tb_allgather = 0.0

        self.solves = 0

        return

    def _schur_forward(self, comm, b_bv):

        t0 = time.time()

        L = self.L

        y = {}
        c = np.zeros(L.corner.shape[0])
        for idx in L.diag_blocks.keys():
            y[idx] = la.spsolve(L.diag_blocks[idx], b_bv[idx])
            c += L.lower_blocks[idx] @ y[idx]

        t1 = time.time()

        comm.Allreduce(MPI.IN_PLACE, c, op=MPI.SUM)

        t2 = time.time()

        if self.dense_corner:
            yn = dla.solve(L.corner, b_bv[L.block_dim - 1] - c)
        else:
            yn = la.spsolve(L.corner, b_bv[L.block_dim - 1] - c)

        t3 = time.time()

        self.tf_yloop += t1 - t0
        self.tf_allreduce += t2 - t1
        self.tf_csolve += t3 - t2

        return (y, yn)

    def _schur_backward(self, comm, y, yn, b_bv):

        t0 = time.time()

        U = self.U

        if self.dense_corner:
            xn = dla.solve(U.corner, yn)
        else:
            xn = la.spsolve(U.corner, yn)

        t1 = time.time()

        x_rank = np.empty(self.my_dimension, dtype=float)

        start_idx = 0
        for idx in range(self.my_idx_range[0], self.my_idx_range[1]):
            rhs_i = y[idx] - U.right_blocks[idx] @ xn
            xi = la.spsolve(U.diag_blocks[idx], rhs_i, permc_spec="NATURAL")
            x_rank[start_idx:start_idx+xi.size] = xi
            start_idx += xi.size

        t2 = time.time()

        x_dense = np.empty(b_bv.size, dtype=float)
        recv = [x_dense, self.rank_size, self.rank_start_idx_dense, MPI.DOUBLE]
        self.comm.Allgatherv(x_rank, recv)
        x_dense[-xn.size:] = xn

        t3 = time.time()

        self.tb_csolve += t1 - t0
        self.tb_loop += t2 - t1
        self.tb_allgather += t3 - t2

        return x_dense

    def schur_solve(self, b_dense):

        t0 = time.time()

        # b_bv = self.comm.bcast(b_bv, root=0)
        b_bv = block_vector(self.block_sizes, x_dense=b_dense)

        t1 = time.time()

        (y, yn) = self._schur_forward(self.comm, b_bv)

        t2 = time.time()

        x_dense = self._schur_backward(self.comm, y, yn, b_bv)

        t3 = time.time()

        self.t_formbv += t1 - t0
        self.t_forward += t2 - t1
        self.t_backward += t3 - t2
        self.solves += 1

        return x_dense

    def print_timing(self):

        total = self.t_forward + self.t_backward
        time_str = """
        Rank:       {:d}
        Calls:      {:d}
        BlkVect:    {:10.2e} {:8.2%} {:8.2e}
        Forward:    {:10.2e} {:8.2%} {:8.2e}
          FLoop:    {:10.2e} {:8.2%} {:8.2e}
          FReduce:  {:10.2e} {:8.2%} {:8.2e}
          FCSolve:  {:10.2e} {:8.2%} {:8.2e}
        Backward:   {:10.2e} {:8.2%} {:8.2e}
          BCSolve:  {:10.2e} {:8.2%} {:8.2e}
          BLoop:    {:10.2e} {:8.2%} {:8.2e}
          BGather:  {:10.2e} {:8.2%} {:8.2e}
        SchurSolve: {:10.2e}
        """.format(
            self.comm.Get_rank(),
            self.solves,
            self.t_formbv, self.t_formbv/total, self.t_formbv/self.solves,
            self.t_forward, self.t_forward/total, self.t_forward/self.solves,
            self.tf_yloop, self.tf_yloop/total, self.tf_yloop/self.solves,
            self.tf_allreduce, self.tf_allreduce/total, self.tf_allreduce/self.solves,
            self.tf_csolve, self.tf_csolve/total, self.tf_csolve/self.solves,
            self.t_backward, self.t_backward/total, self.t_backward/self.solves,
            self.tb_csolve, self.tb_csolve/total, self.tb_csolve/self.solves,
            self.tb_loop, self.tb_loop/total, self.tb_loop/self.solves,
            self.tb_allgather, self.tb_allgather/total, self.tb_allgather/self.solves,
            total,
        )

        print(time_str)

        return

    def print_summary(self):

        # sum_str = "        Rank:  {:d}\n".format(self.comm.Get_rank())
        # sum_str += self.L.summarize()
        # sum_str += self.U.summarize()

        # sum_strings = self.comm.gather(sum_str, 0)

        Lsum = self.L.summarize()
        Usum = self.U.summarize()

        Lsums = self.comm.gather(Lsum, 0)
        Usums = self.comm.gather(Usum, 0)

        if self.comm.Get_rank() == 0:
            lsum = Lsums[0]
            for ls in Lsums[1:]:
                lsum[0] += ls[0]
                for k in range(1,5):
                    if k < 3:
                        if lsum[k] > ls[k]:
                            lsum[k] = ls[k]
                    else:
                        if lsum[k] < ls[k]:
                            lsum[k] = ls[k]

            usum = Usums[0]
            for us in Usums[1:]:
                usum[0] += us[0]
                for k in range(1,5):
                    if k < 3:
                        if usum[k] > us[k]:
                            usum[k] = us[k]
                    else:
                        if usum[k] < us[k]:
                            usum[k] = us[k]


            bbd_str = """
            **** {0} Summary ****
            {0} Total NNZ:      {1:7d}
            {0} Min Block Size: {2:7d}
            {0} Min Block NNZ:  {3:7d}
            {0} Max Block Size: {4:7d}
            {0} Max Block NNZ:  {5:7d}
            {0} Corner Size:    {6:7d}
            {0} Corner NNZ:     {7:7d}
            """

            print(bbd_str.format(
                'L',
                lsum[0],
                lsum[1],
                lsum[2],
                lsum[3],
                lsum[4],
                lsum[5],
                lsum[6],
            ))

            print(bbd_str.format(
                'U',
                usum[0],
                usum[1],
                usum[2],
                usum[3],
                usum[4],
                usum[5],
                usum[6],
            ))

        return

