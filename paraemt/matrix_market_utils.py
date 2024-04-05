
import scipy.io as sio

from bbd_matrix import *

def write_bbd_to_matrix_market(Abbd,
                               target,
                               systemN,
                               N_row,
                               N_col,
                               nparts,
                               mode='full'
                               ):
    blk_dim = Abbd.block_dim
    for idx in range(blk_dim-1):
        k = idx + 1
        akk = Abbd.get_diag_block(idx)
        sio.mmwrite(target + '_blk_{}_{}'.format(k,k),
                    akk,
                    comment='BBD admittance matrix block.\nCoordinate=({},{})\nSystem={}\nNRows={}\nNCols={}\nNPartitions={}'.format(k, k, systemN, N_row, N_col, nparts)
                    )

        if mode != 'L':
            akn = Abbd.get_right_block(idx)
            sio.mmwrite(target + '_blk_{}_{}'.format(k,blk_dim),
                        akn,
                        comment='BBD admittance matrix block.\nCoordinate=({},{})\nSystem={}\nNRows={}\nNCols={}\nNPartitions={}'.format(k, blk_dim, systemN, N_row, N_col, nparts)
                        )

        if mode != 'U':
            ank = Abbd.get_lower_block(idx)
            sio.mmwrite(target + '_blk_{}_{}'.format(blk_dim,k),
                        ank,
                        comment='BBD admittance matrix block.\nCoordinate=({},{})\nSystem={}\nNRows={}\nNCols={}\nNPartitions={}'.format(blk_dim, k, systemN, N_row, N_col, nparts)
                        )
    # End for loop

    acorn = Abbd.corner
    sio.mmwrite(target + '_blk_{}_{}'.format(blk_dim, blk_dim),
                acorn,
                comment='BBD admittance matrix corner block.\nCoordinate=({},{})\nSystem={}\nNRows={}\nNCols={}\nNPartitions={}'.format(blk_dim, blk_dim, systemN, N_row, N_col, nparts)
                )

    return

def write_to_matrix_market(A, target, comment):
    
    sio.mmwrite(target,
                A,
                comment=comment,
                )

    return
