# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from . import trainer
from .utils import typemap

__doc__= '''\
Distributed trainers manage trainers in distributed environment.
'''

class WorkerDescriptor(cntk_py.DistributedWorkerDescriptor):
    '''
    Distributed worker descriptor, returned by :class:`Communicator` instance.
    '''

    @property
    def global_rank(self):
        '''
        The global rank of the worker.
        '''
        return super().m_global_rank

    @property
    def host_id(self):
        '''
        The host id of the worker.
        '''
        return super().m_host_id

class Parallelization:
    '''
    Creates a parallization object that encapsulates distributed communicator and distributed trainer
    '''
    def __init__(self, communicator, distributed_trainer):
        self.comm = communicator
        self.dist_trainer = distributed_trainer

    # TODO: change workers/current_worker/barrier to be staticmethod (C++ side as well)
    @typemap
    def workers(self):
        '''
        (`list`) of :class:`WorkerDescriptor`: workers in this communicator.
        '''
        return self.comm.workers()

    @typemap
    def current_worker(self):
        '''
        :class:`WorkerDescriptor`: descriptor of current process.
        '''
        return self.comm.current_worker()

    def barrier(self):
        '''
        sync point to make sure all workers reach the same state
        '''
        self.comm.barrier()
        
    @staticmethod
    def finalize():
        '''
        calls MPI_Finalize. can't call any MPI functions afterwards
        '''
        cntk_py.DistributedCommunicator.finalize();
        
def data_parallel(bits=32):
    '''
    Creates a parallization object for data parallel SGD with optional quantization `bits`
    
    Args:
        bits (`int`): quantization bits, default is 32 for no quantization
        
    Returns:
        (:class:`Parallelization`): a parallization instance to pass to trainer/reader
    '''
    if bits == 32:
        comm = cntk_py.mpicommunicator()
        dist_trainer = cntk_py.create_data_parallel_distributed_trainer(comm, False)
    else:
        comm = cntk_py.quantized_mpicommunicator(True, True, bits)
        dist_trainer = cntk_py.create_quantized_data_parallel_distributed_trainer(comm, False)
    return Parallelization(comm, dist_trainer)