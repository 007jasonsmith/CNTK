# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# NOTE:
# This example is meant as an illustration of how to use CNTKs distributed training feature from the python API.
# The training hyper parameters here are not necessarily optimal and for optimal convergence need to be tuned 
# for specific parallelization degrees that you want to run the example with.
# Example for running in parallel:
#     mpiexec -np 2 python CifarResNet_Distributed.py

import numpy as np
import sys
import os
from cntk import distributed, device, persist
from cntk.learner import momentum_sgd, learning_rate_schedule
from cntk.cntk_py import DeviceKind_GPU
from examples.CifarResNet.CifarResNet import CIFAR10Reader, CIFAR10TrainManager

#
# Paths relative to current python file.
#
abs_path   = os.path.dirname(os.path.abspath(__file__))
cntk_path  = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path  = os.path.join(cntk_path, "Examples", "Image", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

def check_gpu():
    # check if we have multiple-GPU, and fallback to 1 GPU if not
    devices = device.all_devices()
    gpu_count = 0
    for dev in devices:
        gpu_count += (1 if dev.type() == DeviceKind_GPU else 0)
    print("Found {} GPUs".format(gpu_count))
    
    if gpu_count == 0:
        print("No GPU found, exiting")
        quit()

    return gpu_count

def check_config(parallelization):
    # list all MPI workers
    workers = parallelization.workers()
    current_worker = parallelization.current_worker()
    print("List all distributed workers")
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            print("* {} {}".format(wk.global_rank, wk.host_id))
        else:
            print("  {} {}".format(wk.global_rank, wk.host_id))

    # check GPU availabilty, as ResNet requires GPU BatchNormalization
    gpu_count = check_gpu()
    if gpu_count == 1 and len(workers) > 1 :
        print("Warning: running distributed training on 1-GPU might be slow")
        device.set_default_device(gpu(0))

def train_and_evaluate(max_epochs, warmup_epochs):
    # Create distributed communicator for 1-bit SGD for better scaling to multiple GPUs
    # If you'd like to avoid quantization loss, use simple one instead
    quantization_bit = 1
    parallelization = distributed.data_parallel(quantization_bit)

    check_config(parallelization)

    # create distributed reader
    reader_train = CIFAR10Reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True, parallelization)
    reader_test  = CIFAR10Reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)

    # 1-bit SGD requires non-distributed training for a few epochs at the beginning
    warmup_train_manager = CIFAR10TrainManager(reader_train)

    # training config
    epoch_size     = 50000
    minibatch_size = 128
    warmup_model   = os.path.join(model_path, 'warmup.model')

    # create learner
    lr_per_sample          = [1/minibatch_size]*80+[0.1/minibatch_size]*40+[0.01/minibatch_size]
    lr_schedule            = learning_rate_schedule(lr_per_sample, units=epoch_size)
    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight          = 0.0001
    
    # start training only in one worker, and save the warm-up model for next step
    if parallelization.current_worker().global_rank == 0:
        warmup_train_manager.train(momentum_sgd(warmup_train_manager.z.parameters, lr_schedule, momentum_time_constant,
                                                l2_regularization_weight = l2_reg_weight), 
                                   minibatch_size, epoch_size, warmup_epochs,
                                   save_model_filename = warmup_model)
    
    # all workers need to sync before entering next step
    parallelization.barrier()
    
    train_manager = CIFAR10TrainManager(reader_train, reader_test, load_model_filename = warmup_model)

    # NOTE, the parallel step uses the same lr_schedule
    train_manager.train(momentum_sgd(train_manager.z.parameters, lr_schedule, momentum_time_constant,
                                     l2_regularization_weight = l2_reg_weight), 
                        minibatch_size, epoch_size, max_epochs - warmup_epochs)

    #
    # Evaluation action
    #
    epoch_size     = 10000
    minibatch_size = 16
    train_manager.test(minibatch_size, epoch_size)

    # clean up MPI, cannot call any Parallelization functions afterwards
    distributed.Parallelization.finalize()

if __name__ == '__main__':
    train_and_evaluate(max_epochs=5, warmup_epochs=1)
