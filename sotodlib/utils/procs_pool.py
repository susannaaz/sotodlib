from __future__ import annotations
from typing import Tuple, Union

try:
    from mpi4py.futures import MPICommExecutor, as_completed
    from mpi4py import MPI
except Exception as e:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import set_start_method as mp_set_start_method


def get_exec_env(
    nprocs: int = None,
) -> tuple[int, Union["MPICommExecutor", "ProcessPoolExecutor"]]:
    try:
        comm = MPI.COMM_WORLD
    except NameError:
        comm = None
    # Set up logging.
    if comm is not None:
        rank = comm.Get_rank()
        max_workers = comm.Get_size() - 1 if nprocs is None else nprocs
        executor = MPICommExecutor(comm, root=0, max_workers=max_workers).__enter__()
    else:
        mp_set_start_method("spawn")
        rank = 0
        executor = ProcessPoolExecutor(max_workers=nprocs)
    return rank, executor
