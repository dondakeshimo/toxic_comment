import time
from termcolor import cprint


def show_elapsed_time(task, start, prev):
    lap = time.time() - start - prev
    elapsed = time.time() - start
    cprint("{:22}: {:10.2f}[sec]{:10.2f}[sec]".format(task, lap, elapsed),
           "blue")
    return elapsed
