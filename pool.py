# import pickle
import multiprocessing
import threading
import constants

def reload_tf(i):
   # import tensorflow as tf
   # tf.Session()
    pass

class ProcessPool(object):
    _pool = None
    _lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not cls._pool:
            with cls._lock:
                if not cls._pool:
                    cls._pool = multiprocessing.Pool(constants.CPU_COUNTS)
                    # cls._pool.map(reload_tf,
                    #          [x for x in range(constants.CPU_COUNTS)])

        return cls._pool
