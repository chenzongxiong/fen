import copy

from feng import utils
from feng import log as logging
from feng play import Play


LOG = logging.getLogger(__name__)
SENTINEL = -1


class Task(object):
    def __init__(self,
                 play=None,
                 func=None,
                 func_args=None):
        self.play = play
        self.func = func
        self.func_args = func_args

        if play is None:
            self.id = SENTINEL
        elif play._name.startswith('play-'):
            self.id = int(play._name.split('-')[-1])
        else:
            raise Exception("the format of play name isn't compatiable")

    def __getstate__(self):
        if self.play is None:
            return {
                'id': self.id
            }

        play_state = self.play.__getstate__()
        return {
            'id': self.id,
            'play': play_state,
            'func': self.func,
            'func_args': self.func_args
        }

    def __setstate__(self, d):
        cache = utils.get_cache()
        self.__dict__ = d
        if self.id == SENTINEL:
            return

        play_state = copy.deepcopy(d['play'])
        self.play = cache.get(play_state['_name'], None)

        if self.play is None:
            self.play = Play()
            self.play.__setstate__(play_state)
            cache[play_state['_name']] = self.play
            LOG.debug("Init play in sub-process")
        else:
            LOG.debug("Reuse play inside Cache.")

    def __hash__(self):
        return self.play.__hash__()
