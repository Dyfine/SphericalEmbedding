import os, sys, \
    subprocess, glob, re, \
    numpy as np, \
    logging, \
    collections, copy, \
    datetime
from os import path as osp
import time

root_path = osp.normpath(osp.join(osp.abspath(osp.dirname(__file__)), )) + '/'
sys.path.insert(0, root_path)

def set_stream_logger(log_level=logging.DEBUG):
    import colorlog
    sh = colorlog.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(
        colorlog.ColoredFormatter(
            ' %(asctime)s %(filename)s [line:%(lineno)d] %(log_color)s%(levelname)s%(reset)s %(message)s'))
    logging.root.addHandler(sh)

def set_file_logger(work_dir=None, log_level=logging.DEBUG):
    work_dir = work_dir or root_path
    fh = logging.FileHandler(os.path.join(work_dir, 'log-ing'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'))
    logging.root.addHandler(fh)

logging.root.setLevel(logging.INFO)
set_stream_logger(logging.DEBUG)

def shell(cmd, block=True, return_msg=True, verbose=True, timeout=None):
    import os
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    my_env['http_proxy'] = ''
    my_env['https_proxy'] = ''
    if verbose:
        logging.info('cmd is ' + cmd)
    if block:
        
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        if return_msg:
            msg = task.communicate(timeout)
            msg = [msg_.decode('utf-8') for msg_ in msg]
            if msg[0] != '' and verbose:
                logging.info('stdout {}'.format(msg[0]))
            if msg[1] != '' and verbose:
                logging.error('stderr {}'.format(msg[1]))
            return msg
        else:
            return task
    else:
        logging.debug('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        return task

def rm(path, block=True):
    path = osp.abspath(path)
    if not osp.exists(path):
        logging.info(f'no need rm {path}')
    stdout, _ = shell('which trash', verbose=False)
    if 'trash' not in stdout:
        dst = glob.glob('{}.bak*'.format(path))
        parsr = re.compile(r'{}.bak(\d+?)'.format(path))
        used = [0, ]
        for d in dst:
            m = re.match(parsr, d)
            if not m:
                used.append(0)
            elif m.groups()[0] == '':
                used.append(0)
            else:
                used.append(int(m.groups()[0]))
        dst_path = '{}.bak{}'.format(path, max(used) + 1)
        cmd = 'mv {} {} '.format(path, dst_path)
        return shell(cmd, block=block)
    else:
        return shell(f'trash -r {path}', block=block)

def mkdir_p(path, delete=True):
    path = str(path)
    if path == '':
        return
    if delete and osp.exists(path):
        rm(path)
    if not osp.exists(path):
        shell('mkdir -p ' + path)


class Logger(object):
    def __init__(self, fpath=None, console=sys.stdout):
        self.console = console
        self.file = None
        if fpath is not None:
            mkdir_p(os.path.dirname(fpath), delete=False)
            
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Timer(object):
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import cvbase as cvb
    >>> with cvb.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with cvb.Timer(print_tmpl='hey it taks {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    hey it taks 1.0 seconds
    >>> timer = cvb.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000

    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self, aux=''):
        """Total time since the timer is started.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        self._t_last = time.time()
        logging.info(f'{aux} time {self.print_tmpl.format(self._t_last - self._t_start)}')
        return self._t_last - self._t_start

    def since_last_check(self, aux='', verbose=True):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking operation.

        Returns(float): the time in seconds
        """
        if not self._is_running:
            raise ValueError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        if verbose:
            logging.info(f'{aux} time {self.print_tmpl.format(dur)}')
        return dur


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, maxlen=100):
        
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.mem = collections.deque(maxlen=maxlen)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.mem.append(val)
        self.avg = np.mean(list(self.mem))


timer = Timer()
logging.info('import myutils')
