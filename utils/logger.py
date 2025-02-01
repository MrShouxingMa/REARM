import logging
import os


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logger = logging.getLogger("normal")

    name_ = "{}-{}-lr_{}-rww_{}-nl_{}-sdp_{}-mdp_{}-clt_{}-diffw_{}-semw_{}.log"
    logfilename = name_.format(config.model_name, config.dataset, config.learning_rate,
                               config.reg_weight, config.n_layers,
                               config.s_drop, config.m_drop, config.cl_tmp,
                               config.diff_loss_weight, config.cl_loss_weight)
    logfilepath = os.path.join(LOGROOT, logfilename)
    filefmt = "%(asctime)-15s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"

    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(asctime)-15s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)

    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setFormatter(sformatter)

    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
