import logging, coloredlogs
import os, json
import numpy as np

CACHE_PATH = "".join((os.getcwd().split("PCNN")[0], "/PCNN/src/cache"))


def setup_logger(name: str="MAIN",
                 colored: bool=True,
                 level: int=0,
                 is_debugging: bool=True,
                 is_warning: bool=True) -> logging.Logger:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True
    level : int
        the level that is currently used.
        Default=0
    is_debugging : bool
        use debugging mode. Default=True
    is_warning : bool
        use warning mode. Default=True

    Returns
    -------
    logger : object
        logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create a custom formatter
    if colored:
        formatter = coloredlogs.ColoredFormatter(
            f"{name} | %(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # create a colored stream handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # add the handler to the logger and disable propagation
        logger.addHandler(handler)

    logger.propagate = False

    # wrapper class
    class LoggerWrapper:
        def __init__(self, logger,
                     level: int,
                     is_debugging: bool=False,
                     is_warning: bool=False):
            self.logger = logger
            self.level = level
            self.is_debugging = is_debugging
            self.is_warning = is_warning

            # self.logger.info(self)

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name}," + \
                   f"level={self.level}, " + \
                   f"debugging={self.is_debugging}, " + \
                   f"warning={self.is_warning})"

        def __call__(self, msg: str="", level: int=1):
            if level <= self.level:
                self.logger.info(msg)

        def info(self, msg: str="", level: int=1):
            self(msg, level)

        def warning(self, msg: str=""):
            if self.is_warning:
                self.logger.warning(msg)

        def error(self, msg: str=""):
            if self.is_warning:
                self.logger.error(msg)

        def debug(self, msg):
            if self.is_debugging:
                self.logger.debug(msg)

        def set_debugging(self, is_debugging: bool):
            self.is_debugging = is_debugging

        def set_warning(self, is_warning: bool):
            self.is_warning = is_warning

        def set_level(self, level: int):
            self.level = level

    return LoggerWrapper(logger=logger, level=level,
                         is_debugging=is_debugging,
                         is_warning=is_warning)


def edit_logger(level: int=-1,
                is_debugging: bool=True,
                is_warning: bool=False):
    global logger
    logger.set_level(level)
    logger.set_debugging(is_debugging)
    logger.set_warning(is_warning)


def load_session():

    logger = setup_logger(name="UTILS", colored=True,
                          level=2, is_debugging=True,
                          is_warning=False)

    logger("\n----\nLoading from evolution")

    # --- select file ---

    files = os.listdir(CACHE_PATH)

    # sort the files by name
    files = sorted(files, key=lambda x: int(x.split("_")[0]))

    for i, file in enumerate(files):
        print(f"{i}: {file}")

    ans = input("\n>Select file: ")
    idx = -1 if ans == "" else int(ans)

    # --- load file ---

    with open(f"{CACHE_PATH}/{files[idx]}", "r") as f:
        run_data = json.load(f)

    logger(f"[] Loaded session: {files[idx]}")
    logger(f"[] Note: {run_data['info']['other']}")
    logger("[] Agent fitness=" + \
        f"{run_data['info']['record_genome']['0']['fitness']:.3f}")

    parameters = run_data["info"]["record_genome"]["0"]["genome"]
    session_config = run_data["info"]["data"]

    return parameters, session_config


def load_parameters(idx: int=None):

    logger = setup_logger(name="UTILS", colored=True,
                          level=2, is_debugging=True,
                          is_warning=False)

    logger("\n----\nLoading from evolution")

    # load and sort files
    files = os.listdir(CACHE_PATH)
    files = sorted(files, key=lambda x: int(x.split("_")[0]))

    if idx is None:

        for i, file in enumerate(files):
            print(f"{i}: {file}")

        ans = input("\n>Select file: ")
        idx = -1 if ans == "" else int(ans)

    with open(f"{CACHE_PATH}/{files[idx]}", "r") as f:
        run_data = json.load(f)

    logger(f"Loaded {files[idx]}")
    logger(f"Agent fitness={run_data['info']['record_genome']['0']['fitness']:.3f}")
    logger(f"Agent fitness={run_data['info']['record_genome']['0']['genome']}")

    return run_data["info"]["record_genome"]["0"]["genome"]


def create_lattice(L, dx):
    # Calculate the number of points
    num_points = int(L / dx) + 1

    # Create a 1D array of points
    points = np.arange(0, L + dx, dx)

    # Create a 2D grid from the 1D array of points
    X, Y = np.meshgrid(points, points)

    points = []
    num_rows = X.shape[0]

    for i in range(num_rows):
        if i % 2 == 0:
            # Even row: left to right
            row_points = list(zip(X[i], Y[i]))
        else:
            # Odd row: right to left
            row_points = list(zip(X[i][::-1], Y[i][::-1]))

        points.extend(row_points)

    return points

