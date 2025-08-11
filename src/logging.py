from datetime import datetime
from pathlib import Path
import logging
import sys

class PML_Logger:
    def __init__(
        self,
        name_prefix: str = "UnifiedLogger",      # variable part of the logger’s name
        log_dir: str | Path = "../logs",         # directory that will hold the log files
        log_level: int = logging.INFO,
    ):
        """
        Initialize a unique logger for the current run.  
        The logger (and its file) will be named:

            <name_prefix>_<YYMMDD_HHMMSS>

        Parameters
        ----------
        name_prefix : str
            A prefix that identifies your application or experiment.
        log_dir : str | pathlib.Path
            Folder where the log file will be created (will be created if missing).
        log_level : int
            Minimum level to be recorded (INFO, DEBUG, ERROR, …).
        """
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        full_name = f"{name_prefix}_{stamp}"          # final logger name

        self.logger = logging.getLogger(full_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False                 # avoid double logging to root

        # -------- Formatting --------
        fmt = (
            "%(asctime)s - %(levelname)s:\n"
            "  [%(filename)s]:%(lineno)d- %(message)s"
        )
        formatter = logging.Formatter(fmt)

        # -------- Handlers (console + file) --------
        if not self.logger.handlers:                  # prevent duplicates on import
            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

            # File handler
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{full_name}.log"   # same name as the logger
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger