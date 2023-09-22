import logging


# The Logger class is a Python class that sets up logging functionality, including console output and
# optional file output.
class Logger:
    def __init__(
        self,
        logging_level: str = "DEBUG",
        logger_name: str = "Reports",
        log_file: str = None,
    ):
        self.log_file = log_file
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, logging_level))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a file handler for writing logs to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Create a stream handler for displaying logs on the terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def log(self, message, level="INFO"):
        if level == "DEBUG":
            self.logger.debug(message)
        elif level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
