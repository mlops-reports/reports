import logging


# The Logger class is a Python class that sets up logging functionality, including console output and
# optional file output.
class Logger:
    def __init__(
        self,
        logging_level: str = "DEBUG",
        logger_name: str = "Reports",
        output_path: str = None,
    ):
        self.logger = logging.getLogger(logger_name)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Stream Handler for console output
        sh = logging.StreamHandler()
        self.logger.setLevel(getattr(logging, logging_level))
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)

        if output_path:
            self.output_to_file(output_path)

    def output_to_file(self, output_path: str):
        """The function `output_to_file` adds a file handler to the logger to log messages to a specified
        output file.

        Parameters
        ----------
        output_path : str
            The `output_path` parameter is a string that represents the file path where the log messages
        will be written to.

        """
        fh = logging.FileHandler(output_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
