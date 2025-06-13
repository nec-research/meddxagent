import logging
from pathlib import Path
import json
from typing import Union

from ddxdriver.utils import find_project_root

# Global variables to store the log file path and initialization status
log_file_path = None
file_logging_enabled = False
console_logging_enabled = False

# Handlers declared globally but not initialized until logging is initialized
file_handler = None
stream_handler = None

# Create a log
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

class ConditionalFormatter(logging.Formatter):
    def __init__(self, format_warning_or_above, format_info_or_below, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_warning_or_above = format_warning_or_above
        self.format_info_or_below = format_info_or_below

    def format(self, record: logging.LogRecord) -> str:
        # Select the appropriate format based on the log level
        if record.levelno >= logging.WARNING:
            fmt = self.format_warning_or_above
        else:
            fmt = self.format_info_or_below
        # Apply the selected format
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


# Create formatter
conditional_formatter = ConditionalFormatter(
    format_warning_or_above="%(levelname)s - %(module)s - line %(lineno)d - %(asctime)s\n%(message)s\n",
    format_info_or_below="%(message)s",
)


class HTTPRequestFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress log messages starting with "HTTP Request:"
        return not record.getMessage().startswith("HTTP Request:")


def enable_logging(
    console_logging: bool = True,
    file_logging: bool = False,
):
    """
    Sets up the logging (console_logging and/or file_logging).
    To enable file_logging, must call set_file_handler at least once before.
    Params:
    console_logging: whether to set up and use console logging
    file_logging: whether to set up and use file logging
    - If True and the file_handler is not set up yet, will raise an error. Set up with set_file_handler()
    """
    enable_file_logging(file_logging)
    enable_console_logging(console_logging)


def set_stream_handler():
    global stream_handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(conditional_formatter)
    stream_handler.addFilter(HTTPRequestFilter())  # Apply filter to console handler
    stream_handler.setLevel(logging.INFO)


def set_file_handler(file_path: Union[str, Path], mode: str = "a"):
    """
    Resets a file handler given a file_path. Will create the Path if it doesn't exist + is in valid format
    If you created a prior file_handler, you do not need to call enable_logging again (it will continue logging but at your new file)
    Params:
        file_path: path to log
        mode: whether to set the file_handler in append or write model. By default, append.
        - write = "w"
        - append = "a"

    Error:
        Will not work if the file already exists
    """
    global log_file_path, file_handler

    try:
        # Convert the input path to a Path object
        log_file_path = Path(file_path)
        # Ensure the parent directories exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Update the file handler with the new path
        if "file_handler" in globals():  # Check if file_handler is already created
            log.removeHandler(file_handler)
        file_handler = logging.FileHandler(log_file_path, mode=mode)
        file_handler.setFormatter(conditional_formatter)
        file_handler.addFilter(HTTPRequestFilter())  # Apply filter to console handler
        file_handler.setLevel(logging.INFO)

        # If file logging is enabled, update the handler in the log
        if file_logging_enabled:
            log.addHandler(file_handler)
            log.debug(f"Log file path set to: {log_file_path}")
    except:
        raise ValueError(
            f"Trying to set file handler but {file_path} is not a valid path format, either call set_file_handler or set_log to do so"
        )


def enable_file_logging(enable=True):
    """
    Enables or disables file logging
    """
    global file_logging_enabled
    if enable and not file_logging_enabled:
        log.addHandler(file_handler)
        file_logging_enabled = True
        log.debug("File logging enabled.")
    elif not enable and file_logging_enabled:
        if "file_handler" in globals():
            log.removeHandler(file_handler)
        file_logging_enabled = False
        log.debug("File logging disabled.")


def enable_console_logging(enable=True):
    """
    Enables or disables console logging
    """
    global console_logging_enabled
    if enable and not console_logging_enabled:
        set_stream_handler()
        log.addHandler(stream_handler)
        console_logging_enabled = True
        log.debug("Console logging enabled.")
    elif not enable and console_logging_enabled:
        log.removeHandler(stream_handler)
        console_logging_enabled = False
        log.debug("Console logging disabled.")


def log_json_data(
    json_data: dict,
    file_path: Union[str, Path] = None,
):
    """
    Tries to log json results to a JSON file in the specified file_path. Logging should be specified before the function.
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(
            f"Eror accessing or creating file_path, returning early from log_json_results:\n{e}"
        )

    try:
        # Try to serialize the dictionary to JSON and write to the file
        with file_path.open("w") as json_file:
            json.dump(json_data, json_file, indent=4)

        log.info(f"Successfully logged json data to {file_path}\n")

    except (TypeError, ValueError) as e:
        # Log the error and fallback to logging the data as a plain text file
        log.error(
            f"Error serializing data to JSON, instead logging as a string to a text file: {e}"
        )
        file_path = file_path.with_suffix(".txt")
        with file_path.open("w") as text_file:
            text_file.write(str(json_data))
        log.info(f"Logged json data to text file instead at {file_path}\n")
