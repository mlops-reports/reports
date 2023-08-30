# import pathlib
import os
import dotenv
import time

from experiment.utils import transformation

dotenv.load_dotenv()
LABEL_STUDIO_HOST = "https://label.drgoktugasci.com"
ANNOTATIONS_PATH = (
    transformation.get_project_root() / "data" / "output" / "annotations.json"
)


def start_label_studio(waiting_time: int = 15) -> None:
    """The function starts the Label Studio application on Heroku and waits for a specified amount of time.

    Parameters
    ----------
    waiting_time : int, optional
        The `waiting_time` parameter is an integer that represents the number of seconds to wait before
    executing the next line of code. In this case, it is used to pause the execution of the program for
    a specified amount of time (in seconds) before continuing.

    """
    os.system("heroku ps:scale web=1 --app er-reports")
    time.sleep(waiting_time)


def stop_label_studio() -> None:
    """The function `stop_label_studio()` stops the Label Studio application running on Heroku by scaling
    down the web dynos to 0.

    """
    os.system("heroku ps:scale web=0 --app er-reports")


def download_annotations() -> None:
    """The `download_annotations` function uses the `os.system` function to execute a curl command that
    downloads annotations from a Label Studio project and saves them to a specified file path.

    """
    os.system(
        f"""
            curl \
            -X GET {LABEL_STUDIO_HOST}/api/projects/4/export?exportType=JSON \
            -H 'Authorization: Token {os.getenv("LABEL_STUDIO_API_KEY")}' \
            --output {ANNOTATIONS_PATH.as_posix()}
        """
    )
