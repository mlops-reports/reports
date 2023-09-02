# import pathlib
import os
import dotenv
import time

from experiment.utils import transformation

dotenv.load_dotenv()
LABEL_STUDIO_HOST = "https://label.drgoktugasci.com"
LABEL_STUDIO_LABEL_APP_NAME = "label-reports"
ANNOTATIONS_PATH = (
    transformation.get_project_root() / "data" / "output" / "annotations.json"
)


def start_label_studio(waiting_time: int = 15, app_name: str = LABEL_STUDIO_LABEL_APP_NAME) -> None:
    '''The function starts a Label Studio application on Heroku and waits for a specified amount of time
    before continuing.
    
    Parameters
    ----------
    waiting_time : int, optional
        The `waiting_time` parameter is an integer that represents the number of seconds to wait before
    executing the next line of code. It is used to introduce a delay in the program execution.
    app_name : str
        The `app_name` parameter is a string that represents the name of the Label Studio app on Heroku. It
    is used in the `os.system` command to scale the web dyno of the app to 1.
    
    '''
    os.system(f"heroku ps:scale web=1 --app {app_name}")
    time.sleep(waiting_time)


def stop_label_studio(app_name: str = LABEL_STUDIO_LABEL_APP_NAME) -> None:
    '''The function `stop_label_studio` stops the Label Studio application on Heroku by scaling down the
    web dynos to 0.
    
    Parameters
    ----------
    app_name : str
        The `app_name` parameter is a string that represents the name of the Label Studio application on
    Heroku.
    
    '''
    os.system(f"heroku ps:scale web=0 --app {app_name}")


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
