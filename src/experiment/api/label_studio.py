# import pathlib
import os
import dotenv
import time
import requests

from experiment.utils import transformation

dotenv.load_dotenv()
LABEL_STUDIO_HOST = "https://label.drgoktugasci.com"
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_LABEL_APP_NAME = "label-reports"
ANNOTATIONS_PATH = (
    transformation.get_project_root() / "data" / "output" / "annotations.json"
)

def check_host_up() -> bool:
    '''The function `check_host_up()` checks if the host is up by sending a HEAD request to the login page
    and returning True if the response status code is 200 or 302.
    
    Returns
    -------
        The function `check_host_up()` returns a boolean value indicating whether the host is up or not.
    
    '''
    response = requests.head(f"{LABEL_STUDIO_HOST}/user/login/")
    return response.status_code in [200, 302]

def start_label_studio(
    waiting_time: int = 15, app_name: str = LABEL_STUDIO_LABEL_APP_NAME
) -> None:
    """The function starts a Label Studio application on Heroku and waits for a specified amount of time
    before continuing.

    Parameters
    ----------
    waiting_time : int, optional
        The `waiting_time` parameter is an integer that represents the number of seconds to wait before
    executing the next line of code. It is used to introduce a delay in the program execution.
    app_name : str
        The `app_name` parameter is a string that represents the name of the Label Studio app on Heroku. It
    is used in the `os.system` command to scale the web dyno of the app to 1.

    """
    os.system(f"heroku ps:scale web=1 --app {app_name}")
    time.sleep(waiting_time)


def stop_label_studio(app_name: str = LABEL_STUDIO_LABEL_APP_NAME) -> None:
    """The function `stop_label_studio` stops the Label Studio application on Heroku by scaling down the
    web dynos to 0.

    Parameters
    ----------
    app_name : str
        The `app_name` parameter is a string that represents the name of the Label Studio application on
    Heroku.

    """
    os.system(f"heroku ps:scale web=0 --app {app_name}")


def upload_csv_tasks(csv_path: str, project_id: int) -> None:
    '''The function `upload_csv_tasks` uploads a CSV file to a Label Studio project using the Label Studio
    API.
    
    Parameters
    ----------
    csv_path : str
        The `csv_path` parameter is a string that represents the file path of the CSV file that you want to
    upload.
    project_id : int
        The `project_id` parameter is an integer that represents the ID of the project in which you want to
    upload the CSV tasks.
    
    '''

    if not check_host_up():
        start_label_studio()

    os.system(
        f"""
            curl \
            -H 'Authorization: Token {LABEL_STUDIO_API_KEY}' \
            -X POST '{LABEL_STUDIO_HOST}/api/projects/{project_id}/import' \
            -F 'file=@{csv_path}' \
        """
    )

    stop_label_studio()


def download_annotations() -> None:
    """The `download_annotations` function uses the `os.system` function to execute a curl command that
    downloads annotations from a Label Studio project and saves them to a specified file path.

    """
    os.system(
        f"""
            curl \
            -X GET {LABEL_STUDIO_HOST}/api/projects/4/export?exportType=JSON \
            -H 'Authorization: Token {LABEL_STUDIO_API_KEY}' \
            --output {ANNOTATIONS_PATH.as_posix()}
        """
    )
