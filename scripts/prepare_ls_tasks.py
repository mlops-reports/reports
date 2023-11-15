# %%
import datetime
import time

import openai

from experiment.api import label_studio
from experiment.utils import dbutils, transformation
from experiment.utils.logging import logger
from experiment.utils.tables.upload_tasks_table import UploadTasksTable

# %%
db = dbutils.DatabaseUtils()

# %%
PROMPT_N_MORE_REPORTS = 200
PROMPT = "Perform the following transformation on the report: Translate into English"


# %%
reports_raw, Base = UploadTasksTable()

# %%
# get reports directly from database
query = """
            SELECT * FROM annotation.upload_tasks ut 
            ORDER BY patient_report_count DESC, report_length DESC 
        """

# get values from the database
df_reports = db.read_sql_query(query)
df_reports.head()

# %%
# get annotated reports
query = """
            SELECT 
                DISTINCT data ->> 'patient_no' as patient_no
            FROM task
            WHERE is_labeled = TRUE
        """

# get values from the database
annotated_patient_nos = db.read_sql_query(query)["patient_no"].to_list()

# %%
# get tasks that have been prompted
query = """
            SELECT 
                report_id
            FROM annotation.upload_tasks
            WHERE report_prompted != '' 
        """

# get values from the database
upload_tasks_prompted = db.read_sql_query(query)["report_id"].to_list()

# %%
# use only non-prompted reports & non-annotated patients
df_upload_tasks = (
    df_reports.loc[~df_reports["patient_no"].isin(annotated_patient_nos)]
    .loc[~df_reports["report_id"].isin(upload_tasks_prompted)]
    .head(PROMPT_N_MORE_REPORTS)
)

# %%
cols_to_upsert = df_upload_tasks.columns.to_list()
cols_to_upsert.remove("report_id")
data_to_insert = []
for _, row in df_upload_tasks.iterrows():
    try:
        data_to_insert.append(
            {
                "report_id": row["report_id"],
                "patient_no": row["patient_no"],
                "protocol_no": row["protocol_no"],
                "report_original": row["report_original"],
                "report_prompted": transformation.prompt_report(
                    report=row["report_original"], prompt=PROMPT
                ),
                "report_length": row["report_length"],
                "patient_report_count": row["patient_report_count"],
            }
        )

        db.upsert_values(reports_raw, data_to_insert, cols_to_upsert, ["report_id"])

        time.sleep(20)
    except openai.error.RateLimitError:
        # openai restriction: 3 RPM - 200 RPD
        logger.warning(f"Rate limit for: {datetime.datetime.now()}")

logger.info(f"Finished prompting {len(data_to_insert)} reports")

# %%
# get reports directly from database
query = """
            SELECT
                report_id,
                patient_no,
                protocol_no,
                report_original,
                report_prompted as text,
                report_length,
                patient_report_count
            FROM
                annotation.upload_tasks
            WHERE
                report_id NOT IN (
                SELECT
                    (DATA ->> 'report_id')::INT AS report_id
                FROM
                    public.task)
                AND report_prompted != ''
        """

# get values from the database
df_upload_tasks = db.read_sql_query(query)

# output tasks as a csv file
output_path = transformation.get_project_root() / "tmp" / "data" / "upload_tasks.csv"
df_upload_tasks.to_csv(output_path, index=False)

# %%
# upload tasks to label studio
label_studio.upload_csv_tasks(csv_path=output_path, project_id=7)


# %%
label_studio.stop_label_studio()
