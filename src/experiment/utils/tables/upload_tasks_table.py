from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base


def UploadTasksTable():
    Base = declarative_base(metadata=MetaData(schema="annotation"))

    class UploadTasksTable(Base):
        __tablename__ = "upload_tasks"

        id = Column(Integer, primary_key=True)
        patient_no = Column(String)
        full_name = Column(String)
        report_original = Column(String)
        report_english = Column(String, default=None)
        text = Column(String, default=None)
        report_date = Column(String, default=None)
        study_no = Column(String, default=None)
        report_count = Column(String, default=None)

    return UploadTasksTable, Base
