from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base


def UploadTasksTable():
    Base = declarative_base(metadata=MetaData(schema="annotation"))

    class UploadTasksTable(Base):
        __tablename__ = "upload_tasks"

        report_id = Column(Integer, primary_key=True)
        patient_no = Column(String)
        protocol_no = Column(Integer)
        report_original = Column(String)
        report_prompted = Column(String, default=None)
        report_length = Column(Integer, default=None)
        patient_report_count = Column(Integer, default=None)

    return UploadTasksTable, Base
