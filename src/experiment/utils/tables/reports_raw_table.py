from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base

def ReportsRawTable():

    Base = declarative_base(metadata=MetaData(schema="raw_data"))

    class ReportsRawTable(Base):
        __tablename__ = 'reports'

        id = Column(Integer, primary_key=True)
        patient_no = Column(String)
        full_name = Column(String)
        report_original = Column(String)
        report_english = Column(String, default=None)

    return ReportsRawTable, Base