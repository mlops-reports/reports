from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base

def ReportsRawTable():

    Base = declarative_base(metadata=MetaData(schema="raw_data"))

    class ReportsRawTable(Base):
        __tablename__ = 'reports'

        report_id = Column(Integer, primary_key=True)
        patient_no = Column(String)
        protocol_no = Column(Integer)
        full_name = Column(String)
        report_original = Column(String)

    return ReportsRawTable, Base