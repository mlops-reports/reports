"""query.py: Includes various tools to interact with databases"""

__author__ = "Göktuğ Aşcı"

import datetime
import os

# import pathlib
# import zipfile
from typing import Any
from urllib import parse

import pandas as pd
import sqlalchemy as db


def convert_to_chunks(iterable: list, chunk_size: int) -> list:
    """The function `convert_to_chunks` takes an iterable and a chunk size as input, and returns a list of
    chunks where each chunk has the specified size.

    Parameters
    ----------
    iterable : list
        The `iterable` parameter is a list of elements that you want to convert into chunks. It can be any
    list of elements, such as numbers, strings, or even other lists.
    chunk_size : int
        The `chunk_size` parameter specifies the number of elements that should be included in each chunk.

    Returns
    -------
        The function `convert_to_chunks` returns a list of chunks, where each chunk is a sublist of the
    original iterable.

    """
    chunks = []

    for i in range(0, len(iterable), chunk_size):
        chunk = iterable[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


def build_query(text: str) -> str:
    """The `build_query` function removes unnecessary characters from a given text string.

    Parameters
    ----------
    text : str
        The `text` parameter is a string that represents a query or a piece of text that needs to be
    processed.

    Returns
    -------
        the modified text string.

    """
    text = text.replace("'NULL'", "NULL").replace(r"\'", "").replace(r"\"", "")

    return text


def get_db_engine(
    db_username: str = os.getenv("PSQL_DB_USERNAME"),
    db_password: str = os.getenv("PSQL_DB_PASSWORD"),
    db_host: str = os.getenv("PSQL_DB_HOST"),
    db_name: str = "report_labeling",
) -> db.Engine:
    """The function `get_db_engine` returns a SQLAlchemy engine object for connecting to a PostgreSQL
    database.

    Parameters
    ----------
    db_username : str
        The `db_username` parameter is the username used to authenticate with the PostgreSQL database.
    db_password : str
        The `db_password` parameter is the password used to authenticate the user specified in the
    `db_username` parameter when connecting to the PostgreSQL database.
    db_host : str
        The `db_host` parameter is the hostname or IP address of the PostgreSQL database server. It is used
    to specify the location of the database server where the database is hosted.
    db_name : str, optional
        The `db_name` parameter is the name of the database you want to connect to. In this case, the
    default value is set to "report_labeling".

    Returns
    -------
        The function `get_db_engine` returns a SQLAlchemy engine object.

    """
    db_username = parse.quote(db_username)
    db_password = parse.quote(db_password)
    db_host = parse.quote(db_host)
    db_name = parse.quote(db_name)
    database_uri = f"postgresql://{db_username}:{db_password}@{db_host}:5432/{db_name}"

    sqlalchemy_engine = db.create_engine(database_uri)

    return sqlalchemy_engine


def query_db(
    sql: str,
    db_username: str = os.getenv("PSQL_DB_USERNAME"),
    db_password: str = os.getenv("PSQL_DB_PASSWORD"),
    db_host: str = os.getenv("PSQL_DB_HOST"),
    db_name: str = "report_labeling",
    output_format: str = "dataframe",
) -> Any:
    """The `query_db` function executes a SQL query on a PostgreSQL database and returns the result in the
    specified output format, which is by default a Pandas DataFrame.

    Parameters
    ----------
    sql : str
        The `sql` parameter is a string that represents the SQL query you want to execute on the database.
    db_username : str
        The `db_username` parameter is used to specify the username for connecting to the database.
    db_password : str
        The `db_password` parameter is used to specify the password for accessing the database. It is
    expected to be a string value. In the code snippet, it is set to the value of the environment
    variable `PSQL_DB_PASSWORD` using `os.getenv()`.
    db_host : str
        The `db_host` parameter is used to specify the host or IP address of the database server. It is the
    location where the database is hosted and can be accessed.
    db_name : str, optional
        The `db_name` parameter is the name of the database you want to connect to. In this case, the
    default value is set to "report_labeling".
    output_format : str, optional
        The `output_format` parameter specifies the format in which the query results should be returned.

    Returns
    -------
        The function `query_db` returns the output of the SQL query in the specified format. If the
    `output_format` parameter is set to "dataframe", it returns a pandas DataFrame containing the query
    results. If the `output_format` is not "dataframe" or if no output format is specified, it returns
    `None`.

    """
    with get_db_engine(
        db_username=db_username,
        db_password=db_password,
        db_host=db_host,
        db_name=db_name,
    ).connect() as con:
        result = con.execute(db.text(build_query(sql)))
        data = result.fetchall()

        output = None

        if output_format == "dataframe":
            column_names = result.keys()  # Get column names from the result set
            output = pd.DataFrame(data, columns=column_names)

    return output


def get_sql_tuple(column_items: list[str]) -> str:
    """The function `get_sql_tuple` takes a list of column items and returns a string representation of a
    SQL tuple for use in a query.

    Parameters
    ----------
    column_items : list[str]
        A list of strings representing the items in a column.

    Returns
    -------
        a string that represents a SQL tuple.

    """
    if len(column_items) > 1:
        sql_tuple = f"in {tuple(column_items)}"
    else:
        sql_tuple = f"= '{column_items[0]}'"

    return sql_tuple


def filter_by_column(query: str, column_name: str, column_items: list[str]) -> str:
    """The function `filter_by_column` filters a SQL query by a specific column and its corresponding
    items.

    Parameters
    ----------
    query : str
        The `query` parameter is a string representing a SQL query. It is the main part of the query that
    you want to filter.
    column_name : str
        The `column_name` parameter is a string that represents the name of the column in a database table.
    column_items : list[str]
        The `column_items` parameter is a list of strings representing the values that you want to filter
    by in a specific column.

    Returns
    -------
        a modified query string.

    """

    query = f"{query} and {column_name} {get_sql_tuple(column_items)}"

    return query


def get_select_values(
    sql: str,
    db_username: str = os.getenv("PSQL_DB_USERNAME"),
    db_password: str = os.getenv("PSQL_DB_PASSWORD"),
    db_host: str = os.getenv("PSQL_DB_HOST"),
    db_name: str = "report_labeling",
    output_format: str = "dataframe",
) -> list[tuple]:
    """The function `get_select_values` retrieves data from a PostgreSQL database using a provided SQL
    query and returns the result in the specified output format.

    Parameters
    ----------
    sql : str
        The `sql` parameter is a string that represents the SQL query you want to execute on the database.
    db_username : str
        The `db_username` parameter is used to specify the username for connecting to the PostgreSQL
    database.
    db_password : str
        The `db_password` parameter is used to specify the password for the database user. It is used to
    authenticate and establish a connection to the database.
    db_host : str
        The `db_host` parameter is used to specify the host or IP address of the PostgreSQL database
    server. It is used to establish a connection to the database server.
    db_name : str, optional
        The `db_name` parameter is the name of the database you want to connect to. In this case, the
    default value is set to "report_labeling".
    output_format : str, optional
        The `output_format` parameter specifies the format in which the query results should be returned.
    It has a default value of "dataframe", which means that the query results will be returned as a
    pandas DataFrame object.

    Returns
    -------
        The function `get_select_values` returns the output of the `query_db` function, which is a list of
    tuples.

    """
    output = query_db(
        sql=sql,
        db_username=db_username,
        db_password=db_password,
        db_host=db_host,
        db_name=db_name,
        output_format=output_format,
    )

    return output


def upsert_values(
    schema_name: str,
    table_name: str,
    select_cols: dict,
    constraint: str,
    cols_to_upsert: list[str],
    values: list[tuple],
    timestamp_col_name: str = None,
    engine: db.Engine = None,
) -> Any:
    """The `upsert_values` function performs an upsert operation (insert or update) on a specified table in
    a database, using the provided values and constraints.

    Parameters
    ----------
    schema_name : str
        The `schema_name` parameter is a string that represents the name of the database schema where the
    table is located.
    table_name : str
        The `table_name` parameter is a string that represents the name of the table in the database where
    the upsert operation will be performed.
    select_cols : dict
        The `select_cols` parameter is a dictionary that specifies the columns to be selected from the
    table.
    constraint : str
        The `constraint` parameter is a string that specifies the conflict resolution constraint for the
    upsert operation. It is used to determine which columns or combination of columns should be used to
    identify conflicts and update existing rows instead of inserting new ones.
    cols_to_upsert : list[str]
        The `cols_to_upsert` parameter is a list of column names that should be updated if a conflict
    occurs during the upsert operation. These columns will be included in the `set_` clause of the
    `on_conflict_do_update` method.
    values : list[tuple]
        The `values` parameter is a list of tuples representing the values to be upserted into the table.
    Each tuple should contain the values for each column in the order specified by the `select_cols`
    parameter.
    timestamp_col_name : str
        The `timestamp_col_name` parameter is used to specify the name of the column that stores the
    timestamp or last update time in the database table.
    engine : db.Engine
        The `engine` parameter is used to specify the db connection

    Returns
    -------
        the response from executing the query on the database engine.

    """

    metadata = db.MetaData(bind=engine if engine else get_db_engine())

    if timestamp_col_name == "Django":
        set_values: dict = {
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
        }

        cols_to_upsert.remove(timestamp_col_name)
        cols_to_upsert.append("updated_at")

        select_cols["created_at"] = {"type": "DateTime"}
        select_cols["updated_at"] = {"type": "DateTime"}
    else:
        if timestamp_col_name is None:
            timestamp_col_name = "last_update"

        set_values: dict = {timestamp_col_name: datetime.datetime.utcnow()}
        cols_to_upsert.append(timestamp_col_name)
        select_cols[timestamp_col_name] = {"type": "DateTime"}

    col_builders = [
        db.Column(
            col_name,
            getattr(db, col_data["type"]),
            primary_key=col_data.get("primary_key", False),
        )
        for col_name, col_data in select_cols.items()
    ]

    table = db.Table(
        table_name,
        metadata,
        *col_builders,
        schema=schema_name,
    )

    query = db.dialects.postgresql.insert(table).values(values)

    for col in cols_to_upsert:
        set_values[col] = getattr(query.excluded, col)

    query = query.on_conflict_do_update(constraint=constraint, set_=set_values)

    response = engine.execute(query)

    return response


def get_unique_value_col(df: pd.DataFrame, col_name: str) -> int:
    """The function `get_unique_value_col` returns the unique value in a specified column of a pandas
    DataFrame, or raises a ValueError if the column values are not unique.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the data.
    col_name : str
        The `col_name` parameter is a string that represents the name of the column in the DataFrame `df`
    for which you want to get the unique value.

    Returns
    -------
        the unique value in the specified column of the given DataFrame.

    """
    if df[col_name].nunique() != 1:
        raise ValueError("Column value is not Unique!")

    return list(df[col_name].unique())[0]
