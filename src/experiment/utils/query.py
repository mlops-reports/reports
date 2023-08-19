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
    """
    "For each chunk_size, take the next chunk_size elements from the iterable and append
    them to the chunks list."

    Args:
      iterable (list): the list you want to split into chunks
      chunk_size (int): The size of each chunk.

    Returns:
      A list of lists.
    """

    chunks = []

    for i in range(0, len(iterable), chunk_size):
        chunk = iterable[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


def build_query(text: str) -> str:
    """
    It replaces the string 'NULL' with the string NULL

    Args:
        text: The text of the query to be executed.

    Returns:
        The query is being returned.
    """

    text = text.replace("'NULL'", "NULL").replace(r"\'", "").replace(r"\"", "")

    return text


def get_db_engine(
    db_username: str = os.getenv("PSQL_DB_USERNAME"),
    db_password: str = os.getenv("PSQL_DB_PASSWORD"),
    db_host: str = os.getenv("PSQL_DB_HOST"),
    db_name: str = "report_labeling",
) -> object:
    """
    > This function returns a database engine object that can be used to connect to the
    database

    Args:
      db_username (str): The username for the database.
      db_password (str): The password for the database.
      db_host (str): The hostname of the database server.
      db_name (str): The name of the db that will be used during login
    """
    db_username = parse.quote(db_username)
    db_password = parse.quote(db_password)
    db_host = parse.quote(db_host)
    db_name = parse.quote(db_name)
    database_uri = (
        f"postgresql://{db_username}:{db_password}@{db_host}:5432/{db_name}"
    )

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
    """
    `query_db` queries the database and returns the result

    Args:
      sql (str): the SQL query you want to run
      db_username (str): The username for the database.
      db_password (str): The password for the database.
      db_host (str): The hostname of the database server. Defaults to 80.158.16.22
      db_name (str): The name of the db that will be used during login
      output_format (str): Format of the output. It can be list, or dataframe
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

        if output_format == 'dataframe':
            column_names = result.keys()  # Get column names from the result set
            output = pd.DataFrame(data, columns=column_names)

    return output


def get_sql_tuple(column_items: list[str]) -> str:
    """
    This function takes a list of strings and returns a SQL tuple string representation of
    the list.

    Args:
      column_items (list[str]): The parameter `column_items` is a list of strings
    representing the values of a column in a database table.

    Returns:
      The function `get_sql_tuple` returns a string that represents a SQL tuple.
    """
    if len(column_items) > 1:
        sql_tuple = f"in {tuple(column_items)}"
    else:
        sql_tuple = f"= '{column_items[0]}'"

    return sql_tuple


def filter_by_column(query: str, column_name: str, column_items: list[str]) -> str:
    """
    It takes a query, a column name, and a list of items, and returns a query with the column name and
    items added to the WHERE clause

    Args:
      query (str): the query string that you want to filter (must accept `and statement in the end` or `where true`)
      column_name (str): The name of the column you want to filter by.
      column_items (list[str]): a list of strings that are the values you want to filter by

    Returns:
      A enhanced query
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
    """
    > This function takes a SQL select query and returns the results of that query as a list of
    tuples

    Args:
      sql (str): the SQL query you want to run
      db_username (str): The username for the database.
      db_password (str): The password for the database.
      db_host (str): The hostname of the database server. Defaults to 80.158.16.22
      db_name (str): The name of the db that will be used during login
      output_format (str): Format of the output. It can be list, or dataframe

    Returns:
      A list of tuples.
    """

    output = query_db(
        sql=sql,
        db_username=db_username,
        db_password=db_password,
        db_host=db_host,
        db_name=db_name,
        output_format=output_format
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
) -> Any:
    """
    > Inserts or updates values in a table, using a unique constraint to determine whether to
    insert or update. This method requires a unique constraint on the table

    Args:
      schema_name (str): The name of the schema that the table is in
      table_name (str): The name of the table to insert or update values in
      select_cols (dict): A dictionary of column names and their types
      constraint (str): The name of the column that is used to determine whether to insert
    or update
      cols_to_upsert (list[str]): A list of column names that should be updated
      values (list[tuple]): A list of tuples, where each tuple contains the values to insert or update
      timestamp_col_name (str): The name of the timestamp column
    """
    engine = get_db_engine()
    metadata = db.MetaData(bind=engine)

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
    """
    It takes a dataframe and a column name, and returns the unique value in that column

    Args:
    df (pd.DataFrame): the dataframe to be processed
    col_name (str): The name of the column to get the value from.

    Returns:
    The number of unique values in the column.
    """
    if df[col_name].nunique() != 1:
        raise ValueError("Column value is not Unique!")

    return list(df[col_name].unique())[0]



