"""dbutils.py: Includes various tools to interact with databases"""

__author__ = "Göktuğ Aşcı"

import os
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import sqlalchemy


class DatabaseUtils:
    """
    Start and manage all Postgresql database connections.

    Examples
    --------
    >>> dbutils = DatabaseUtils()
    >>> dbutils.connect_database()
    >>> df = dbutils.pandas_read_sql_table("annotations")
    >>> df = dbutils.pandas_read_sql_query("select * from annotations")
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        host : str
            Parameter used to specify the host or IP address of the database server. It is the
        port:str

        username : str
            Parameter used to specify the username for connecting to the database.
        password : str
            Parameter used to specify the password for accessing the database. It is
        expected to be a string value. In the code snippet, it is set to the value of the environment
        variable `PSQL_DB_PASSWORD` using `os.getenv()`.
        location where the database is hosted and can be accessed.
        db_name : str, optional
            The `db_name` parameter is the name of the database you want to connect to. In this case, the
        default value is set to "report_labeling".

        """
        self.host: str = os.environ.get("PSQL_DB_HOST", "localhost")
        self.port: Union[str, int] = os.environ.get("PSQL_DB_PORT", 5432)
        self.username: str = os.environ.get("PSQL_DB_USERNAME", "")
        self.password: str = os.environ.get("PSQL_DB_PASSWORD", "")
        # Database name can also be env variable.
        self.database: str = "report_labeling"

        self.engine: Optional[sqlalchemy.engine.Engine]
        self.metadata: Optional[sqlalchemy.MetaData]

    def _build_connection_string(self, database: Optional[str] = None) -> str:
        """Build and store a connection string based on the provided parameters."""
        if database is not None:
            self.database = database

        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def connect_database(self, database: Optional[str] = None) -> None:
        """Connect to another database or the predefined one."""
        if database is not None:
            self.database = database

        self.connection_string = self._build_connection_string()
        self.engine = sqlalchemy.create_engine(self.connection_string)
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

    def close_connection(self) -> None:
        """Safely closes connection and removes metadata object."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
        if self.metadata is not None:
            self.metadata = None

    def refresh_connection(self) -> None:
        """Closes database connection and starts a new one."""
        self.close_connection()
        self.connect_database()

    def pandas_read_sql_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Reads the specified table from postgres database."""
        if self.engine is None:
            self.refresh_connection()
            assert self.engine is not None

        with self.engine.connect() as connection:
            try:
                df = pd.read_sql_table(table_name, connection)
            except Exception as e:
                print(e)
                return None
        return df

    def pandas_read_sql_query(self, query: str) -> pd.DataFrame:
        """"""
        # Same function with pandas_read_sql_table but with sql query directly. Maybe should be handled together?
        pass

    def _build_sql_query(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[Tuple[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> str:
        """
        Based on the specified parameters, build and return an sql query.
        Handle where clause, columns and column items here.

        Parameters
        ----------

        Returns
        -------

        """
        if columns is not None:
            columns_string = ",".join(columns)
            query = f"SELECT {columns_string} FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        if where is not None:
            where_col, where_val = where
            if isinstance(where_val, int) or isinstance(where_val, float):
                query += f" WHERE {where_col}={where_val}"
            elif isinstance(where_val, str):
                query += f" WHERE {where_col}='{where_val}'"
            else:
                raise ValueError("Unsupported where clause value type.")

            # TODO: Add filter_by_column and column items when needed here.
            # if len(column_items) > 1:
            #     query += f"AND {column_name} in {tuple(column_items)}"
            # else:
            #     query += f"AND {column_name} = '{column_items[0]}'"

        if limit is not None:
            query += f" LIMIT {limit}"

        if offset is not None:
            query += f" OFFSET {offset}"

        return query

    def pandas_read_table_in_chunks(
        self, table_name: str, chunk_size: int, chunk_idx: int
    ) -> pd.DataFrame:
        """
        Lazy load for the database read operation.

        Parameters
        ----------
        table_name: str
            Requested table.
        chunk_size: int
            Requested max data size.
        chunk_idx: int
            Skips the rows of the previous chunks.

        """
        limit, offset = chunk_size, chunk_idx * chunk_size
        query = self._build_sql_query(table_name, limit=limit, offset=offset)
        return self.pandas_read_sql_query(query)

    def get_select_values(
        self,
        query: str,
        output_format: str = "dataframe",
    ) -> Union[Dict[Any, Any], pd.DataFrame]:
        """The function `get_select_values` retrieves data from a PostgreSQL database using a provided SQL
        query and returns the result in the specified output format.

        Parameters
        ----------
        query : str
            The `sql` parameter is a string that represents the SQL query you want to execute on the database.
        output_format : str, optional
            The `output_format` parameter specifies the format in which the query results should be returned.
        It has a default value of "dataframe", which means that the query results will be returned as a
        pandas DataFrame object.

        Returns
        -------
            The function `get_select_values` returns the output of the `query_db` function, which is a list of
        tuples.

        """
        if output_format == "dataframe":
            return self.pandas_read_sql_query(query)
        else:
            raise NotImplementedError

    def upsert_values(
        self,
        schema_name: str,
        table_name: str,
        select_cols: dict,
        constraint: str,
        cols_to_upsert: list[str],
        values: list[tuple],
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
        db_engine : db.Engine
            The `db_engine` parameter is used to specify the db connection

        Returns
        -------
            the response from executing the query on the database engine.

        """
        if self.metadata is None or self.engine is None:
            self.refresh_connection()
            assert self.metadata is not None
            assert self.engine is not None

        set_values: dict = {}

        # if timestamp_col_name == "Django":
        #     set_values: dict = {
        #         "created_at": datetime.datetime.utcnow(),
        #         "updated_at": datetime.datetime.utcnow(),
        #     }

        #     cols_to_upsert.remove(timestamp_col_name)
        #     cols_to_upsert.append("updated_at")

        #     select_cols["created_at"] = {"type": "DateTime"}
        #     select_cols["updated_at"] = {"type": "DateTime"}
        # else:
        #     if timestamp_col_name is None:
        #         timestamp_col_name = "last_update"

        #     set_values: dict = {timestamp_col_name: datetime.datetime.utcnow()}
        #     cols_to_upsert.append(timestamp_col_name)
        #     select_cols[timestamp_col_name] = {"type": "DateTime"}

        col_builders = [
            sqlalchemy.Column(
                col_name,
                getattr(sqlalchemy, col_data["type"]),
                primary_key=col_data.get("primary_key", False),
            )
            for col_name, col_data in select_cols.items()
        ]

        table = sqlalchemy.Table(
            table_name,
            self.metadata,
            *col_builders,
            schema=schema_name,
        )

        query = sqlalchemy.dialects.postgresql.insert(table).values(values)

        for col in cols_to_upsert:
            set_values[col] = getattr(query.excluded, col)

        query = query.on_conflict_do_update(constraint=constraint, set_=set_values)

        with self.engine.connect() as conn:
            try:
                conn.execute(query)
                conn.commit()
                is_success = True
            except Exception as e:
                print("Data write failed, rolling back...")
                print(f"Exception: {e}")
                conn.rollback()
                is_success = False

        return is_success

    @staticmethod
    def convert_to_chunks(iterable: List[Any], chunk_size: int) -> list:
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

    @staticmethod
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
