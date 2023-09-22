"""dbutils.py: Includes various tools to interact with databases"""

__author__ = "Göktuğ Aşcı"

import os
from typing import Any, List, Optional, Union

import pandas as pd

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert


class QueryError(Exception):
    pass


class DatabaseUtils:
    """
    Start and manage all Postgresql database connections.

    Examples
    --------
    >>> dbutils = DatabaseUtils()
    >>> dbutils.connect_database()
    >>> df = dbutils.read_sql_query("select * from annotations")
    """

    def __init__(
        self,
        auto_connect: bool = True,
        db_host: str = os.environ.get("PSQL_DB_HOST", "localhost"),
        db_port: Union[str, int] = os.environ.get("PSQL_DB_PORT", 5432),
        db_username: str = os.environ.get("PSQL_DB_USERNAME", ""),
        db_password: str = os.environ.get("PSQL_DB_PASSWORD", ""),
        db_name: str = "report_labeling",
    ) -> None:
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
        self.host: str = db_host
        self.port: Union[str, int] = db_port
        self.username: str = db_username
        self.password: str = db_password
        self.database: str = db_name

        self.engine: Optional[sqlalchemy.engine.Engine]
        self.metadata: Optional[sqlalchemy.MetaData]
        self.session: Optional[sqlalchemy.orm.Engine]

        if auto_connect:
            self.connect_database()

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
        self.session = sessionmaker(bind=self.engine)()
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

    def read_sql_query(
        self, sql: str, output_format: str = "dataframe", db_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """The function `read_sql_query` is used to execute a SQL query on a database and return the result in a
        specified output format, such as a pandas DataFrame.

        Parameters
        ----------
        sql : str
            The `sql` parameter is a string that represents the SQL query you want to execute on the
        database. It can be only valid SELECT SQL statement.
        output_format : str, optional
            The `output_format` parameter specifies the format in which the query results should be
        returned. The default value is "dataframe", which means that the query results will be returned
        as a pandas DataFrame.
        db_name : str
            The `db_name` parameter is used to specify the name of the database you want to query. If you
        provide a value for `db_name`, it will update the `self.database` attribute of the object and
        refresh the connection to the database using the new database name.

        Returns
        -------
            the output of the database query in the specified format. If the output format is "dataframe",
        it returns a pandas DataFrame containing the query results. If the output format is not
        "dataframe", it raises a NotImplementedError.

        """

        if db_name is not None:
            self.database = db_name
            self.refresh_connection()

        if self.engine is None:
            self.refresh_connection()
            assert self.engine is not None

        if not sql.lower().strip().startswith("select"):
            raise QueryError(
                "An SQL write statement passed to the read operation function."
            )

        with self.engine.connect() as connection:
            try:
                result = connection.execute(sqlalchemy.text(sql))
                data = result.fetchall()
            except Exception as e:
                print("Data read operation failed.")
                print(e)
            output = None

            if output_format == "dataframe":
                column_names = result.keys()  # Get column names from the result set
                output = pd.DataFrame(data, columns=column_names)
            else:
                raise NotImplementedError

        return output

    def _build_sql_query_chunk(
        self,
        table_name: str,
        schema_name: str = "dbo",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> str:
        """The `_build_sql_query_chunk` function builds a SQL query string based on the provided schema
        name, table name, columns, limit, and offset.

        Parameters
        ----------
        schema_name : str
            The `schema_name` parameter is a string that represents the name of the database schema where
        the table is located.
        table_name : str
            The `table_name` parameter is a string that represents the name of the table in the database
        that you want to query.
        columns : Optional[List[str]]
            The `columns` parameter is a list of column names that you want to select from the table. If
        this parameter is provided, the SQL query will include only these columns in the SELECT
        statement.
        limit : Optional[int]
            The `limit` parameter is used to specify the maximum number of rows to be returned in the SQL
        query result. It restricts the number of rows returned by the query.
        offset : Optional[int]
            The `offset` parameter is used to specify the number of rows to skip before starting to return
        rows from the query result.

        Returns
        -------
            a SQL query string.

        """
        if columns is not None:
            columns_string = ",".join(columns)
            query = f"SELECT {columns_string} FROM {schema_name}.{table_name}"
        else:
            query = f"SELECT * FROM {schema_name}.{table_name}"

        if limit is not None:
            query += f" LIMIT {limit}"

        if offset is not None:
            query += f" OFFSET {offset}"

        return query

    def read_sql_table(
        self,
        table_name: str,
        schema_name: str = "dbo",
        columns: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Toplevel function that prepares a query and reads data columns in specified table(s) for machine learning."""
        # TODO: JOIN operation will be needed here if annotations and report text are stored in separate tables.
        query = self._build_sql_query_chunk(
            table_name, schema_name=schema_name, columns=columns
        )
        return self.read_sql_query(query)

    def read_table_in_chunks(
        self, table_name: str, chunk_size: int, chunk_idx: int, schema_name: str = "dbo"
    ) -> Optional[pd.DataFrame]:
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
        query = self._build_sql_query_chunk(
            table_name, schema_name=schema_name, limit=limit, offset=offset
        )
        return self.read_sql_query(query)

    def get_table_size(self, table_name: str, schema_name: str = "dbo") -> int:
        """Returns the number of rows of the specified table."""
        query = f"SELECT COUNT(*) AS number_of_rows FROM {schema_name}.{table_name}"
        df = self.read_sql_query(query)
        if df is None:
            return 0
        return df["number_of_rows"].values[0]

    def upsert_values(
        self,
        table_metadata: object,
        data_to_insert: dict,
        cols_to_upsert: list,
        unique_cols: str = ["id"],
    ) -> None:
        """The `upsert_values` function performs an upsert operation on a database table using SQLAlchemy,
        where it inserts new data or updates existing data based on specified columns.

        Parameters
        ----------
        table_metadata : object
            The `table_metadata` parameter is an object that represents the metadata of the table where the
        data will be inserted or updated.
        data_to_insert : dict
            The `data_to_insert` parameter is a dictionary that contains the data to be inserted or updated
        in the table. The keys of the dictionary represent the column names, and the values represent
        the corresponding values to be inserted or updated.
        cols_to_upsert : list
            The `cols_to_upsert` parameter is a list of column names that you want to update if a conflict
        occurs during the upsert operation. These columns will be updated with the values from the
        `data_to_insert` dictionary.
        unique_cols : str
            The `unique_cols` parameter is a list of column names that are used to determine uniqueness in
        the table. These columns are used to identify if a row already exists in the table or not.

        """

        set_values: dict = {}

        stmt = insert(table_metadata).values(data_to_insert)

        for col in cols_to_upsert:
            set_values[col] = getattr(stmt.excluded, col)

        # Specify the conflict resolution
        stmt = stmt.on_conflict_do_update(index_elements=unique_cols, set_=set_values)

        # Execute the upsert statement
        self.session.execute(stmt)

        # Commit the transaction
        self.session.commit()

    @staticmethod
    def get_sql_tuple(column_items: List[str]) -> str:
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

    def run_dbt_model(self, model: str) -> None:
        '''The `run_dbt_model` function changes the current working directory to the specified DBT project
        path and then executes a bash command to run a specific DBT model.
        
        Parameters
        ----------
        model : str
            The `model` parameter is a string that represents the specific model you want to run in your
        dbt project. It is used as an argument in the `run.sh` script to specify which model to execute.
        
        '''
        dbt_project_path = os.getenv("DBT_PROJECT_PATH")

        os.chdir(dbt_project_path)
        os.system(f"bash run.sh {model}")
