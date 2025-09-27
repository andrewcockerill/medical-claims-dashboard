# Packages
import pandas as pd
import json
import requests
import io
import zipfile
import sqlite3
import os
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import joblib

# Manager
class IngestManager:
    """Manager object to handle ingestion of raw data files based on settings listed in a json configuration
    file.
    
    Params
    ------
    config : str
        The path to the configuration json file
    """
    
    def __init__(self, config : str):
        with open(config, 'r') as f:
            self.config = json.load(f)
        self.random_seed = self.config["random_seed"]

    def _df_read_format(self, path, columns, format, options):
        # Get column names
        names = [column["name"] for column in columns]

        # Read in selected fields as strings
        if format["type"] == "fwf":
            colspecs = format["colspecs"]
            df = pd.read_fwf(filepath_or_buffer=path, colspecs=colspecs, names=names, dtype=str, encoding='latin-1')
        elif format["type"] == "csv":
            delimiter = format["delimiter"]
            df = pd.read_csv(filepath_or_buffer=path, delimiter=delimiter, usecols=names, dtype=str)
        else:
            raise AssertionError("Input file types must be fixed-width or delimited text files")
        
        # Sampling if needed
        if options["sample_size"]:
            df = df.sample(n=options["sample_size"], random_state=self.config["random_seed"])
        
        # Apply dtype formatting based on config
        non_date_columns = [column for column in columns if column["dtype"]!="date"]
        if non_date_columns:
            dtype_map = {column["name"]:column["dtype"] for column in non_date_columns}
            df = df.astype(dtype_map)

        date_columns = [column for column in columns if column["dtype"]=="date"]
        if date_columns:
            for column in date_columns:
                name = column["name"]
                df[name] = pd.to_datetime(df[name], format=column["format"]).dt.date

        # Standardize column names to all caps
        df.columns = [name.upper() for name in df.columns]

        return df

    def _ingest_source(self, source):
        # Get metadata from config file
        uri = source['uri']
        filename = source['filename']
        is_archive = source['is_archive']
        columns = source["columns"]
        format = source["format"]
        options = source["options"]

        # Unzip if file is in an archive, otherwise open from uri
        if is_archive:
            with requests.get(uri) as result:
                result.raise_for_status()
                zip_bytes = io.BytesIO(result.content)
                with zipfile.ZipFile(zip_bytes) as archive:
                    with archive.open(filename) as f:
                        df = self._df_read_format(f, columns, format, options)
        else:
            df = self._df_read_format(uri, columns, format, options)

        return df
    
    def run_ingestion(self):
        """Runs ingestion of raw data from URLs in ingest_config.json"""
        data_sources = self.config["raw_sources"]
        database = os.path.join("..", "..", "db", self.config["database"])
        with sqlite3.connect(database) as conn:
            for source in data_sources:
                table_name = source["table_name"]
                df = self._ingest_source(source)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Loaded {table_name}.")


# Database read/write manager
class DbManager:
    """Simple class to manage connections to query and write to the sqlite database.
    
    Params
    ------
    config : str
        The path to the configuration json file
    
    """
    def __init__(self, config :str):
        with open(config, 'r') as f:
            self.config = json.load(f)
        self.database = os.path.join("..", "..", "db", self.config["database"])

    def execute(self, query : str):
        # Execute sql against the database
        with sqlite3.connect(self.database) as conn:
            statements = query.split(";")
            for statement in statements:
                conn.execute(statement)

    def query(self, query : str):
        # Run a query against the database and return a dataframe
        with sqlite3.connect(self.database) as conn:
            df = pd.read_sql(query, conn)
        return df
    
    def write_df(self, df : pd.DataFrame, table_name : str):
        # Write a dataframe to the database
        with sqlite3.connect(self.database) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

# Model class
class DiagnosisSegmentModel:
    """Model object to train an embedding model to encode ICD9 codes. After this, a KMeans Clustering model
    is trained on the averaged value of the first 4 diagnosis codes in a given claim. Results are tabulated
    to provide the top diagnoses in each cluster, along with monthly payment totals grouped by cluster.
    
    Params
    ------
    model_config : str
        The path to the MODEL configuration json file

    db_config : str
        The path to the DB configuration json file
    """
    def __init__(self, model_config : str, db_config : str):
        with open(model_config, 'r') as f:
            self.model_config = json.load(f)

        with open(db_config, 'r') as f:
            self.db_config = json.load(f)

        self.dbm = DbManager(config=db_config)
        self.icd_columns = self.model_config["icd_columns"]

    def _concat_list(self, cols):
        return list(set([i for i in cols.tolist() if i is not None]))

    def _compute_average_embedding(self, cols):
        code_list = self._concat_list(cols)
        average_vector = self.embedding_model.wv.get_mean_vector(code_list)
        return average_vector
    
    def _get_dev_data(self):
        query_path = os.path.join("..","sql",self.model_config["model_dev_query"])
        with open(query_path, 'r') as f:
            claims_table_name = self.db_config["raw_sources"][1]["table_name"]
            query = f.read().format(claims_table_name=claims_table_name)
            df = self.dbm.query(query)
        return df

    def _build_embedding_model(self, df):
        vector_size = self.model_config["vector_size"]
        window = self.model_config["window"]
        min_count = self.model_config["min_count"]
        
        corpus = df[self.icd_columns].apply(self._concat_list, axis=1).tolist()
        self.embedding_model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count)
        
    def _build_and_score_cluster_model(self, df):
        n_clusters = self.model_config["n_clusters"]
        cluster_column = self.model_config["cluster_column"]
        clusters_table_name = self.model_config["clusters_table_name"]
        dev_array = np.array(df[self.icd_columns].apply(self._compute_average_embedding, axis=1).tolist())
        self.cluster_model = KMeans(n_clusters=n_clusters).fit(dev_array)
        out_df = df.copy()
        out_df[cluster_column] = self.cluster_model.predict(dev_array).astype(int)
        self.dbm.write_df(out_df, clusters_table_name)

    def _tabulate_clusters(self):
        # Get references
        descriptions_table_name = self.db_config["raw_sources"][0]["table_name"]
        claims_table_name = self.db_config["raw_sources"][1]["table_name"]
        clusters_table_name = self.model_config["clusters_table_name"]
        cluster_desc_table_name = self.model_config["cluster_desc_table_name"]
        cluster_pmt_totals_table_name = self.model_config["cluster_pmt_totals_table_name"]
        cluster_desc_query_path = os.path.join("..","sql",self.model_config["cluster_desc_query"])
        cluster_pmt_totals_query_path = os.path.join("..","sql",self.model_config["cluster_pmt_totals_query"])

        # Cluster descriptions and code frequencies
        with open(cluster_desc_query_path, 'r') as f:
            cluster_desc_query = f.read().format(descriptions_table_name=descriptions_table_name, cluster_desc_table_name=cluster_desc_table_name)
            self.dbm.execute(cluster_desc_query)

        # Monthly claim payment amounts by cluster
        with open(cluster_pmt_totals_query_path, 'r') as f:
            cluster_pmt_totals_query = f.read().format(cluster_pmt_totals_table_name=cluster_pmt_totals_table_name, 
                                                       claims_table_name=claims_table_name, clusters_table_name=clusters_table_name)
            self.dbm.execute(cluster_pmt_totals_query)

    def _save(self):
        save_path = os.path.join("..","..","models",self.model_config["model_file"])
        joblib.dump(self, save_path)

    def build_model(self):
        """Build the embedding and cluster models, post SQL aggregations for dashboard and save the model
        artifact."""
        # Fetch data
        df = self._get_dev_data()

        # Build models
        self._build_embedding_model(df)
        print('Built embedding model.')
        self._build_and_score_cluster_model(df)
        print('Built cluster model.')

        # Run tabulations and save
        self._tabulate_clusters()
        print('Wrote results to DB.')
        self._save()
        print('Saved model.')