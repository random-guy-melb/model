
from lancedb.rerankers import CrossEncoderReranker
from lancedb.rerankers import RRFReranker

import pandas as pd
import lancedb
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.utils.preprocessing import DataProcessing


class RAG(DataProcessing):
    def __init__(self,
                 db_path: str,
                 project: str,
                 embedding_model,
                 fts_column: str = "content",
                 schema: dict = None,
                 reranker_path: str = None):
        """
        Initialize RAG with a DB connection, project name, embedding model,
        FTS index column (to be created once), and an optional schema mapping (for renaming input fields).
        """
        self.db = lancedb.connect(db_path)
        self.project = project
        self.table_name = f"{project}_vectors"
        self.vector_dim = 1536
        self.embedding_model = embedding_model
        self.fts_column = fts_column
        self.schema = schema
        if reranker_path:
            self.reranker = CrossEncoderReranker(
                model_name=reranker_path, column="content", return_score="relevance"
            )
        else:
            self.reranker = None

        # Open existing table if it exists; otherwise set table to None.
        self.table = self.db.open_table(self.table_name) if self.table_name in self.db.table_names() else None

    def get_min_max_ts(self) -> tuple:
        # ts = self.table.to_pandas()["timestamp"]
        ts = self.table.to_arrow()["timestamp"].combine_chunks().to_numpy()
        return ts.min(), ts.max()

    def add_documents(self, records: list, id_column: str = 'id') -> None:
        """
        Add a list of record dictionaries to DB.
        Assumes each record has keys matching the schema.
        """
        df = pd.DataFrame(records)

        # Validate content field exists upfront
        if "content" not in df.columns:
            raise ValueError("Input records must include a 'content' field.")

        # If table doesn't exist, create it and run FTS index creation once.
        if self.table is None:
            # Generate embeddings for all records when creating new table
            df["vector"] = self.embedding_model.generate_embeddings(df["content"].tolist())

            self.table = self.db.create_table(
                self.table_name,
                data=df,
                mode="create",
                schema=self.schema
            )
            # Create the FTS index after table creation is complete
            self.table.create_fts_index(self.fts_column)
        else:
            # Check which IDs already exist
            ids_to_check = df[id_column].tolist()
            if ids_to_check and isinstance(ids_to_check[0], str):
                ids_str = ', '.join([f"'{id}'" for id in ids_to_check])
            else:
                ids_str = ', '.join([str(id) for id in ids_to_check])

            existing_df = self.table.search().where(f"{id_column} IN ({ids_str})").select([id_column]).to_pandas()
            existing_ids = set(existing_df[id_column].tolist()) if len(existing_df) > 0 else set()

            # Filter out existing records
            df = df[~df[id_column].isin(existing_ids)]

            if len(df) > 0:
                # Generate embeddings only for new records
                df["vector"] = self.embedding_model.generate_embeddings(df["content"].tolist())
                self.table.add(df)

    def search(self,
               query: str = None,
               clause: str = None,
               search_type: str = 'hybrid',
               query_category: str = None,
               threshold: float = .12,
               top_k: int = 5) -> pd.DataFrame:
        """
        Perform search using DB's built-in hybrid (or other) search API.
        If a SQL-like where clause is provided, it is applied.
        Returns a DataFrame of results.
        """
        if not query and not clause:
            raise "Either query or clause needs to be provided!"

        table = self.table.search(self._clean_query(query), query_type=search_type) if query \
            else self.table.search()

        if clause:
            table = table.where(clause)

        if search_type == "hybrid" and query_category == "defect queries":
            table = table.rerank(
                reranker=self.reranker if self.reranker else RRFReranker(return_score="all")
            )

        results = table.limit(top_k).to_pandas()
        if "_relevance_score" in results.columns:
            results = results[results["_relevance_score"] >= threshold]
        return results
