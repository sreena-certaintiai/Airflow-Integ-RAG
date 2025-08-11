from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="pdf_processing_pipeline",
    start_date=pendulum.datetime(2025, 8, 11, tz="Asia/Kolkata"),
    schedule=None,  # This means the DAG will only run when we trigger it manually
    catchup=False,
    doc_md="""
    ### PDF to Vector Pipeline
    Orchestrates the two main Python scripts for our RAG pipeline.
    This DAG will:
    1. Run the script to chunk PDFs.
    2. Run the script to store the chunks and their embeddings in ChromaDB.
    """,
    tags=["rag", "etl"],
) as dag:

    # IMPORTANT: When running in Docker via Astro CLI, your project folder
    # is located at '/usr/local/airflow'. So we provide the full path to our scripts.

    chunk_pdfs_task = BashOperator(
        task_id="chunk_pdfs_from_data_folder",
        bash_command="python /usr/local/airflow/Core/chunks-to-cdb.py",
    )

    store_vectors_task = BashOperator(
        task_id="store_vectors_in_chromadb",
        bash_command="python /usr/local/airflow/Core/cdb_manager.py",
    )

    # This line defines the order: task 1 must succeed before task 2 can run
    chunk_pdfs_task >> store_vectors_task