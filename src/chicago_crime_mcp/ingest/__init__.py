"""Ingestion: paginated, checkpointed backfill from the Socrata (SODA) API plus
an incremental daily sync keyed on `updated_on`. Lands Parquet partitioned by
year. Airflow is the productionization target; see scripts/download_data.py for
the exploration-stage puller.
"""
