from pathlib import Path
from src.ingest.pipeline import IngestionPipeline

pipeline = IngestionPipeline(index_storage_dir=Path("data/index"))
pipeline.ingest_directory(Path("data/raw"))
