"""UFC fight database + Elo baseline pipeline.

Stage 1 only: data ingestion (source adapters -> normalized fights -> SQLite)
and a time-safe Elo baseline. No ML model lives here yet.
"""

__version__ = "0.1.0"
