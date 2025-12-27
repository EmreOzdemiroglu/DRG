"""
DRG API Module - FastAPI Server

Provides REST API endpoints for:
- Graph visualization
- Community reports
- Query provenance
- Knowledge graph exploration
"""

from .server import create_app, DRGAPIServer

__all__ = ["create_app", "DRGAPIServer"]

