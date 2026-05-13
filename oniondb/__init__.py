"""
╔══════════════════════════════════════════════════════════════╗
║  OnionDB — Multi-Shell Geometric Database                    ║
║                                                              ║
║  SYSTEM:   oniondb (standalone project)                      ║
║  PURPOSE:  Hierarchical database with balloon-inside-balloon ║
║            geometry, 4-part addressing, and four native       ║
║            query operations (horizontal, ray, shell, range)  ║
║  CREATED:  May 3-4, 2026 by Nick + Venus IDE                 ║
╚══════════════════════════════════════════════════════════════╝
"""
from .onion_db import OnionDB

__version__ = "0.6.1"
__all__ = ["OnionDB"]
