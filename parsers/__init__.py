# parsers/__init__.py
"""
File parsers: documents (PDF/DOCX/TXT) and the hybrid CSV/Excel processor.
"""

from .document_parser import DocumentParser

__all__ = ['DocumentParser']