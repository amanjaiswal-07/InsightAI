# utils/uploads.py
"""
Validation helpers for file uploads, kept free of heavy imports so they can
be unit-tested without loading the agents or ML models.
"""

import re
from pathlib import Path
from typing import Optional

# The frontend only offers these types. .xls is excluded because reading it
# needs the extra xlrd package.
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".pdf", ".docx", ".txt"}

# Conversation ids are server-generated UUIDs. Anything outside this alphabet
# could escape the uploads folder when used as a directory name.
_CID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_conversation_id(cid: Optional[str]) -> bool:
    """True if cid is safe to use as a single path component."""
    return bool(cid) and bool(_CID_RE.match(cid))


def safe_upload_name(filename: Optional[str]) -> Optional[str]:
    """
    Return the bare file name if it has an allowed extension, else None.

    Any directory part sent by the client (including "../" tricks and
    Windows-style back-slashes) is discarded.
    """
    name = Path((filename or "").replace("\\", "/")).name
    if not name or name in (".", ".."):
        return None
    if Path(name).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        return None
    return name
