import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentChunk:
    """Data class for document chunks"""

    content: str
    source: str
    chunk_id: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
