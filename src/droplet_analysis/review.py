"""Compatibility entry points for the standalone tool's review renderer."""
from watercar_tool.review import write_review, write_combined_review

__all__ = ["write_review", "write_combined_review"]
