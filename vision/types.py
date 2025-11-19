"""Typed data structures for the VisionMate computer vision module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Box:
    """Light-weight detection box representation."""

    xyxy: Tuple[int, int, int, int]
    cls: str
    score: float
    dist: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the detection box into a JSON-friendly dictionary."""
        return {
            "xyxy": self.xyxy,
            "cls": self.cls,
            "score": self.score,
            "dist": self.dist,
        }


@dataclass
class VisionResult:
    """Container for results returned by CVModuleProcess routines."""

    task: str
    image_path: str
    objects: List[Box] = field(default_factory=list)
    caption: Optional[str] = None
    message: Optional[str] = None
    ocr_text: Optional[str] = None
    depth_path: Optional[str] = None
    overlay_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the vision result, skipping empty optional fields."""
        data: Dict[str, Any] = {
            "task": self.task,
            "image_path": self.image_path,
            "objects": [box.to_dict() for box in self.objects],
            "meta": self.meta,
        }
        if self.caption is not None:
            data["caption"] = self.caption
        if self.message is not None:
            data["message"] = self.message
        if self.ocr_text is not None:
            data["ocr_text"] = self.ocr_text
        if self.depth_path is not None:
            data["depth_path"] = self.depth_path
        if self.overlay_path is not None:
            data["overlay_path"] = self.overlay_path
        return data
