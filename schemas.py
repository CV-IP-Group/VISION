"""Pydantic models defining API contracts for VisionMate."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from vision.types import Box, VisionResult


class BoxModel(BaseModel):
    """Serializable representation of a detection box."""

    xyxy: Tuple[int, int, int, int] = Field(description="Bounding box in xyxy format.")
    cls: str = Field(description="Detected class label.")
    score: float = Field(description="Confidence score of the detection.")
    dist: Optional[float] = Field(default=None, description="Optional distance estimate in meters.")

    @classmethod
    def from_domain(cls, box: "Box") -> "BoxModel":
        return cls(xyxy=box.xyxy, cls=box.cls, score=box.score, dist=box.dist)


class VisionResultModel(BaseModel):
    """Serializable form of the CV module output."""

    task: str
    image_path: str
    objects: List[BoxModel] = Field(default_factory=list)
    caption: Optional[str] = None
    message: Optional[str] = None
    ocr_text: Optional[str] = None
    depth_path: Optional[str] = None
    overlay_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, result: "VisionResult") -> "VisionResultModel":
        return cls(
            task=result.task,
            image_path=result.image_path,
            objects=[BoxModel.from_domain(box) for box in result.objects],
            caption=result.caption,
            message=result.message,
            ocr_text=result.ocr_text,
            depth_path=result.depth_path,
            overlay_path=result.overlay_path,
            meta=result.meta or {},
        )


class MCPRequest(BaseModel):
    """Request schema for the MCP API."""

    api_version: str = Field(default="v1", description="API version of the request.")
    text: str = Field(description="User query text.")
    camera_index: Optional[int] = Field(default=None, description="Preferred camera index.")
    tts: bool = Field(default=False, description="Whether to synthesize TTS audio for the response message.")


class MCPResponse(BaseModel):
    """Standardized response schema for the MCP API."""

    api_version: str = Field(default="v1", description="API version of the response.")
    status: Literal["ok", "warn", "error"]
    message: Optional[str] = None
    data: Optional[VisionResultModel] = None
    intent: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
