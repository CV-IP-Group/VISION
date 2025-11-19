"""Vision module package providing typed stubs for computer vision tasks."""

from .engine import CVModuleProcess
from .types import Box, VisionResult

__all__ = ["CVModuleProcess", "Box", "VisionResult"]
