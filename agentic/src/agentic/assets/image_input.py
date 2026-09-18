from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageInputReport:
    path: str
    width: int
    height: int
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_image_input(image_path: str | Path) -> ImageInputReport:
    """Check decodability and the H3 conditioning minimum dimensions."""
    path = Path(image_path).expanduser().resolve()
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        return ImageInputReport(str(path), 0, 0, False, (f"cannot decode image: {exc}",))
    reasons = ("keyframe must be at least 256x192",) if width < 256 or height < 192 else ()
    return ImageInputReport(str(path), width, height, not reasons, reasons)


def assert_image_input(image_path: str | Path) -> ImageInputReport:
    report = inspect_image_input(image_path)
    if not report.passed:
        raise ValueError(f"Image input rejected {report.path}: {'; '.join(report.reasons)}")
    return report
