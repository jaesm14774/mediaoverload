from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class KirbyInputReport:
    path: str
    width: int
    height: int
    pink_ratio: float
    red_ratio: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "pink_ratio": round(self.pink_ratio, 4),
            "red_ratio": round(self.red_ratio, 4),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def inspect_kirby_input(image_path: str | Path, *, allow_external: bool = False) -> KirbyInputReport:
    path = Path(image_path).expanduser().resolve()
    reasons: list[str] = []
    if not path.exists():
        return KirbyInputReport(str(path), 0, 0, 0.0, 0.0, False, ("image does not exist",))
    if path.name.lower() in {"example.png", "example.jpg", "example.jpeg"} and not allow_external:
        reasons.append("generic ComfyUI example image is blocked for Kirby")

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            sample = rgb.resize((64, 64))
            pixels = list(sample.getdata())
    except Exception as exc:
        return KirbyInputReport(str(path), 0, 0, 0.0, 0.0, False, (f"cannot decode image: {exc}",))

    pink = sum(
        1
        for red, green, blue in pixels
        if red >= 120 and red > green * 1.18 and blue >= green * 0.85
    ) / len(pixels)
    red = sum(
        1
        for red_value, green, blue in pixels
        if red_value >= 120 and red_value > green * 1.35 and red_value > blue * 1.25
    ) / len(pixels)
    if width < 256 or height < 192:
        reasons.append("keyframe is too small for reliable H3 conditioning")
    if pink < 0.01:
        reasons.append("no meaningful Kirby-pink color signal detected")
    if red < 0.002:
        reasons.append("no red-foot color signal detected")
    return KirbyInputReport(str(path), width, height, pink, red, not reasons, tuple(reasons))


def assert_kirby_input(image_path: str | Path, *, allow_external: bool = False) -> KirbyInputReport:
    report = inspect_kirby_input(image_path, allow_external=allow_external)
    if not report.passed:
        details = "; ".join(report.reasons) or "unknown Kirby input validation failure"
        raise ValueError(f"Kirby input gate rejected {report.path}: {details}")
    return report
