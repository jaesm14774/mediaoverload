from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

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
    multi_panel_detected: bool = False
    multi_panel_allowed: bool = False
    duplicate_protagonist_detected: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "pink_ratio": round(self.pink_ratio, 4),
            "red_ratio": round(self.red_ratio, 4),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "multi_panel_detected": self.multi_panel_detected,
            "multi_panel_allowed": self.multi_panel_allowed,
            "duplicate_protagonist_detected": self.duplicate_protagonist_detected,
        }


def _detect_multi_panel_layout(image: Image.Image) -> bool:
    """Detect a strong 2x2-style source layout before it reaches I2V."""

    sample = image.convert("RGB").resize((256, 256))
    pixels = sample.load()

    vertical: list[float] = []
    for x in range(255):
        total = 0
        for y in range(256):
            left = pixels[x, y]
            right = pixels[x + 1, y]
            total += sum(abs(int(left[channel]) - int(right[channel])) for channel in range(3))
        vertical.append(total / (256 * 3))

    horizontal: list[float] = []
    for y in range(255):
        total = 0
        for x in range(256):
            top = pixels[x, y]
            bottom = pixels[x, y + 1]
            total += sum(abs(int(top[channel]) - int(bottom[channel])) for channel in range(3))
        horizontal.append(total / (256 * 3))

    def seam_metrics(values: list[float]) -> tuple[float, float]:
        center = len(values) // 2
        window = values[max(0, center - 32) : min(len(values), center + 32)]
        edge = values[:32] + values[-32:]
        baseline = max(1.0, float(median(edge)))
        return max(window), max(window) / baseline

    vertical_peak, vertical_ratio = seam_metrics(vertical)
    horizontal_peak, horizontal_ratio = seam_metrics(horizontal)

    quadrant_means: list[tuple[float, float, float]] = []
    for y0 in (0, 128):
        for x0 in (0, 128):
            total = [0, 0, 0]
            for y in range(y0, y0 + 128):
                for x in range(x0, x0 + 128):
                    pixel = pixels[x, y]
                    for channel in range(3):
                        total[channel] += int(pixel[channel])
            count = 128 * 128
            quadrant_means.append(tuple(value / count for value in total))
    quadrant_distance = sum(
        sum(abs(quadrant_means[left][channel] - quadrant_means[right][channel]) for channel in range(3)) / 3
        for left in range(4)
        for right in range(left)
    ) / 6

    return (
        vertical_peak >= 20.0
        and vertical_ratio >= 3.0
        and horizontal_peak >= 30.0
        and horizontal_ratio >= 4.0
        and quadrant_distance >= 18.0
    )


def _detect_duplicate_kirby_silhouettes(image: Image.Image) -> bool:
    """Detect two separated large pink body regions in one identity frame.

    This intentionally targets only large connected body regions. It does not
    count red shoes or small pink highlights, and it leaves normal wide shots
    with one compact protagonist alone.
    """

    sample = image.convert("RGB").resize((64, 64))
    pixels = sample.load()
    mask: set[tuple[int, int]] = set()
    for y in range(64):
        for x in range(64):
            red, green, blue = pixels[x, y]
            if red >= 120 and red > green * 1.18 and blue >= green * 0.85:
                mask.add((x, y))
    components: list[tuple[int, bool]] = []
    while mask:
        start = mask.pop()
        pending = [start]
        size = 1
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        while pending:
            x, y = pending.pop()
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in mask:
                    mask.remove(neighbor)
                    pending.append(neighbor)
                    size += 1
        touches_border = min_x == 0 or min_y == 0 or max_x == 63 or max_y == 63
        components.append((size, touches_border))
    # A pink sky, gradient, or other background wash can form a large color
    # component that touches the frame boundary. It is not a second
    # protagonist, so do not count boundary-connected regions as silhouettes.
    interior_components = [size for size, touches_border in components if not touches_border]
    return sum(size >= 320 for size in interior_components) >= 2


def inspect_kirby_input(
    image_path: str | Path,
    *,
    allow_external: bool = False,
    allow_multipanel: bool = False,
) -> KirbyInputReport:
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
            multi_panel_detected = _detect_multi_panel_layout(rgb)
            duplicate_protagonist_detected = _detect_duplicate_kirby_silhouettes(rgb)
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
    # A valid generated frame can place Kirby very small in a wide shot.  The
    # full-frame pink area is then far below 1%, but the combination of a
    # pink body and red feet is still useful identity evidence.  Keep the
    # normal threshold for ordinary frames and allow this compact signal only
    # when both colors are present.
    # Wide shots and back-facing silhouettes can have muted/dark pink body
    # pixels and only a small amount of red footwear. Keep a compact identity
    # signal, but do not require the full-frame thresholds used for closeups.
    has_compact_kirby_signal = pink >= 0.0025 and red >= 0.001
    if pink < 0.01 and not has_compact_kirby_signal:
        reasons.append("no meaningful Kirby-pink color signal detected")
    if red < 0.001:
        reasons.append("no red-foot color signal detected")
    if multi_panel_detected and not allow_multipanel:
        reasons.append("multi-panel/collage keyframe is blocked outside the ref2va reference-video route")
    if duplicate_protagonist_detected and not allow_multipanel:
        reasons.append("duplicate Kirby protagonist silhouettes are blocked in a single-character keyframe")
    return KirbyInputReport(
        str(path),
        width,
        height,
        pink,
        red,
        not reasons,
        tuple(reasons),
        multi_panel_detected,
        allow_multipanel,
        duplicate_protagonist_detected,
    )


def assert_kirby_input(
    image_path: str | Path,
    *,
    allow_external: bool = False,
    allow_multipanel: bool = False,
) -> KirbyInputReport:
    report = inspect_kirby_input(
        image_path,
        allow_external=allow_external,
        allow_multipanel=allow_multipanel,
    )
    if not report.passed:
        details = "; ".join(report.reasons) or "unknown Kirby input validation failure"
        raise ValueError(f"Kirby input gate rejected {report.path}: {details}")
    return report
