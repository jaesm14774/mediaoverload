"""Build native ComfyUI canvas workflows from the repo's API workflows.

The repo keeps API prompt graphs for Agentic execution. ComfyUI's Open dialog
expects the canvas serialization instead. This script composes the existing
Anima/keyframe, img2img, and MiniMax H3 graphs and emits both formats' bridge
artifacts without changing the source API workflows.
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "configs" / "workflow"
CANVAS_DIR = API_DIR / "comfyui"
PORTABLE_CANVAS_DIR = Path(r"D:\ComfyUI_windows_portable\ComfyUI\user\default\workflows")
OBJECT_INFO_URL = "http://127.0.0.1:8188/object_info"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_object_info() -> dict[str, Any]:
    try:
        with urlopen(OBJECT_INFO_URL, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - useful offline fallback
        print(f"warning: could not read {OBJECT_INFO_URL}: {exc}", file=sys.stderr)
        return {}


def is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, (str, int)) for x in value)


def as_int(value: str | int) -> int:
    return int(value)


def combine_graphs(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    remove_second: set[str] | None = None,
    bindings: dict[tuple[str, str], list[str | int]] | None = None,
) -> dict[str, Any]:
    """Merge API graphs and optionally bind a second graph input to first output."""

    remove_second = remove_second or set()
    bindings = bindings or {}
    first_ids = {str(key): str(key) for key in first}
    max_first = max((as_int(key.split(":")[0]) for key in first), default=0)
    second_ids = {
        str(key): str(max_first + index + 1)
        for index, key in enumerate(second)
        if str(key) not in remove_second
    }

    merged: dict[str, Any] = {}
    for key, node in first.items():
        merged[first_ids[str(key)]] = copy.deepcopy(node)
    for key, node in second.items():
        key = str(key)
        if key in remove_second:
            continue
        rewritten = copy.deepcopy(node)
        for input_name, value in list(rewritten.get("inputs", {}).items()):
            binding = bindings.get((key, input_name))
            if binding is not None:
                rewritten["inputs"][input_name] = [str(binding[0]), int(binding[1])]
            elif is_link(value):
                source = second_ids.get(str(value[0]))
                if source is None:
                    raise ValueError(f"dangling second-graph link {key}.{input_name} -> {value}")
                rewritten["inputs"][input_name] = [source, int(value[1])]
        merged[second_ids[key]] = rewritten
    return merged


def input_order(info: dict[str, Any]) -> list[str]:
    order = info.get("input_order") or {}
    return list(order.get("required", [])) + list(order.get("optional", []))


def input_spec(info: dict[str, Any], name: str) -> list[Any] | None:
    for section in ("required", "optional"):
        value = ((info.get("input") or {}).get(section) or {}).get(name)
        if value is not None:
            return value
    return None


def widget_input(spec: list[Any] | None) -> bool:
    if not spec:
        return False
    value_type = spec[0]
    if isinstance(value_type, list):
        return True
    # ComfyUI also exposes dynamic widget types such as
    # COMFY_DYNAMICCOMBO_V3. The remaining named types are graph sockets.
    socket_types = {
        "MODEL", "CLIP", "VAE", "IMAGE", "MASK", "CONDITIONING", "LATENT",
        "AUDIO", "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS",
    }
    return value_type not in socket_types


def node_size(node_type: str, info: dict[str, Any]) -> list[float]:
    count = len(input_order(info))
    if node_type in {"KSampler", "KSamplerAdvanced"}:
        return [310.0, 330.0]
    if node_type in {"SpectrumApplyMiniMaxH3"}:
        return [360.0, 430.0]
    if node_type in {"MiniMaxH3ImageToVideo"}:
        return [360.0, 300.0]
    if node_type in {"SaveVideo", "CreateVideo", "VAEDecodeAudio"}:
        return [300.0, 170.0]
    return [310.0, max(90.0, 62.0 + count * 28.0)]


def node_position(node_type: str, node_id: str, index: int, *, continuation: bool, h3_start_id: int) -> list[float]:
    # The merged graph has two VAELoader branches. Keep the repo image VAE on
    # the left and the H3 video/audio VAEs beside the H3 loader branch.
    if node_type == "VAELoader":
        if int(node_id) < h3_start_id:
            return [380.0 if continuation else 0.0, 400.0]
        return [2150.0, float(430 + (int(node_id) - h3_start_id) * 145)]
    if continuation:
        columns = {
            "LoadImage": (0, 0),
            "UNETLoader": (380, 0),
            "CLIPLoader": (380, 200),
            "PrimitiveString": (760, 0),
            "CLIPTextEncode": (1120, 0),
            "VAEEncode": (1120, 430),
            "KSampler": (1500, 200),
            "VAEDecode": (1860, 300),
        }
    else:
        columns = {
            "UNETLoader": (0, 0),
            "CLIPLoader": (0, 210),
            "VAELoader": (0, 430),
            "PrimitiveString": (360, 0),
            "CLIPTextEncode": (720, 0),
            "EmptyLatentImage": (720, 500),
            "KSampler": (1080, 230),
            "VAEDecode": (1450, 330),
            "SaveImage": (1800, 550),
        }
    if node_type in columns:
        x, y = columns[node_type]
        return [float(x), float(y + index * 18)]
    # H3 branch is kept to the right of the repo-derived image branch.
    h3_columns = {
        "UnetLoaderGGUF": (2150, 0),
        "CLIPLoaderGGUF": (2150, 210),
        "MiniMaxH3ImageToVideo": (2550, 180),
        "MiniMaxH3SigmaShift": (2550, 570),
        "SpectrumApplyMiniMaxH3": (2940, 570),
        "BasicScheduler": (3330, 570),
        "KSamplerSelect": (3330, 930),
        "RandomNoise": (3330, 1060),
        "BasicGuider": (3330, 0),
        "SamplerCustomAdvanced": (3720, 350),
        "VAEDecodeAudio": (4110, 610),
        "CreateVideo": (4490, 250),
        "SaveVideo": (4860, 270),
    }
    x, y = h3_columns.get(node_type, (5200, float(index * 130)))
    return [float(x), float(y)]


def build_links(api_nodes: dict[str, Any], infos: dict[str, Any]) -> tuple[list[list[Any]], dict[tuple[str, int], list[int]]]:
    links: list[list[Any]] = []
    output_links: dict[tuple[str, int], list[int]] = {}
    next_link = 1
    for target_id, node in api_nodes.items():
        info = infos.get(node["class_type"], {})
        slot_by_name = {name: index for index, name in enumerate(input_order(info))}
        for input_name, value in node.get("inputs", {}).items():
            if not is_link(value):
                continue
            source_id, source_slot = str(value[0]), int(value[1])
            target_slot = slot_by_name.get(input_name)
            if target_slot is None:
                raise ValueError(f"{node['class_type']} has no visible input slot {input_name}")
            output_types = infos.get(api_nodes[source_id]["class_type"], {}).get("output", [])
            link_type = output_types[source_slot] if source_slot < len(output_types) else "*"
            links.append([next_link, int(source_id), source_slot, int(target_id), target_slot, link_type])
            output_links.setdefault((source_id, source_slot), []).append(next_link)
            next_link += 1
    return links, output_links


def to_canvas(api_nodes: dict[str, Any], infos: dict[str, Any], *, name: str, continuation: bool) -> dict[str, Any]:
    links, output_links = build_links(api_nodes, infos)
    h3_start_id = min(
        (int(node_id) for node_id, node in api_nodes.items() if node["class_type"] == "UnetLoaderGGUF"),
        default=10**9,
    )
    canvas_nodes: list[dict[str, Any]] = []
    for index, (node_id, api_node) in enumerate(api_nodes.items()):
        node_type = api_node["class_type"]
        info = infos.get(node_type, {})
        order = input_order(info)
        widget_values: list[Any] = []
        inputs: list[dict[str, Any]] = []
        for input_name in order:
            spec = input_spec(info, input_name) or ["*"]
            value = api_node.get("inputs", {}).get(input_name)
            value_type = spec[0]
            if isinstance(value_type, list):
                socket_type = "COMBO"
            else:
                socket_type = value_type
            socket: dict[str, Any] = {
                "localized_name": input_name,
                "name": input_name,
                "type": socket_type,
            }
            if is_link(value):
                socket["link"] = None  # filled after link IDs are known below
            elif widget_input(spec) and value is not None:
                socket["widget"] = {"name": input_name}
                widget_values.append(value)
            else:
                socket["link"] = None
            inputs.append(socket)

        # Match ComfyUI's extra seed control widget in saved canvas files.
        if "seed" in order and "seed" in api_node.get("inputs", {}) and not is_link(api_node["inputs"]["seed"]):
            seed_index = order.index("seed")
            widget_before_seed = sum(
                1 for name_before in order[:seed_index]
                if name_before in api_node.get("inputs", {}) and not is_link(api_node["inputs"][name_before]) and widget_input(input_spec(info, name_before))
            )
            widget_values.insert(widget_before_seed + 1, "randomize")

        for input_index, input_name in enumerate(order):
            value = api_node.get("inputs", {}).get(input_name)
            if is_link(value):
                source_id, source_slot = str(value[0]), int(value[1])
                link_ids = output_links.get((source_id, source_slot), [])
                inputs[input_index]["link"] = link_ids[0] if link_ids else None

        output_names = info.get("output_name") or info.get("output") or []
        outputs: list[dict[str, Any]] = []
        for slot, output_name in enumerate(output_names):
            outputs.append({
                "localized_name": output_name,
                "name": output_name,
                "type": (info.get("output") or ["*"])[slot],
                "links": output_links.get((str(node_id), slot), []),
            })
        meta = api_node.get("_meta") or {}
        properties = {
            "Node name for S&R": node_type,
            "ver": "0.30.0",
        }
        if meta.get("title"):
            title = meta["title"]
        else:
            title = ""
        canvas_node: dict[str, Any] = {
            "id": int(node_id),
            "type": node_type,
            "pos": node_position(node_type, node_id, index, continuation=continuation, h3_start_id=h3_start_id),
            "size": node_size(node_type, info),
            "flags": {},
            "order": index,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": properties,
            "widgets_values": widget_values,
        }
        if title:
            canvas_node["title"] = title
        canvas_nodes.append(canvas_node)

    groups = (
        [
            {"title": "Repo source: Anima Kirby keyframe", "bounding": [-120, -120, 2050, 1200], "color": "#3f789e", "font_size": 24},
            {"title": "MiniMax H3 low-VRAM I2V", "bounding": [2050, -120, 3150, 1300], "color": "#9e6b3f", "font_size": 24},
        ]
        if not continuation
        else [
            {"title": "Repo source: img2img identity continuity", "bounding": [-120, -120, 2150, 1050], "color": "#3f789e", "font_size": 24},
            {"title": "MiniMax H3 continuation I2V", "bounding": [2050, -120, 3150, 1300], "color": "#9e6b3f", "font_size": 24},
        ]
    )
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"mediaoverload:{name}")),
        "revision": 0,
        "last_node_id": max((int(node_id) for node_id in api_nodes), default=0),
        "last_link_id": max((link[0] for link in links), default=0),
        "nodes": canvas_nodes,
        "links": links,
        "groups": groups,
        "config": {},
        "extra": {
            "workflowRendererVersion": "Vue-corrected",
            "mediaoverload": {
                "source": "repo configs/workflow API graphs",
                "purpose": name,
                "open_with": "ComfyUI > Workflow > Open",
            },
        },
        "version": 0.4,
    }


def main() -> None:
    infos = fetch_object_info()
    required = {
        "UNETLoader", "CLIPLoader", "VAELoader", "PrimitiveString", "PrimitiveInt", "CLIPTextEncode",
        "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage", "LoadImage", "VAEEncode",
        "UnetLoaderGGUF", "CLIPLoaderGGUF", "MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift",
        "SpectrumApplyMiniMaxH3", "BasicScheduler", "KSamplerSelect", "RandomNoise", "BasicGuider",
        "SamplerCustomAdvanced", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
    }
    missing = sorted(required - set(infos))
    if missing:
        raise RuntimeError(f"ComfyUI object_info is missing nodes; start ComfyUI and retry: {missing}")

    keyframe = load_json(API_DIR / "kirby_keyframe_anima.json")
    identity = load_json(API_DIR / "kirby_identity_img2img.json")
    h3 = load_json(API_DIR / "minimax_h3_lowvram_i2v.json")
    native15 = load_json(API_DIR / "minimax_h3_lowvram_15s_fl2va_i2v.json")

    keyframe_i2v = combine_graphs(
        keyframe,
        h3,
        remove_second={"16"},
        bindings={("5", "first_frame"): ["12", 0]},
    )
    continuation_i2v = combine_graphs(
        identity,
        h3,
        remove_second={"16"},
        bindings={("5", "first_frame"): ["11", 0]},
    )
    native15_i2v = combine_graphs(
        keyframe,
        native15,
        remove_second={"16"},
        bindings={("5", "first_frame"): ["12", 0]},
    )

    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (
            "MediaOverload_Kirby_H3_Keyframe_to_I2V.json",
            to_canvas(keyframe_i2v, infos, name="Kirby keyframe → MiniMax H3 I2V", continuation=False),
        ),
        (
            "MediaOverload_Kirby_H3_Continuation_Img2Img_to_I2V.json",
            to_canvas(continuation_i2v, infos, name="Kirby tail frame → img2img → MiniMax H3 I2V", continuation=True),
        ),
    ]
    outputs.append(
        (
            "MediaOverload_Kirby_H3_Native15_FirstLast.json",
            to_canvas(native15_i2v, infos, name="Kirby native 15s first + last frame MiniMax H3", continuation=False),
        )
    )
    for filename, workflow in outputs:
        path = CANVAS_DIR / filename
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
        if PORTABLE_CANVAS_DIR.exists():
            target = PORTABLE_CANVAS_DIR / filename
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(target)


if __name__ == "__main__":
    main()
