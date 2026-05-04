import os, sys, time, json, importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Permite importar modulos locales cuando corres desde la raiz del repo.
sys.path.insert(0, ".")

BAND_KEYS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RGBN_BAND_KEYS = ["B04", "B03", "B02", "B08"]
RGBN_ALIAS_MAP = {
    "R": {"R", "RED", "B04", "B4", "BAND4", "BAND04"},
    "G": {"G", "GREEN", "B03", "B3", "BAND3", "BAND03"},
    "B": {"B", "BLUE", "B02", "B2", "BAND2", "BAND02"},
    "NIR": {"NIR", "B08", "B8", "BAND8", "BAND08"},
}

BAND_ROLE_LABELS = {
    "B01": "Aerosols",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Red Edge 1",
    "B06": "Red Edge 2",
    "B07": "Red Edge 3",
    "B08": "NIR",
    "B8A": "NIR Narrow",
    "B11": "SWIR 1",
    "B12": "SWIR 2",
}

app = FastAPI(title="Sentinel-2 SR API")

# Aseguramos que exista static
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


# -------------------------- Utils --------------------------
def list_tifs(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []

    files = []
    for x in p.iterdir():
        if x.is_file() and x.suffix.lower() in {".tif", ".tiff"}:
            files.append(x.resolve())

    uniq = sorted({str(x): x for x in files}.keys(), key=lambda s: Path(s).name.lower())
    return uniq


def safe_delete(path: str):
    if os.path.exists(path):
        os.remove(path)


def pad_to_patch(x: np.ndarray, patch: int):
    c, h, w = x.shape
    out = np.zeros((c, patch, patch), dtype=x.dtype)
    out[:, :h, :w] = x
    return out, (h, w)


def normalize_input(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx > 1.5:  # tipico Sentinel en 0..10000
        x = x / 10000.0
    return x


def denormalize_output(y: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(y)) if y.size else 0.0
    if mx <= 1.5:  # tipico si esta en [0,1]
        y = y * 10000.0
    y = np.clip(y, 0.0, 10000.0)
    return y.astype(np.uint16)


def resolve_device(device_mode: str) -> str:
    if device_mode == "cpu":
        return "cpu"
    if device_mode.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Seleccionaste GPU pero torch no detecta CUDA.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_loader_module(weights_dir: Path):
    load_py = weights_dir / "load.py"
    if not load_py.exists():
        raise RuntimeError(
            f"No existe loader en {load_py}. El directorio de pesos debe contener load.py"
        )

    module_name = f"weights_loader_{abs(hash(str(load_py.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(load_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el modulo desde {load_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "trainable_model"):
        raise RuntimeError(f"{load_py} no define trainable_model(path, device=...)")
    return module


def get_band_names(src) -> List[str]:
    src_desc = list(src.descriptions) if src.descriptions else [None] * src.count
    band_names = []
    for i in range(src.count):
        desc = src_desc[i] if i < len(src_desc) else None
        if desc is None or str(desc).strip() == "":
            desc = BAND_KEYS[i] if i < len(BAND_KEYS) else f"Band {i + 1}"
        band_names.append(str(desc))
    return band_names


def normalize_band_token(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def infer_rgbn_indices_from_metadata(src) -> Optional[List[int]]:
    selected: Dict[str, int] = {}
    for band_idx in range(1, src.count + 1):
        tokens = set()

        if src.descriptions and band_idx - 1 < len(src.descriptions):
            desc = src.descriptions[band_idx - 1]
            if desc:
                tokens.add(normalize_band_token(desc))

        for key, value in src.tags(band_idx).items():
            tokens.add(normalize_band_token(key))
            tokens.add(normalize_band_token(value))

        for channel, aliases in RGBN_ALIAS_MAP.items():
            if channel in selected:
                continue
            if any(tok in aliases for tok in tokens):
                selected[channel] = band_idx

    if len(selected) != 4:
        return None

    rgbn = [selected["R"], selected["G"], selected["B"], selected["NIR"]]
    if len(set(rgbn)) != 4:
        return None
    return rgbn


def resolve_input_band_selection(
    src,
    expected_input_bands: Optional[int],
) -> Tuple[List[int], List[str], Optional[str]]:
    all_band_names = get_band_names(src)

    if expected_input_bands is None:
        return list(range(1, src.count + 1)), all_band_names, None

    if expected_input_bands <= 0:
        raise RuntimeError(f"Valor invalido EXPECTED_INPUT_BANDS={expected_input_bands}")

    if expected_input_bands == 10:
        if src.count != 10:
            raise RuntimeError(f"Expected 10 bands input. Got {src.count}.")
        return list(range(1, 11)), all_band_names, None

    if expected_input_bands == 4:
        selection_reason = None
        rgbn_indices = infer_rgbn_indices_from_metadata(src)
        if rgbn_indices is not None:
            selection_reason = (
                "RGB-NIR detectado por metadatos. "
                f"Usando bandas 1-based: {rgbn_indices}"
            )
        elif src.count >= 7:
            # Fallback Sentinel-2: [B02, B03, B04, B05, B06, B07, B08, ...]
            rgbn_indices = [3, 2, 1, 7]  # R, G, B, NIR
            selection_reason = (
                "RGB-NIR detectado por orden Sentinel-2 canonico. "
                f"Usando bandas 1-based: {rgbn_indices}"
            )
        elif src.count == 4:
            # Fallback mas comun para stacks Sentinel-2 de 4 bandas: [B02, B03, B04, B08].
            rgbn_indices = [3, 2, 1, 4]  # R, G, B, NIR
            selection_reason = (
                "Input de 4 bandas sin metadata clara; se asume orden Sentinel-2 "
                "[B02,B03,B04,B08]. Usando RGB-NIR con indices 1-based: [3,2,1,4]"
            )
        else:
            raise RuntimeError(
                "El modelo de 4 bandas requiere RGB-NIR. "
                f"No se pudo inferir desde un raster de {src.count} bandas."
            )

        selected_names = []
        for out_idx, src_idx in enumerate(rgbn_indices):
            if 1 <= src_idx <= len(all_band_names):
                selected_names.append(all_band_names[src_idx - 1])
            else:
                selected_names.append(RGBN_BAND_KEYS[out_idx])

        # Si el raster de 4 bandas no trae descripciones, usa nombres canonicos RGB-NIR.
        if src.count == 4 and not any(src.descriptions or []):
            selected_names = RGBN_BAND_KEYS.copy()

        return rgbn_indices, selected_names, selection_reason

    if src.count != expected_input_bands:
        raise RuntimeError(
            f"Expected {expected_input_bands} bands input. Got {src.count}."
        )

    return (
        list(range(1, expected_input_bands + 1)),
        all_band_names[:expected_input_bands],
        None,
    )


def get_output_canonical_bands(expected_input_bands: Optional[int], output_count: int) -> List[str]:
    if expected_input_bands == 4:
        base = RGBN_BAND_KEYS
    elif expected_input_bands == 10:
        base = BAND_KEYS
    else:
        base = []

    if len(base) >= output_count:
        return base[:output_count]

    extra = [f"Band {i + 1}" for i in range(len(base), output_count)]
    return base + extra


def build_output_band_annotations(
    expected_input_bands: Optional[int],
    selected_band_indices: List[int],
    selected_source_names: List[str],
) -> List[Dict[str, str]]:
    canonical_bands = get_output_canonical_bands(expected_input_bands, len(selected_band_indices))
    annotations: List[Dict[str, str]] = []

    for out_idx, src_idx in enumerate(selected_band_indices, start=1):
        canonical_band = canonical_bands[out_idx - 1] if out_idx - 1 < len(canonical_bands) else f"Band {out_idx}"
        role = BAND_ROLE_LABELS.get(canonical_band, "Unknown")
        src_name = selected_source_names[out_idx - 1] if out_idx - 1 < len(selected_source_names) else ""
        description = f"{canonical_band} ({role})"

        annotations.append(
            {
                "out_index": str(out_idx),
                "src_index": str(src_idx),
                "canonical_band": canonical_band,
                "role": role,
                "source_name": src_name,
                "description": description,
            }
        )

    return annotations


def build_model(weights_dir: Path, device: str):
    loader = load_loader_module(weights_dir)
    obj = loader.trainable_model(weights_dir, device=device)

    if callable(obj):
        model = obj
    elif isinstance(obj, dict):
        model = None
        for k in ["model", "sr_model", "net", "network", "sr"]:
            if k in obj and callable(obj[k]):
                model = obj[k]
                break
        if model is None:
            for v in obj.values():
                if callable(v):
                    model = v
                    break
    elif isinstance(obj, (tuple, list)):
        model = next((v for v in obj if callable(v)), None)
    else:
        model = None

    if model is None:
        raise RuntimeError(f"trainable_model() returned unsupported type: {type(obj)}")

    expected_bands = getattr(loader, "EXPECTED_INPUT_BANDS", None)
    if expected_bands is not None:
        try:
            expected_bands = int(expected_bands)
        except Exception:
            raise RuntimeError(
                f"EXPECTED_INPUT_BANDS invalido en {weights_dir / 'load.py'}: {expected_bands}"
            )

    recommended_batch = getattr(loader, "RECOMMENDED_BATCH", None)
    if recommended_batch is not None:
        try:
            recommended_batch = int(recommended_batch)
            if recommended_batch <= 0:
                raise ValueError
        except Exception:
            raise RuntimeError(
                f"RECOMMENDED_BATCH invalido en {weights_dir / 'load.py'}: {recommended_batch}"
            )

    runtime_hints = {
        "recommended_batch": recommended_batch,
    }

    return model, expected_bands, runtime_hints


@torch.no_grad()
def run_batch(model, device, batch_x_np):
    x = np.stack(batch_x_np, axis=0)  # (B,C,H,W)
    xt = torch.from_numpy(x).to(device)
    yt = model(xt)
    if isinstance(yt, (tuple, list)):
        yt = yt[0]
    if isinstance(yt, dict):
        yt = yt.get("raster_pixels", next(iter(yt.values())))
    y = yt.detach().cpu().numpy()
    return [y[i] for i in range(y.shape[0])]


# -------------------------- Inference Generator --------------------------
def run_sr_generator(
    weights_dir: Path,
    in_tif: str,
    out_tif: str,
    factor: int,
    patch: int,
    pad: int,
    batch: int,
    device_mode: str,
    ui_update_every: float = 0.25,
):
    start_t = time.time()
    step = patch - 2 * pad
    if step <= 0:
        yield {"error": "PATCH - 2*PAD must be > 0."}
        return

    try:
        device = resolve_device(device_mode)
        yield {"log": f"Dispositivo resuelto: {device}"}

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        model, expected_input_bands, runtime_hints = build_model(weights_dir, device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)

        recommended_batch = runtime_hints.get("recommended_batch")
        if recommended_batch is not None and batch > recommended_batch:
            yield {
                "log": (
                    f"Batch {batch} demasiado alto para este modelo. "
                    f"Se ajusta automaticamente a {recommended_batch}."
                )
            }
            batch = recommended_batch

        yield {
            "log": (
                f"Modelo cargado desde: {weights_dir / 'load.py'} | "
                f"EXPECTED_INPUT_BANDS={expected_input_bands if expected_input_bands is not None else 'auto'}"
            )
        }

        os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
        safe_delete(out_tif)

        with rasterio.open(in_tif) as src:
            selected_band_indices, selected_source_band_names, band_selection_log = resolve_input_band_selection(
                src, expected_input_bands
            )
            if band_selection_log:
                yield {"log": band_selection_log}

            output_band_annotations = build_output_band_annotations(
                expected_input_bands=expected_input_bands,
                selected_band_indices=selected_band_indices,
                selected_source_names=selected_source_band_names,
            )
            mapping_log = " | ".join(
                [
                    f"{a['out_index']}:{a['canonical_band']}({a['role']}) <- src{a['src_index']}[{a['source_name']}]"
                    for a in output_band_annotations
                ]
            )
            yield {"log": f"Band mapping output: {mapping_log}"}

            global_tags = src.tags()
            per_band_tags = {
                out_i: src.tags(src_i)
                for out_i, src_i in enumerate(selected_band_indices, start=1)
            }

            profile = src.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)

            profile.update(
                driver="GTiff",
                dtype="uint16",
                count=len(selected_band_indices),
                width=src.width * factor,
                height=src.height * factor,
                transform=src.transform * src.transform.scale(1 / factor, 1 / factor),
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )

            total_tiles = ((src.height + step - 1) // step) * ((src.width + step - 1) // step)

            done_tiles = 0
            last_draw = 0.0

            yield {
                "progress": 0.0,
                "status": f"Iniciando... Total tiles: {total_tiles} | device={device}",
                "stage": "running"
            }

            with rasterio.open(out_tif, "w", **profile) as dst:
                if global_tags:
                    dst.update_tags(**global_tags)

                dst.update_tags(
                    sr_output_band_mapping="; ".join(
                        [f"{a['out_index']}:{a['canonical_band']}({a['role']})" for a in output_band_annotations]
                    ),
                    sr_selected_input_indices=",".join([a["src_index"] for a in output_band_annotations]),
                    sr_expected_input_bands=(
                        str(expected_input_bands) if expected_input_bands is not None else "auto"
                    ),
                )

                for i, ann in enumerate(output_band_annotations, start=1):
                    dst.set_band_description(i, ann["description"])

                    tags_i = dict(per_band_tags.get(i) or {})
                    tags_i.update(
                        {
                            "sr_output_band_index": str(i),
                            "sr_source_band_index": ann["src_index"],
                            "sr_source_band_name": ann["source_name"],
                            "sr_canonical_band": ann["canonical_band"],
                            "sr_band_role": ann["role"],
                        }
                    )
                    dst.update_tags(i, **tags_i)

                batch_tiles = []
                batch_meta = []

                def flush():
                    nonlocal batch_tiles, batch_meta
                    if not batch_tiles:
                        return

                    outs = run_batch(model, device, batch_tiles)

                    for out_sr, meta in zip(outs, batch_meta):
                        (c0, r0, w, h, is_left, is_top, is_right, is_bottom) = meta

                        left = 0 if is_left else pad
                        top = 0 if is_top else pad
                        right = w if is_right else (w - pad)
                        bottom = h if is_bottom else (h - pad)

                        sr_left = left * factor
                        sr_top = top * factor
                        sr_right = right * factor
                        sr_bottom = bottom * factor

                        out_crop = out_sr[:, sr_top:sr_bottom, sr_left:sr_right]
                        out_crop = denormalize_output(out_crop)

                        out_win = Window(
                            (c0 + left) * factor,
                            (r0 + top) * factor,
                            (right - left) * factor,
                            (bottom - top) * factor,
                        )

                        dst.write(out_crop, window=out_win)

                    batch_tiles = []
                    batch_meta = []

                for r0 in range(0, src.height, step):
                    for c0 in range(0, src.width, step):
                        w = min(patch, src.width - c0)
                        h = min(patch, src.height - r0)

                        win = Window(c0, r0, w, h)
                        x = src.read(indexes=selected_band_indices, window=win)
                        x = normalize_input(x)
                        xpad, (hh, ww) = pad_to_patch(x, patch)

                        is_left = (c0 == 0)
                        is_top = (r0 == 0)
                        is_right = (c0 + w >= src.width)
                        is_bottom = (r0 + h >= src.height)

                        batch_tiles.append(xpad)
                        batch_meta.append((c0, r0, ww, hh, is_left, is_top, is_right, is_bottom))

                        done_tiles += 1
                        now = time.time()

                        if now - last_draw > ui_update_every:
                            pct = done_tiles / total_tiles if total_tiles else 1.0
                            elapsed = now - start_t
                            rate = done_tiles / elapsed if elapsed > 0 else 0.0
                            eta = (total_tiles - done_tiles) / rate if rate > 0 else 0.0

                            status_msg = (
                                f"device={device} | tiles {done_tiles}/{total_tiles} | "
                                f"{rate:.2f} tiles/s | ETA {int(eta//60):02d}:{int(eta%60):02d}"
                            )
                            yield {"progress": pct * 100, "status": status_msg, "stage": "running"}
                            last_draw = now

                        if len(batch_tiles) == batch:
                            flush()

                flush()

            yield {
                "progress": 100.0,
                "status": f"Completado. Guardado en {out_tif}",
                "stage": "completed"
            }

    except Exception as e:
        err = str(e)
        if "CUDA out of memory" in err:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            err = (
                err
                + " | Sugerencia: usa batch=1 para LDSR-S2, cierra otros procesos que usen GPU "
                + "y reintenta."
            )
        yield {"error": err, "stage": "error"}


# -------------------------- API Endpoints --------------------------
@app.get("/api/tifs")
def api_get_tifs(folder: str = "./input_10bands_LR"):
    return {"files": [Path(f).name for f in list_tifs(folder)]}


@app.get("/api/run")
def api_run_get(
    weights_dir: str = "./SEN2SRLite",
    in_tif: str = "",
    out_dir: str = "./output_10bands_SR",
    out_name: str = "",
    factor: int = 4,
    patch: int = 128,
    pad: int = 4,
    batch: int = 7,
    device: str = "auto"
):
    if not Path(weights_dir).exists():
        return StreamingResponse((f"data: {json.dumps({'error': f'No existe WEIGHTS_DIR: {weights_dir}'})}\n\n" for _ in range(1)), media_type="text/event-stream")
    if not Path(in_tif).exists():
        return StreamingResponse((f"data: {json.dumps({'error': f'No existe Input: {in_tif}'})}\n\n" for _ in range(1)), media_type="text/event-stream")

    out_path = str(Path(out_dir) / out_name)

    def event_generator():
        for event_data in run_sr_generator(Path(weights_dir), in_tif, out_path, factor, patch, pad, batch, device):
            yield f"data: {json.dumps(event_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/device_info")
def api_device_info():
    cuda_av = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_av else "N/A"
    return {
        "cuda_available": cuda_av,
        "gpu_name": gpu_name
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
