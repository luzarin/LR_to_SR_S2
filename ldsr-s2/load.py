import pathlib
from io import StringIO
from typing import Union

import requests
import torch
from omegaconf import OmegaConf

import opensr_model

EXPECTED_INPUT_BANDS = 4
RECOMMENDED_BATCH = 6
CONFIG_FILENAME = "config_10m.yaml"
DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)
DEFAULT_WEIGHTS_BASE_URL = "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main"


def _download_text(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _download_binary(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(dest)


def _load_config(path: pathlib.Path):
    cfg_path = path / CONFIG_FILENAME
    if cfg_path.exists():
        return OmegaConf.load(cfg_path)

    text = _download_text(DEFAULT_CONFIG_URL)
    cfg_path.write_text(text, encoding="utf-8")
    return OmegaConf.load(StringIO(text))


def _ensure_checkpoint(path: pathlib.Path, ckpt_name: str) -> pathlib.Path:
    ckpt_path = path / ckpt_name
    if ckpt_path.exists():
        return ckpt_path

    url = f"{DEFAULT_WEIGHTS_BASE_URL}/{ckpt_name}"
    _download_binary(url, ckpt_path)
    return ckpt_path


def _build_model(path: pathlib.Path, device: Union[str, torch.device] = "cpu"):
    model_path = pathlib.Path(path)
    model_path.mkdir(parents=True, exist_ok=True)

    config = _load_config(model_path)
    ckpt_name = str(config.ckpt_version)
    ckpt_path = _ensure_checkpoint(model_path, ckpt_name)

    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(str(ckpt_path))
    model = model.to(device)
    model.eval()
    return model


def trainable_model(path, device: str = "cpu", *args, **kwargs):
    return _build_model(pathlib.Path(path), device=device)


def compiled_model(path, device: str = "cpu", *args, **kwargs):
    return _build_model(pathlib.Path(path), device=device)
