# Low-Res to High-Res Sentinel-2 (macOS)

Herramienta web para mejorar la resolucion espacial de imagenes satelitales Sentinel-2 de Low-Res (10m) a Super-Res (2.5m).

## Modelos

- `./SEN2SRLite`: pipeline de 10 bandas (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
- `./ldsr-s2`: pipeline RGB+NIR (B04, B03, B02, B08), salida de 4 bandas.

## Entrada y salida

| Tipo | Formato | Notas |
|---|---|---|
| **Input** | GeoTIFF multibanda | SEN2SRLite espera 10 bandas. LDSR-S2 usa RGB+NIR (si hay mas bandas, toma automaticamente B04/B03/B02/B08). |
| **Output** | GeoTIFF multibanda | SEN2SRLite entrega 10 bandas. LDSR-S2 entrega 4 bandas. |

## Requisitos

- macOS
- Python 3.11.9 (recomendado con `pyenv`)
- Zsh o Bash
- GPU NVIDIA opcional (CUDA) para acelerar inferencia (metal en Apple Silicon)

## Instalacion

```bash
# 1. Clonar el repositorio
git clone https://github.com/luzarin/LR_to_SR_S2.git
cd LR_to_SR_S2

# 2. Instalar y configurar version de Python
pyenv install 3.11.9
pyenv local 3.11.9

# 3. Crear entorno virtual y activarlo
python -m venv .venv
source .venv/bin/activate

# 4. Instalar dependencias
pip install -U pip setuptools wheel
pip install opensr-utils opensr-model
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Verificacion GPU (opcional)

```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
