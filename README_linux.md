# Low-Res to High-Res Sentinel-2 (Linux / WSL2)

Herramienta web para mejorar la resolucion espacial de imagenes satelitales Sentinel-2 de Low-Res (10m) a Super-Res (2.5m).

## Modelos

- `./SEN2SRLite`: pipeline de 10 bandas (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
- `./ldsr-s2`: pipeline RGB+NIR (B04, B03, B02, B08), salida de 4 bandas.
- En la UI se elige desde el toggle de **Weights Directory**.

## Entrada y salida

| Tipo | Formato | Notas |
|---|---|---|
| **Input** | GeoTIFF multibanda | SEN2SRLite espera 10 bandas. LDSR-S2 usa RGB+NIR (si hay mas bandas, toma automaticamente B04/B03/B02/B08). |
| **Output** | GeoTIFF multibanda | SEN2SRLite entrega 10 bandas. LDSR-S2 entrega 4 bandas. |

## Requisitos

- Linux o WSL2
- Python 3.11.9 (recomendado con `pyenv`)
- Bash
- GPU NVIDIA opcional (CUDA) para acelerar inferencia

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Verificacion GPU (opcional)

```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

## Uso

```bash
uv run python -m uvicorn app:app --reload --host 127.0.0.1 --port 8010
```

Abrir en el navegador: [http://127.0.0.1:8010](http://127.0.0.1:8010)

### Flujo basico

1. Colocar los GeoTIFF en `input_10bands_LR/`.
2. Abrir la UI y refrescar la lista de archivos.
3. Elegir el modelo en `Weights Directory` (`./SEN2SRLite` o `./ldsr-s2`).
4. Elegir `Device Mode` (`auto`, `cuda` o `cpu`).
5. Ajustar parametros (`factor`, `patch`, `pad`, `batch`) si hace falta.
6. Ejecutar el proceso y revisar el progreso en tiempo real.
7. El resultado se guarda en `output_10bands_SR/`.

## Parametros principales

| Parametro | Default | Descripcion |
|---|---|---|
| `factor` | `4` | Escala espacial (10m -> 2.5m cuando factor=4) |
| `patch` | `128` | Tamano del tile de inferencia |
| `pad` | `4` | Padding para evitar bordes entre tiles |
| `batch` | `7` | Cantidad de tiles por batch |
| `device` | `auto` | Seleccion de computo (`auto`/`cuda`/`cpu`) |
| `weights_dir` | `./ldsr-s2` | Seleccion de modelo (desde el toggle de UI) |

## Estructura

```text
LR_to_SR_S2/
|-- app.py                    # API FastAPI + motor de inferencia
|-- static/
|   `-- index.html            # Interfaz web
|-- SEN2SRLite/               # Pesos .safetensor y loader SEN2SRLite
|-- ldsr-s2/                  # Loader/pesos LDSR-S2
|-- sen2sr/                   # Modulos/modelos auxiliares
|-- input_10bands_LR/         # Inputs GeoTIFF
|-- output_10bands_SR/        # Outputs GeoTIFF super-resueltos
`-- requirements.txt          # Dependencias
```

## API

| Metodo | Endpoint | Descripcion |
|---|---|---|
| `GET` | `/` | Sirve la UI |
| `GET` | `/api/device_info` | Informa disponibilidad CUDA y nombre de GPU |
| `GET` | `/api/tifs` | Lista TIFF disponibles en la carpeta input |
| `GET` | `/api/run` | Ejecuta inferencia SR y stream de progreso (SSE) |
