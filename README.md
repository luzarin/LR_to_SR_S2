# SEN2SRLite (Windows PowerShell)

Herramienta web para mejorar la resolución espacial de imágenes satelitales de Low-Res (10m) a Super-Res (2.5m) utilizando OpenSR y exportarlas en formato GeoTIFF.

## Instalación

Se recomienda el uso de **`uv`** para instalar una versión específica de Python.

```powershell
# 1. Instalar uv (si no lo tienes)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Instalar la versión exacta de Python (3.11.9)
uv python install 3.11.9

# 3. Clonar el repositorio
git clone https://github.com/luzarin/LR_to_SR.git
cd LR_to_SR

# 4. Crear entorno virtual usando esa versión de Python
uv venv --python 3.11.9
.\.venv\Scripts\activate

# 5. Instalar dependencias usando uv (es mucho más rápido que pip)
uv pip install opensr-utils opensr-model
uv pip install -r requirements.txt
```

6. Instalar PyTorch (Elige una opción, CPU o GPU)

*Para GPU (CUDA 12.1):*
```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*Para CPU:*
```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

7. Verificar si PyTorch está instalado con GPU (Opcional)
```powershell
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

8. Correr la app
```powershell
streamlit run app.py
```