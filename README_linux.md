# SEN2SRLite (Linux / WSL2)

Herramienta web para mejorar la resolución espacial de imágenes satelitales de Low-Res (10m) a Super-Res (2.5m) utilizando OpenSR y exportarlas en formato GeoTIFF.

## Instalación

Se recomienda instalar las dependencias siguiendo estos pasos:

```bash
# 1. Clonar el repositorio y entrar al directorio
git clone https://github.com/luzarin/LR_to_SR.git
cd LR_to_SR

# 2. Instalar la versión exacta de Python y definirla localmente
pyenv install 3.11.9
pyenv local 3.11.9

# 3. Crear entorno virtual y activarlo
python -m venv .venv
source .venv/bin/activate

# 4. Actualizar pip e instalar dependencias base
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

5. Instalar PyTorch (Elige una opción, CPU o GPU)

*Para GPU (CUDA 12.1):*
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

*Para CPU:*
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

6. Verificar si PyTorch está instalado con GPU (Opcional)
```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

7. Correr la app
```bash
streamlit run app.py
```
