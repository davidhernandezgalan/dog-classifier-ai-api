Crear entorno virtual

```bash
python -m venv env
```

Activar entorno

```bash
# Windows
.\env\Scripts\activate
# Linux
source ./env/Scripts/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

Ejecutar servidor

```bash
fastapi run dev
o
uvicorn main:app --reload
```