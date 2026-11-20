"""Carga de las fuentes de datos propias de la organización (etapas 3 y 4).

TODO etapa 3: separar_front_matter y cargar_documentos.
TODO etapa 4: leer_pdf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXTENSIONES = {".md", ".txt", ".pdf"}


@dataclass
class Documento:
    fuente: str          # nombre del archivo, se usa para citar
    texto: str           # contenido sin el front matter
    metadata: dict = field(default_factory=dict)


def separar_front_matter(contenido: str) -> tuple[dict, str]:
    """Separa el bloque '--- clave: valor ---' del resto del texto."""
    if not contenido.startswith("---"):
        return {}, contenido
    partes = contenido.split("---", 2)
    if len(partes) < 3:
        return {}, contenido
    metadata = {}
    for linea in partes[1].strip().splitlines():
        if ":" in linea:
            clave, valor = linea.split(":", 1)
            metadata[clave.strip()] = valor.strip()
    return metadata, partes[2].lstrip()



def leer_pdf(ruta: Path) -> str:
    """Extrae el texto de un PDF página por página."""
    raise NotImplementedError("Etapa 4: leer_pdf")


def leer_archivo(ruta: Path) -> str:
    if ruta.suffix.lower() == ".pdf":
        return leer_pdf(ruta)
    return ruta.read_text(encoding="utf-8")


def cargar_documentos(carpeta: str | Path) -> list[Documento]:
    """Lee todos los archivos soportados de la carpeta, en orden alfabético."""
    carpeta = Path(carpeta)
    documentos = []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        metadata, texto = separar_front_matter(leer_archivo(ruta))
        metadata.setdefault("titulo", ruta.stem.replace("_", " ").capitalize())
        metadata["fuente"] = ruta.name
        documentos.append(Documento(fuente=ruta.name, texto=texto, metadata=metadata))
    return documentos
