"""Etapa 2 · Bases de datos vectoriales y búsqueda por similitud.

TODO etapa 2: implemente los dos almacenes siguiendo la guía.
Prueba: pytest tests/test_etapa2_vectores.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Resultado:
    id: str
    texto: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # similitud coseno: 1 = idéntico, 0 = nada en común

class MemoryVectorStore:
    """Almacén vectorial en memoria con búsqueda exacta por similitud coseno."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.ids: list[str] = []
        self.textos: list[str] = []
        self.metadatas: list[dict] = []
        self.vectores = np.zeros((0, embedder.dim), dtype=np.float32)

    def agregar(self, ids, textos, metadatas=None) -> None:
        metadatas = metadatas or [{} for _ in textos]
        nuevos = self.embedder.embed_many(textos)
        self.vectores = np.vstack([self.vectores, nuevos])
        self.ids += list(ids)
        self.textos += list(textos)
        self.metadatas += list(metadatas)

    def contar(self) -> int:
        return len(self.ids)

    def buscar(self, consulta: str, k: int = 4, filtro: dict | None = None) -> list[Resultado]:
        if self.contar() == 0:
            return []
        q = self.embedder.embed(consulta)
        # Vectores normalizados: el producto punto ES la similitud coseno.
        puntajes = self.vectores @ q
        candidatos = range(len(self.ids))
        if filtro:
            candidatos = [
                i for i in candidatos
                if all(self.metadatas[i].get(c) == v for c, v in filtro.items())
            ]
        orden = sorted(candidatos, key=lambda i: float(puntajes[i]), reverse=True)[:k]
        return [
            Resultado(self.ids[i], self.textos[i], self.metadatas[i], float(puntajes[i]))
            for i in orden
        ]

class ChromaVectorStore:
    """Nosotros calculamos los embeddings y se los pasamos a Chroma; así podemos
    cambiar de modelo sin depender del modelo por defecto de Chroma."""

    def __init__(self, embedder, ruta: str, coleccion: str = "andina") -> None:
        import chromadb

        self.embedder = embedder
        self.cliente = chromadb.PersistentClient(path=ruta)
        self.nombre = coleccion
        self.coleccion = self._abrir()

    def _abrir(self):
        return self.cliente.get_or_create_collection(
            name=self.nombre,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},  # distancia coseno = 1 - similitud
        )

    def reiniciar(self) -> None:
        try:
            self.cliente.delete_collection(self.nombre)
        except Exception:
            pass
        self.coleccion = self._abrir()

    def agregar(self, ids, textos, metadatas=None) -> None:
        if not textos:
            return
        vectores = self.embedder.embed_many(textos)
        self.coleccion.upsert(
            ids=list(ids),
            documents=list(textos),
            embeddings=vectores.tolist(),
            metadatas=metadatas or [{"_": ""} for _ in textos],
        )

    def contar(self) -> int:
        return self.coleccion.count()

    def buscar(self, consulta: str, k: int = 4, filtro: dict | None = None) -> list[Resultado]:
        if self.contar() == 0:
            return []
        q = self.embedder.embed(consulta)
        respuesta = self.coleccion.query(
            query_embeddings=[q.tolist()],
            n_results=min(k, self.contar()),
            where=filtro or None,
        )
        return [
            Resultado(id_, texto, meta or {}, 1.0 - float(dist))
            for id_, texto, meta, dist in zip(
                respuesta["ids"][0],
                respuesta["documents"][0],
                respuesta["metadatas"][0],
                respuesta["distances"][0],
            )
        ]
