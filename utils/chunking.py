"""
Utilidades para chunking semántico de documentos.
Implementa estrategias de fragmentación que preservan el contexto semántico.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuración para el chunking semántico."""
    chunk_size: int = 500  # Tamaño objetivo del chunk en caracteres
    overlap_size: int = 50  # Tamaño del overlap entre chunks
    min_chunk_size: int = 100  # Tamaño mínimo de chunk
    max_chunk_size: int = 800  # Tamaño máximo de chunk
    preserve_sentences: bool = True  # Preservar oraciones completas
    preserve_paragraphs: bool = True  # Preservar párrafos cuando sea posible


class SemanticChunker:
    """Chunker semántico que preserva el contexto y estructura del texto."""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Fragmenta texto preservando contexto semántico.
        
        Args:
            text: Texto a fragmentar
            
        Returns:
            Lista de chunks con metadatos
        """
        if not text or len(text.strip()) < self.config.min_chunk_size:
            return [{"text": text, "chunk_id": "0", "start_pos": 0, "end_pos": len(text)}]
        
        # Limpiar y normalizar texto
        cleaned_text = self._clean_text(text)
        
        # Estrategia 1: Chunking por párrafos si es posible
        if self.config.preserve_paragraphs:
            chunks = self._chunk_by_paragraphs(cleaned_text)
            if chunks:
                return chunks
        
        # Estrategia 2: Chunking por oraciones
        if self.config.preserve_sentences:
            chunks = self._chunk_by_sentences(cleaned_text)
            if chunks:
                return chunks
        
        # Estrategia 3: Chunking por caracteres con overlap
        return self._chunk_by_characters(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Limpia el texto para chunking."""
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Asegurar que termina con punto si no tiene puntuación
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, any]]:
        """Fragmenta por párrafos preservando contexto."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Si agregar este párrafo excede el tamaño máximo
            if len(current_chunk) + len(para) > self.config.max_chunk_size and current_chunk:
                # Guardar chunk actual
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": str(chunk_id),
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk)
                })
                
                # Iniciar nuevo chunk con overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + para
                start_pos = start_pos + len(current_chunk) - len(overlap_text) - len(para)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Agregar último chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": str(chunk_id),
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk)
            })
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[Dict[str, any]]:
        """Fragmenta por oraciones preservando contexto."""
        # Dividir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Si agregar esta oración excede el tamaño máximo
            if len(current_chunk) + len(sentence) > self.config.max_chunk_size and current_chunk:
                # Guardar chunk actual
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": str(chunk_id),
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk)
                })
                
                # Iniciar nuevo chunk con overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                start_pos = start_pos + len(current_chunk) - len(overlap_text) - len(sentence)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Agregar último chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": str(chunk_id),
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk)
            })
        
        return chunks
    
    def _chunk_by_characters(self, text: str) -> List[Dict[str, any]]:
        """Fragmenta por caracteres con overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            
            # Ajustar para no cortar palabras
            if end < len(text):
                # Buscar último espacio antes del límite
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.config.min_chunk_size:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": str(chunk_id),
                    "start_pos": start,
                    "end_pos": end
                })
            
            # Mover inicio con overlap
            start = max(start + self.config.chunk_size - self.config.overlap_size, end - self.config.overlap_size)
            chunk_id += 1
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Obtiene texto de overlap desde el final del chunk."""
        if len(text) <= self.config.overlap_size:
            return text
        
        # Buscar último espacio antes del overlap
        overlap_start = len(text) - self.config.overlap_size
        last_space = text.rfind(' ', overlap_start)
        
        if last_space > overlap_start - 50:  # Si el espacio está cerca
            return text[last_space + 1:]
        else:
            return text[-self.config.overlap_size:]


def chunk_document_for_rag(text: str, document_id: str, 
                          chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, any]]:
    """
    Función de conveniencia para chunking de documentos para RAG.
    
    Args:
        text: Texto del documento
        document_id: ID del documento
        chunk_size: Tamaño objetivo del chunk
        overlap: Tamaño del overlap
        
    Returns:
        Lista de chunks con metadatos completos
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        overlap_size=overlap,
        min_chunk_size=100,
        max_chunk_size=800
    )
    
    chunker = SemanticChunker(config)
    chunks = chunker.chunk_text(text)
    
    # Agregar metadatos adicionales
    for i, chunk in enumerate(chunks):
        chunk.update({
            "document_id": document_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "word_count": len(chunk["text"].split()),
            "char_count": len(chunk["text"])
        })
    
    return chunks
