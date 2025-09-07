"""
Recuperador de contexto para el sistema RAG.
Maneja la búsqueda y recuperación de información relevante.
"""
from typing import List, Dict
from config.settings import logger


class ContextRetriever:
    """Recuperador de contexto que maneja búsquedas por embedding y palabras clave."""
    
    def __init__(self, embedding_model, vector_store, similarity_threshold):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        
        # Entidades críticas que requieren priorización
        self.critical_entities = {
            'denuncia': {
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'context_words': ['presentar', 'cómo', 'como', 'donde', 'dónde', 'procedimiento'],
                'article_patterns': ['art. 5', 'artículo 5', 'art. 5º', 'artículo 5º'],
                'priority': 0.95,
                'secondary_priority': 0.80
            },
            'regimen_disciplinario': {
                'keywords': ['régimen disciplinario', 'regimen disciplinario', 'disciplina', 'sanción', 'sancion'],
                'context_words': ['suspensión', 'suspension', 'aplazo', 'falta', 'sumario'],
                'article_patterns': [],
                'priority': 0.90,
                'secondary_priority': 0.75
            },
            'regularidad': {
                'keywords': ['regularidad', 'regular', 'condiciones'],
                'context_words': ['alumno', 'estudiante', 'requisito', 'mantener', 'perder'],
                'article_patterns': [],
                'priority': 0.90,
                'secondary_priority': 0.75
            }
        }
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Recupera chunks relevantes para una consulta usando múltiples estrategias.
        
        Args:
            query: La consulta del usuario
            k: Número máximo de chunks a recuperar
            
        Returns:
            Lista de chunks relevantes con metadatos
        """
        query = query.strip()
        if not query:
            return []
        
        query_lower = query.lower().strip()
        
        # Detectar entidades críticas
        detected_entities = self._detect_critical_entities(query_lower)
        
        # Estrategia 1: Búsqueda principal por embedding
        results = self._search_by_embedding(query, k)
        
        # Estrategia 2: Búsqueda por variaciones de la consulta
        query_variations = self._generate_query_variations(query)
        for variation in query_variations[:2]:  # Máximo 2 variaciones
            variation_results = self._search_by_embedding(variation, k//2)
            results = self._merge_results(results, variation_results, k)
        
        # Estrategia 3: Procesar entidades críticas si se detectaron
        if detected_entities:
            results = self._process_critical_entities(detected_entities, results, k)
        
        # Estrategia 4: Complementar con palabras clave si es necesario
        results = self._supplement_with_keywords(query_lower, results, k)
        
        # Log de resultados
        self._log_results(results)
        
        return results
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extrae palabras clave importantes de una consulta.
        
        Args:
            query: La consulta en minúsculas
            
        Returns:
            Lista de palabras clave ordenadas por importancia
        """
        # Palabras vacías a excluir
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'con', 'por', 'para', 'a', 'ante',
            'bajo', 'sobre', 'tras', 'durante', 'mediante', 'según',
            'sin', 'so', 'y', 'e', 'o', 'u', 'pero', 'mas', 'sino',
            'que', 'quien', 'quienes', 'cual', 'cuales', 'como',
            'cuando', 'donde', 'porque', 'si', 'no', 'ni', 'ya',
            'se', 'te', 'me', 'le', 'nos', 'os', 'les', 'lo', 'la',
            'los', 'las', 'su', 'sus', 'mi', 'mis', 'tu', 'tus',
            'es', 'son', 'está', 'están', 'fue', 'fueron', 'será',
            'serán', 'ha', 'han', 'había', 'habían', 'habrá', 'habrán',
            'hacer', 'hace', 'hizo', 'hará', 'haga', 'muy', 'más',
            'menos', 'tanto', 'tan', 'todo', 'toda', 'todos', 'todas',
            'algún', 'alguna', 'algunos', 'algunas', 'ningún', 'ninguna',
            'ningunos', 'ningunas', 'otro', 'otra', 'otros', 'otras',
            'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos',
            'esas', 'aquel', 'aquella', 'aquellos', 'aquellas',
            'mismo', 'misma', 'mismos', 'mismas'
        }
        
        # Limpiar y tokenizar
        import re
        words = re.findall(r'\b\w{3,}\b', query.lower())
        words = [word for word in words if word not in stop_words]
        
        # Palabras importantes con mayor peso
        important_terms = {
            'denuncia': 10, 'denunciar': 10, 'denuncias': 10,
            'regularidad': 8, 'regular': 8, 'condiciones': 8,
            'disciplinario': 8, 'sanción': 8, 'sancion': 8,
            'estudiante': 6, 'alumno': 6, 'materia': 6, 'materias': 6,
            'examen': 6, 'examenes': 6, 'curso': 6, 'cursos': 6,
            'inscripción': 6, 'inscripcion': 6, 'tramite': 6, 'tramites': 6,
            'certificado': 5, 'titulo': 5, 'graduación': 5, 'graduacion': 5,
            'presentar': 4, 'solicitar': 4, 'obtener': 4, 'conseguir': 4,
            'requisito': 4, 'requisitos': 4, 'procedimiento': 4,
            'plazo': 4, 'fecha': 4, 'horario': 4, 'horarios': 4
        }
        
        # Calcular puntuaciones
        word_scores = {}
        for word in words:
            score = important_terms.get(word, 1)
            # Aumentar score si aparece múltiples veces
            word_scores[word] = word_scores.get(word, 0) + score
        
        # Ordenar por puntuación
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:10]]
    
    def enhance_context(self, query: str, context: str) -> str:
        """
        Mejora el contexto con información específica según el tipo de consulta.
        
        Args:
            query: La consulta original
            context: El contexto base
            
        Returns:
            Contexto mejorado
        """
        query_lower = query.lower()
        enhanced_context = context
        
        # Mensajes introductorios específicos
        intro_messages = {
            'denuncia': "INFORMACIÓN SOBRE DENUNCIAS UNIVERSITARIAS:",
            'regularidad': "INFORMACIÓN SOBRE REGULARIDAD ACADÉMICA:",
            'disciplinario': "INFORMACIÓN SOBRE RÉGIMEN DISCIPLINARIO:",
            'examen': "INFORMACIÓN SOBRE EXÁMENES:",
            'inscripción': "INFORMACIÓN SOBRE INSCRIPCIONES:",
            'certificado': "INFORMACIÓN SOBRE CERTIFICADOS Y DOCUMENTACIÓN:"
        }
        
        # Detectar tipos de consulta
        detected_types = []
        for doc_type, keywords in {
            'denuncia': ['denuncia', 'denunciar', 'denuncias'],
            'regularidad': ['regularidad', 'regular', 'condiciones'],
            'disciplinario': ['disciplinario', 'sanción', 'sancion', 'suspensión'],
            'examen': ['examen', 'examenes', 'evaluación', 'evaluacion'],
            'inscripción': ['inscripción', 'inscripcion', 'inscribir'],
            'certificado': ['certificado', 'constancia', 'titulo', 'diploma']
        }.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_types.append(doc_type)
        
        # Agregar mensaje introductorio si corresponde
        if detected_types:
            for doc_type in detected_types:
                if doc_type in intro_messages:
                    enhanced_context = f"{intro_messages[doc_type]}\n\n{enhanced_context}"
                    break
        
        return enhanced_context
    
    def _detect_critical_entities(self, query_lower: str) -> List:
        """Detecta entidades críticas en la consulta."""
        detected_entities = []
        
        for entity_name, entity_data in self.critical_entities.items():
            if any(keyword in query_lower for keyword in entity_data['keywords']):
                has_context = any(context in query_lower 
                                for context in entity_data['context_words'])
                detected_entities.append((entity_name, entity_data, has_context))
        
        return detected_entities
    
    def _search_by_embedding(self, query: str, k: int) -> List[Dict]:
        """Realiza búsqueda principal por embedding."""
        query_embedding = self.embedding_model.encode([query])[0]
        return self.vector_store.search(query_embedding, k=k)
    
    def _process_critical_entities(self, detected_entities: List, 
                                 results: List[Dict], k: int) -> List[Dict]:
        """Procesa entidades críticas detectadas."""
        logger.info(f"Detectadas entidades críticas: {[e[0] for e in detected_entities]}")
        
        priority_chunks = []
        for entity_name, entity_data, is_high_priority in detected_entities:
            logger.info(f"Realizando búsqueda prioritaria para entidad: {entity_name}")
            # Aquí se podría implementar búsqueda específica si el vector store lo soporta
        
        # Añadir chunks prioritarios sin duplicados
        if priority_chunks:
            for chunk in priority_chunks:
                if not any(r.get('text') == chunk.get('text') for r in results):
                    results.insert(0, chunk)
            
            if len(results) > k + 3:
                results = results[:k + 3]
            
            logger.info(f"Se agregaron {len(priority_chunks)} chunks prioritarios")
        
        return results
    
    def _supplement_with_keywords(self, query_lower: str, 
                                results: List[Dict], k: int) -> List[Dict]:
        """Complementa resultados con búsqueda por palabras clave si es necesario."""
        if (len(results) < 3 or 
            (results and results[-1]['similarity'] < self.similarity_threshold + 0.05)):
            
            important_keywords = self.extract_keywords_from_query(query_lower)
            logger.info(f"Resultados iniciales insuficientes, "
                       f"intentando complementar con palabras clave: {important_keywords}")
            
            # Búsquedas adicionales por palabras clave importantes
            for keyword in important_keywords[:3]:
                logger.info(f"Búsqueda adicional por palabra clave: {keyword}")
                # Implementar búsqueda por keyword si el vector store lo soporta
        
        return results
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Genera variaciones de la consulta para mejorar la recuperación."""
        variations = []
        query_lower = query.lower()
        
        # Variación 1: Sin palabras de conexión
        stop_words = ['como', 'cómo', 'donde', 'dónde', 'que', 'qué', 'para', 'por', 'con', 'de', 'la', 'el', 'los', 'las']
        words = [word for word in query_lower.split() if word not in stop_words]
        if len(words) > 2:
            variations.append(' '.join(words))
        
        # Variación 2: Forma más formal
        formal_mapping = {
            'como': 'cómo',
            'donde': 'dónde', 
            'que': 'qué',
            'cuando': 'cuándo',
            'porque': 'por qué'
        }
        formal_words = []
        for word in query_lower.split():
            formal_words.append(formal_mapping.get(word, word))
        variations.append(' '.join(formal_words))
        
        # Variación 3: Sinónimos académicos
        academic_synonyms = {
            'tramite': 'trámite',
            'tramites': 'trámites',
            'inscripcion': 'inscripción',
            'inscripciones': 'inscripciones',
            'examen': 'evaluación',
            'examenes': 'evaluaciones'
        }
        synonym_words = []
        for word in query_lower.split():
            synonym_words.append(academic_synonyms.get(word, word))
        variations.append(' '.join(synonym_words))
        
        return variations
    
    def _merge_results(self, results1: List[Dict], results2: List[Dict], k: int) -> List[Dict]:
        """Combina resultados de diferentes búsquedas eliminando duplicados."""
        # Crear un diccionario para evitar duplicados
        unique_results = {}
        
        # Agregar resultados de la primera búsqueda
        for result in results1:
            text_key = result.get('text', '')[:100]  # Usar primeros 100 chars como clave
            if text_key not in unique_results or result.get('similarity', 0) > unique_results[text_key].get('similarity', 0):
                unique_results[text_key] = result
        
        # Agregar resultados de la segunda búsqueda
        for result in results2:
            text_key = result.get('text', '')[:100]
            if text_key not in unique_results or result.get('similarity', 0) > unique_results[text_key].get('similarity', 0):
                unique_results[text_key] = result
        
        # Convertir de vuelta a lista y ordenar por similitud
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return merged_results[:k]
    
    def _log_results(self, results: List[Dict]):
        """Registra información sobre los resultados encontrados."""
        logger.info(f"Número total de chunks recuperados: {len(results)}")
        
        for i, result in enumerate(results):
            filename = result.get('filename', 'unknown')
            similarity = result.get('similarity', 0)
            text_preview = result.get('text', '')[:100]
            if len(result.get('text', '')) > 100:
                text_preview += '...'
            
            logger.info(f"Chunk {i+1}: {filename} (similitud: {similarity:.2f}) - {text_preview}")