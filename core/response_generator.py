"""
Generador de respuestas para el sistema RAG.
Maneja la generación de respuestas contextuales usando modelos de OpenAI.
"""
import re
import random
from typing import List
from config.settings import logger
from config.constants import information_emojis
from models.openai_model import OpenAIModel
from utils.text_utils import strip_markdown_emphasis


class ResponseGenerator:
    """Generador de respuestas inteligentes con contexto específico."""
    
    def __init__(self, primary_model, fallback_model_name, openai_api_key, 
                 api_timeout, max_output_tokens):
        self.model = primary_model
        self.fallback_model_name = fallback_model_name
        self.openai_api_key = openai_api_key
        self.api_timeout = api_timeout
        self.max_output_tokens = max_output_tokens
        
        # Clasificadores de consultas para respuestas especializadas
        self.query_classifiers = {
            'denuncias': {
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'patterns': [
                    r'(como|cómo|de que forma|donde|dónde).*?(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                    r'(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE DENUNCIAS:
- Menciona EXPLÍCITAMENTE que las denuncias se presentan POR ESCRITO
- Indica claramente que debe incluir una relación circunstanciada de hechos y personas
- Menciona la alternativa de presentación verbal en casos de urgencia
- Indica que la denuncia verbal debe ratificarse por escrito dentro de las 48 horas
- Si hay artículos relevantes en la información, cítalos textualmente
- Asegúrate de mencionar que la Universidad puede iniciar sumarios de oficio
"""
            },
            'regimen_disciplinario': {
                'keywords': ['régimen disciplinario', 'regimen disciplinario', 'disciplina', 'sanción', 'sancion'],
                'patterns': [
                    r'(sancion|sanción|castigo|pena|amonestacion|expulsión|suspension).*?(estudiante|alumno)',
                    r'(que|qué).*?(sancion|sanción|castigo|pena).*?(corresponde|aplica)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE RÉGIMEN DISCIPLINARIO:
- Cita con precisión los artículos relevantes del Régimen Disciplinario
- Incluye los tipos de sanciones que pueden aplicarse y su graduación
- Menciona qué autoridades pueden aplicar cada tipo de sanción
- Si se mencionan plazos o procedimientos específicos, destácalos claramente
- Explica claramente qué derechos tiene el estudiante en un proceso disciplinario
"""
            },
            'regularidad': {
                'keywords': ['regularidad', 'regular', 'condiciones'],
                'patterns': [
                    r'(como|cómo).*?(mantener|conseguir|obtener|perder).*?(regularidad|condición|condicion)',
                    r'(requisito|requisitos).*?(alumno regular|regularidad|condición)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE REGULARIDAD:
- Destaca claramente el número mínimo de materias a aprobar en cada período
- Menciona el porcentaje máximo de aplazos permitidos
- Explica los plazos establecidos para completar la carrera
- Si hay excepciones o situaciones especiales, menciónalas
- Cita los artículos específicos sobre regularidad que sean relevantes
"""
            }
        }
    
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta contextualizada usando OpenAI.
        
        Args:
            query: La consulta del usuario
            context: Contexto mejorado de la consulta
            sources: Lista de fuentes consultadas
            
        Returns:
            Respuesta generada y formateada
        """
        emoji = random.choice(information_emojis)
        
        # Preparar fuentes
        sources_text = self._format_sources(sources)
        
        # Detectar tipo específico de consulta
        specific_instructions = self._get_specific_instructions(query)
        
        # Generar prompt
        prompt = self._build_prompt(query, context, sources_text, specific_instructions)
        
        # Generar respuesta con fallback
        response = self._generate_with_fallback(prompt, emoji)
        
        # Post-procesar respuesta
        response = self._post_process_response(response, query, emoji)
        
        # Validar calidad de respuesta
        self._validate_response_quality(response, query)
        
        return response
    
    def _format_sources(self, sources: List[str]) -> str:
        """Formatea la lista de fuentes."""
        if not sources:
            return ""
        
        unique_sources = list(set(sources))
        return f"\nFUENTES CONSULTADAS:\n{', '.join(unique_sources)}"
    
    def _get_specific_instructions(self, query: str) -> str:
        """Obtiene instrucciones específicas basadas en el tipo de consulta."""
        query_lower = query.lower()
        
        for category, data in self.query_classifiers.items():
            # Verificar keywords
            if any(keyword in query_lower for keyword in data['keywords']):
                return data['instructions']
            
            # Verificar patrones regex
            for pattern in data['patterns']:
                if re.search(pattern, query_lower):
                    return data['instructions']
        
        return ""
    
    def _build_prompt(self, query: str, context: str, sources_text: str, 
                     specific_instructions: str) -> str:
        """Construye el prompt para el modelo."""
        return f"""
Sos DrCecim, un asistente virtual especializado de la Facultad de Medicina UBA. Tu tarea es proporcionar respuestas sobre administración y trámites de la facultad y deben ser breves, precisas y útiles.

INFORMACIÓN RELEVANTE:
{context}
{sources_text}

CONSULTA ACTUAL: {query}
{specific_instructions}

RESPONDE SIGUIENDO ESTAS REGLAS:
1. Sé muy conciso y directo
2. Usa la información de los documentos oficiales proporcionados
3. Si hay artículos, resoluciones o reglamentos específicos, cita exactamente el número y fuente 
4. No omitas información importante de los documentos relevantes
5. Si hay documentos específicos, cita naturalmente su origen ("Según el reglamento...")
6. NO uses formato Markdown ya que esto no se procesa correctamente en mensajería
7. Para enfatizar texto, usa MAYÚSCULAS o comillas
8. Usa viñetas con guiones (-) cuando sea útil para organizar información
9. Si la información está incompleta, sugiere contactar a @cecim.nemed por instagram
10. No inventes o asumas información que no esté en los documentos
11. No hagas preguntas adicionales en tu respuesta
"""
    
    def _generate_with_fallback(self, prompt: str, emoji: str) -> str:
        """Genera respuesta con modelo principal y fallback si falla."""
        try:
            return self.model.generate(prompt)
        except Exception as e:
            logger.error(f"Error al generar respuesta con el modelo primario: {str(e)}")
            return self._generate_fallback_response(prompt, emoji)
    
    def _generate_fallback_response(self, prompt: str, emoji: str) -> str:
        """Genera respuesta usando modelo de fallback."""
        try:
            logger.info(f"Intentando con el modelo de respaldo: {self.fallback_model_name}")
            fallback_model = OpenAIModel(
                model_name=self.fallback_model_name,
                api_key=self.openai_api_key,
                timeout=self.api_timeout,
                max_output_tokens=self.max_output_tokens
            )
            return fallback_model.generate(prompt)
        except Exception as e2:
            logger.error(f"Error también con el modelo de respaldo: {str(e2)}")
            return f"{emoji} Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo o contacta a @cecim.nemed por instagram."
    
    def _post_process_response(self, response: str, query: str, emoji: str) -> str:
        """Post-procesa la respuesta generada."""
        # Limpiar formatos markdown
        response = strip_markdown_emphasis(response)
        
        # Agregar emoji si no tiene uno al inicio
        if not re.match(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]', 
                       response.strip()[:1]):
            response = f"{emoji} {response}"
        
        return response
    
    def _validate_response_quality(self, response: str, query: str):
        """Valida la calidad de la respuesta para tipos específicos de consultas."""
        query_lower = query.lower()
        response_lower = response.lower()
        
        for category, data in self.query_classifiers.items():
            is_category = (
                any(keyword in query_lower for keyword in data['keywords']) or
                any(re.search(pattern, query_lower) for pattern in data['patterns'])
            )
            
            if is_category:
                self._validate_category_response(category, response_lower, response)
    
    def _validate_category_response(self, category: str, response_lower: str, response: str):
        """Valida respuesta para categoría específica."""
        validations = {
            'denuncias': {
                'check': "por escrito" not in response_lower,
                'warning': "La respuesta sobre denuncias no incluye información sobre presentación por escrito",
                'append': "\n\nRECUERDA: Las denuncias DEBEN presentarse POR ESCRITO con todos los detalles relevantes."
            },
            'regimen_disciplinario': {
                'check': not any(keyword in response_lower for keyword in ['apercibimiento', 'suspensión', 'sanción']),
                'warning': "La respuesta sobre régimen disciplinario no menciona tipos de sanciones",
                'append': None
            },
            'regularidad': {
                'check': not any(term in response_lower for term in ['materias', 'aprobar', 'porcentaje', 'plazo']),
                'warning': "La respuesta sobre regularidad no incluye información sobre requisitos clave",
                'append': None
            }
        }
        
        validation = validations.get(category)
        if validation and validation['check']:
            logger.warning(validation['warning'])
            if validation['append']:
                response += validation['append']