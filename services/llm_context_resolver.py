"""
Resolvedor de contexto conversacional usando LLM.
Interpreta consultas relativas de forma flexible sin patrones hardcodeados.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from config.settings import config
import openai

logger = logging.getLogger(__name__)


@dataclass
class ContextResolution:
    """Resultado de la resolución de contexto por LLM."""
    is_relative: bool = False
    resolved_context: Optional[str] = None
    context_type: str = ""  # "month", "week", "time_period"
    offset: int = 0
    explanation: str = ""
    confidence: float = 0.0


class LLMContextResolver:
    """Resolvedor de contexto conversacional usando LLM."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 200):
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=config.openai.openai_api_key)
    
    def resolve_relative_query(self, current_query: str, session_context: Dict[str, Any]) -> ContextResolution:
        """
        Resuelve una consulta potencialmente relativa usando el contexto de la sesión.
        
        Args:
            current_query: La consulta actual del usuario
            session_context: Contexto de la sesión (última consulta, tipo, etc.)
            
        Returns:
            ContextResolution: Resultado de la interpretación
        """
        try:
            prompt = self._build_context_prompt(current_query, session_context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.1  # Baja temperatura para consistencia
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            logger.error(f"Error en resolución LLM de contexto: {e}")
            return ContextResolution(explanation=f"Error: {str(e)}")
    
    def _build_context_prompt(self, current_query: str, session_context: Dict[str, Any]) -> str:
        """Construye el prompt para el LLM basado en el contexto."""
        
        last_query = session_context.get('last_query', '')
        last_query_type = session_context.get('last_query_type', '')
        last_month = session_context.get('last_month_requested', '')
        last_time_ref = session_context.get('last_time_reference', '')
        last_subject = session_context.get('last_subject_requested', '')
        last_teacher = session_context.get('last_teacher_requested', '')
        last_procedure = session_context.get('last_procedure_requested', '')
        last_resource = session_context.get('last_resource_requested', '')
        last_department = session_context.get('last_department_requested', '')
        
        prompt = f"""Eres un asistente que interpreta consultas conversacionales en español para un chatbot universitario.

CONTEXTO DE LA CONVERSACIÓN:
- Última consulta: "{last_query}"
- Tipo de consulta anterior: "{last_query_type}"
- Mes mencionado anteriormente: "{last_month}"
- Referencia temporal anterior: "{last_time_ref}"
- Materia mencionada anteriormente: "{last_subject}"
- Docente mencionado anteriormente: "{last_teacher}"
- Trámite mencionado anteriormente: "{last_procedure}"
- Recurso mencionado anteriormente: "{last_resource}"
- Departamento mencionado anteriormente: "{last_department}"

CONSULTA ACTUAL: "{current_query}"

TAREA:
Determina si la consulta actual es una referencia relativa a la consulta anterior. Las consultas relativas típicamente usan frases como:
- "y el/la que sigue", "y el/la siguiente", "y el/la anterior"
- "y el mes que viene", "y la semana pasada" 
- "¿y después?", "¿y antes?"
- "y los requisitos", "y el programa", "y la bibliografía"
- "y sus materias", "y su horario", "y su consultorio"
- "y el formulario", "y los plazos", "y el costo"
- "y los horarios", "y los servicios", "y la ubicación"
- "y qué documentación", "y cómo hacerlo", "y los pasos"
- "y qué servicios", "y dónde está", "y cómo llegar"

Si ES una consulta relativa, resuelve a qué se refiere específicamente.

EJEMPLOS:
- Si la consulta anterior fue sobre "cursos de AGOSTO" y la actual es "¿y el que sigue?", se refiere a "SEPTIEMBRE"
- Si la consulta anterior fue sobre "eventos esta semana" y la actual es "¿y la que sigue?", se refiere a "la próxima semana"
- Si la consulta anterior fue sobre "Anatomía" y la actual es "¿y los requisitos?", se refiere a "requisitos de Anatomía"
- Si la consulta anterior fue sobre "Dr. García" y la actual es "¿y sus materias?", se refiere a "materias del Dr. García"
- Si la consulta anterior fue sobre "inscripción" y la actual es "¿y los plazos?", se refiere a "plazos de inscripción"
- Si la consulta anterior fue sobre "biblioteca" y la actual es "¿y los horarios?", se refiere a "horarios de biblioteca"
- Si la consulta anterior fue sobre "inscripción a materias" y la actual es "¿y qué documentación necesito?", se refiere a "documentación para inscripción a materias"
- Si la consulta anterior fue sobre "biblioteca" y la actual es "¿y qué servicios ofrece?", se refiere a "servicios de biblioteca"
- Si la consulta anterior fue sobre "Dr. García" y la actual es "¿y su horario de consultas?", se refiere a "horario de consultas del Dr. García"
- Si la consulta anterior fue sobre "secretaría de ciclo clínico" y la actual es "y donde se encuentra?", se refiere a "ubicación de secretaría de ciclo clínico"
- Si la consulta anterior fue sobre "hospital durand" y la actual es "y cual es el contacto?", se refiere a "contacto del hospital durand"
- Si la consulta anterior fue sobre "cursos de octubre" y la actual es "y para noviembre?", se refiere a "cursos de noviembre"

RESPONDE EXACTAMENTE en formato JSON:
{{
    "is_relative": true/false,
    "resolved_context": "contexto resuelto específico o null",
    "context_type": "month/week/time_period/subject/teacher/procedure/library/none",
    "offset": número entero (1 = siguiente, -1 = anterior, 0 = mismo),
    "explanation": "explicación breve de la interpretación",
    "confidence": número decimal entre 0.0 y 1.0
}}"""

        return prompt
    
    def _parse_llm_response(self, response_text: str) -> ContextResolution:
        """Parsea la respuesta JSON del LLM."""
        try:
            # Limpiar la respuesta por si tiene texto extra
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(response_text)
            
            return ContextResolution(
                is_relative=data.get('is_relative', False),
                resolved_context=data.get('resolved_context'),
                context_type=data.get('context_type', ''),
                offset=data.get('offset', 0),
                explanation=data.get('explanation', ''),
                confidence=data.get('confidence', 0.0)
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Error parseando respuesta LLM: {e}. Respuesta: {response_text}")
            return ContextResolution(
                explanation=f"Error parseando respuesta: {response_text[:100]}..."
            )
        except Exception as e:
            logger.error(f"Error inesperado parseando respuesta LLM: {e}")
            return ContextResolution(explanation=f"Error inesperado: {str(e)}")


class HybridContextResolver:
    """
    Resolvedor híbrido que combina patrones rápidos con LLM para casos complejos.
    Optimiza velocidad y costo.
    """
    
    def __init__(self, llm_resolver: Optional[LLMContextResolver] = None):
        self.llm_resolver = llm_resolver or LLMContextResolver()
        
        # Patrones extendidos para múltiples tipos de consultas
        self.quick_patterns = {
            # Patrones temporales (existentes + variaciones)
            "y el siguiente": {"type": "forward", "offset": 1},
            "y el anterior": {"type": "backward", "offset": -1},
            "y la siguiente": {"type": "forward", "offset": 1},
            "y la anterior": {"type": "backward", "offset": -1},
            "y el que sigue": {"type": "forward", "offset": 1},
            "y el que viene": {"type": "forward", "offset": 1},
            "y el que viene después": {"type": "forward", "offset": 1},
            "y el mes que viene": {"type": "forward", "offset": 1},
            "y el próximo": {"type": "forward", "offset": 1},
            "y el próximo mes": {"type": "forward", "offset": 1},
            "y el que pasó": {"type": "backward", "offset": -1},
            "y el anterior": {"type": "backward", "offset": -1},
            "y el mes pasado": {"type": "backward", "offset": -1},
            "y la próxima semana": {"type": "forward", "offset": 1},
            "y la semana que viene": {"type": "forward", "offset": 1},
            "y la semana pasada": {"type": "backward", "offset": -1},
            
            # Patrones para materias
            "y esa materia": {"type": "same_subject", "offset": 0},
            "y otra materia": {"type": "other_subject", "offset": 0},
            "y las materias relacionadas": {"type": "related_subjects", "offset": 0},
            "y el programa": {"type": "subject_program", "offset": 0},
            "y la bibliografía": {"type": "subject_bibliography", "offset": 0},
            "y el horario": {"type": "subject_schedule", "offset": 0},
            "y la cursada": {"type": "subject_course", "offset": 0},
            "y los requisitos de la materia": {"type": "subject_requirements", "offset": 0},
            "y los requisitos para cursar": {"type": "subject_requirements", "offset": 0},
            "y los requisitos": {"type": "subject_requirements", "offset": 0},
            "y los requisitos de cursar": {"type": "subject_requirements", "offset": 0},
            "y el programa de la materia": {"type": "subject_program", "offset": 0},
            "y la bibliografía de la materia": {"type": "subject_bibliography", "offset": 0},
            "y el horario de la materia": {"type": "subject_schedule", "offset": 0},
            
            # Patrones para docentes
            "y sus materias": {"type": "teacher_subjects", "offset": 0},
            "y su horario": {"type": "teacher_schedule", "offset": 0},
            "y su consultorio": {"type": "teacher_office", "offset": 0},
            "y su email": {"type": "teacher_contact", "offset": 0},
            "y su información": {"type": "teacher_info", "offset": 0},
            "y las materias que dicta": {"type": "teacher_subjects", "offset": 0},
            "y el horario de consultas": {"type": "teacher_schedule", "offset": 0},
            "y su horario de consultas": {"type": "teacher_schedule", "offset": 0},
            "y su contacto": {"type": "teacher_contact", "offset": 0},
            
            # Patrones para trámites
            "y los requisitos": {"type": "procedure_requirements", "offset": 0},
            "y el formulario": {"type": "procedure_form", "offset": 0},
            "y los plazos": {"type": "procedure_deadlines", "offset": 0},
            "y el costo": {"type": "procedure_cost", "offset": 0},
            "y el procedimiento": {"type": "procedure_steps", "offset": 0},
            "y la documentación": {"type": "procedure_docs", "offset": 0},
            "y qué documentación": {"type": "procedure_docs", "offset": 0},
            "y qué documentación necesito": {"type": "procedure_docs", "offset": 0},
            "y los pasos": {"type": "procedure_steps", "offset": 0},
            "y cómo hacerlo": {"type": "procedure_steps", "offset": 0},
            
            # Patrones para biblioteca
            "y los horarios": {"type": "library_hours", "offset": 0},
            "y los servicios": {"type": "library_services", "offset": 0},
            "y la ubicación": {"type": "library_location", "offset": 0},
            "y el catálogo": {"type": "library_catalog", "offset": 0},
            "y qué servicios": {"type": "library_services", "offset": 0},
            "y qué servicios ofrece": {"type": "library_services", "offset": 0},
            "y dónde está": {"type": "library_location", "offset": 0},
            "y cómo llegar": {"type": "library_location", "offset": 0},
            
            # Patrones para consultas administrativas argentinas
            "y donde se encuentra": {"type": "admin_location", "offset": 0},
            "y en que horario": {"type": "admin_hours", "offset": 0},
            "y en que horario de atención": {"type": "admin_hours", "offset": 0},
            "y cual es el contacto": {"type": "admin_contact", "offset": 0},
            "y el email": {"type": "admin_email", "offset": 0},
            "y el mail": {"type": "admin_email", "offset": 0},
            "y el telefono": {"type": "admin_phone", "offset": 0},
            "y el teléfono": {"type": "admin_phone", "offset": 0},
            "y la dirección": {"type": "admin_address", "offset": 0},
            "y la direccion": {"type": "admin_address", "offset": 0},
            "y para noviembre": {"type": "forward", "offset": 1},
            "y para septiembre": {"type": "backward", "offset": -1},
            "y el que sigue después": {"type": "forward", "offset": 1},
            "y el que sigue": {"type": "forward", "offset": 1},
        }
    
    def resolve_relative_query(self, current_query: str, session_context: Dict[str, Any]) -> ContextResolution:
        """
        Resuelve consultas relativas usando un enfoque híbrido.
        
        1. Primero intenta patrones rápidos
        2. Si no encuentra coincidencia, usa LLM
        """
        query_lower = current_query.lower().strip()
        
        # PASO 1: Verificar patrones rápidos básicos
        for pattern, config in self.quick_patterns.items():
            if pattern in query_lower:
                return self._resolve_with_quick_pattern(config, session_context)
        
        # PASO 2: Si no hay coincidencia rápida, preguntar al LLM
        # Solo si la consulta parece relativa (contiene palabras clave específicas)
        relative_keywords = ['siguiente', 'anterior', 'sigue', 'viene', 'después', 'antes', 'requisitos', 'programa', 'bibliografía', 'horario', 'materias', 'consultorio', 'formulario', 'plazos', 'costo', 'documentación', 'servicios', 'ubicación', 'catálogo', 'pasos', 'cómo', 'qué', 'dónde']
        if any(keyword in query_lower for keyword in relative_keywords):
            logger.info(f"Usando LLM para resolver consulta relativa: '{current_query}'")
            return self.llm_resolver.resolve_relative_query(current_query, session_context)
        
        # PASO 3: No es relativa
        return ContextResolution(explanation="No es una consulta relativa")
    
    def _resolve_with_quick_pattern(self, pattern_config: Dict, session_context: Dict[str, Any]) -> ContextResolution:
        """Resuelve usando un patrón rápido identificado."""
        
        last_query_type = session_context.get('last_query_type', '')
        pattern_type = pattern_config['type']
        offset = pattern_config['offset']
        
        # Para cursos (temporal)
        if last_query_type == 'cursos' and session_context.get('last_month_requested'):
            if pattern_type in ['forward', 'backward']:
                resolved_month = self._calculate_month_offset(
                    session_context['last_month_requested'], 
                    offset
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_month,
                    context_type="month",
                    offset=offset,
                    explanation=f"Patrón rápido: {session_context['last_month_requested']} -> {resolved_month}",
                    confidence=0.9
                )
        
        # Para calendario (temporal)
        elif last_query_type.startswith('calendario') and session_context.get('last_time_reference'):
            if pattern_type in ['forward', 'backward']:
                resolved_time = self._calculate_time_offset(
                    session_context['last_time_reference'],
                    offset
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_time,
                    context_type="week" if "semana" in resolved_time else "time_period",
                    offset=offset,
                    explanation=f"Patrón rápido: {session_context['last_time_reference']} -> {resolved_time}",
                    confidence=0.9
                )
        
        # Para materias
        elif last_query_type == 'materia' and session_context.get('last_subject_requested'):
            if pattern_type.startswith('subject_'):
                resolved_context = self._resolve_subject_pattern(
                    pattern_type, session_context['last_subject_requested']
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_context,
                    context_type="subject",
                    offset=0,
                    explanation=f"Patrón rápido: {pattern_type} para {session_context['last_subject_requested']}",
                    confidence=0.9
                )
        
        # Para docentes
        elif last_query_type == 'docente' and session_context.get('last_teacher_requested'):
            if pattern_type.startswith('teacher_'):
                resolved_context = self._resolve_teacher_pattern(
                    pattern_type, session_context['last_teacher_requested']
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_context,
                    context_type="teacher",
                    offset=0,
                    explanation=f"Patrón rápido: {pattern_type} para {session_context['last_teacher_requested']}",
                    confidence=0.9
                )
        
        # Para trámites
        elif last_query_type == 'tramite' and session_context.get('last_procedure_requested'):
            if pattern_type.startswith('procedure_'):
                resolved_context = self._resolve_procedure_pattern(
                    pattern_type, session_context['last_procedure_requested']
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_context,
                    context_type="procedure",
                    offset=0,
                    explanation=f"Patrón rápido: {pattern_type} para {session_context['last_procedure_requested']}",
                    confidence=0.9
                )
        
        # Para biblioteca
        elif last_query_type == 'biblioteca' and session_context.get('last_resource_requested'):
            if pattern_type.startswith('library_'):
                resolved_context = self._resolve_library_pattern(
                    pattern_type, session_context['last_resource_requested']
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_context,
                    context_type="library",
                    offset=0,
                    explanation=f"Patrón rápido: {pattern_type} para {session_context['last_resource_requested']}",
                    confidence=0.9
                )
        
        # Para consultas administrativas
        elif last_query_type == 'consulta_administrativa' and session_context.get('last_department_requested'):
            if pattern_type.startswith('admin_'):
                resolved_context = self._resolve_admin_pattern(
                    pattern_type, session_context['last_department_requested']
                )
                return ContextResolution(
                    is_relative=True,
                    resolved_context=resolved_context,
                    context_type="admin",
                    offset=0,
                    explanation=f"Patrón rápido: {pattern_type} para {session_context['last_department_requested']}",
                    confidence=0.9
                )
        
        return ContextResolution(explanation="Patrón detectado pero contexto insuficiente")
    
    def _calculate_month_offset(self, current_month: str, offset: int) -> str:
        """Calcula el mes con offset."""
        months = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
                  'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
        
        try:
            current_idx = months.index(current_month)
            new_idx = (current_idx + offset) % 12
            return months[new_idx]
        except ValueError:
            return current_month
    
    def _calculate_time_offset(self, current_time_ref: str, offset: int) -> str:
        """Calcula la referencia temporal con offset."""
        if "semana" in current_time_ref.lower():
            if offset == 1:
                return "la próxima semana"
            elif offset == -1:
                return "la semana pasada"
        elif "mes" in current_time_ref.lower():
            if offset == 1:
                return "el próximo mes"
            elif offset == -1:
                return "el mes pasado"
        
        return f"offset {offset} desde {current_time_ref}"
    
    def _resolve_subject_pattern(self, pattern_type: str, subject: str) -> str:
        """Resuelve patrones relacionados con materias."""
        pattern_mapping = {
            "same_subject": f"información sobre {subject}",
            "other_subject": f"otras materias relacionadas con {subject}",
            "related_subjects": f"materias relacionadas con {subject}",
            "subject_requirements": f"requisitos de {subject}",
            "subject_program": f"programa de {subject}",
            "subject_bibliography": f"bibliografía de {subject}",
            "subject_schedule": f"horarios de {subject}",
            "subject_course": f"cursada de {subject}",
        }
        return pattern_mapping.get(pattern_type, f"información sobre {subject}")
    
    def _resolve_teacher_pattern(self, pattern_type: str, teacher: str) -> str:
        """Resuelve patrones relacionados con docentes."""
        pattern_mapping = {
            "teacher_subjects": f"materias que dicta {teacher}",
            "teacher_schedule": f"horarios de {teacher}",
            "teacher_office": f"consultorio de {teacher}",
            "teacher_contact": f"contacto de {teacher}",
            "teacher_info": f"información sobre {teacher}",
        }
        return pattern_mapping.get(pattern_type, f"información sobre {teacher}")
    
    def _resolve_procedure_pattern(self, pattern_type: str, procedure: str) -> str:
        """Resuelve patrones relacionados con trámites."""
        pattern_mapping = {
            "procedure_requirements": f"requisitos para {procedure}",
            "procedure_form": f"formulario de {procedure}",
            "procedure_deadlines": f"plazos de {procedure}",
            "procedure_cost": f"costo de {procedure}",
            "procedure_steps": f"procedimiento de {procedure}",
            "procedure_docs": f"documentación para {procedure}",
        }
        return pattern_mapping.get(pattern_type, f"información sobre {procedure}")
    
    def _resolve_library_pattern(self, pattern_type: str, resource: str) -> str:
        """Resuelve patrones relacionados con biblioteca."""
        pattern_mapping = {
            "library_hours": f"horarios de {resource}",
            "library_services": f"servicios de {resource}",
            "library_location": f"ubicación de {resource}",
            "library_catalog": f"catálogo de {resource}",
        }
        return pattern_mapping.get(pattern_type, f"información sobre {resource}")
    
    def _resolve_admin_pattern(self, pattern_type: str, department: str) -> str:
        """Resuelve patrones relacionados con consultas administrativas."""
        pattern_mapping = {
            "admin_location": f"ubicación de {department}",
            "admin_hours": f"horarios de atención de {department}",
            "admin_contact": f"contacto de {department}",
            "admin_email": f"email de {department}",
            "admin_phone": f"teléfono de {department}",
            "admin_address": f"dirección de {department}",
        }
        return pattern_mapping.get(pattern_type, f"información sobre {department}")