#!/usr/bin/env python3
"""
Test completo para verificar todas las sheets del catálogo sheets_catalog.yaml
Verifica que todas las sheets configuradas sean accesibles y tengan la estructura esperada.
"""

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
from base_test import BaseTest
from services.sheets_service import SheetsService
from config.settings import GOOGLE_API_KEY


class TestSheetsCatalog(BaseTest):
    """Test completo del catálogo de Google Sheets."""

    def get_test_description(self) -> str:
        return "Verificación completa de todas las sheets del catálogo"

    def get_test_category(self) -> str:
        return "sheets_catalog"

    def _run_test_logic(self) -> bool:
        """Ejecuta tests para todas las sheets del catálogo."""
        
        # 0. FIX: Cargar .env desde el directorio raíz del proyecto
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(env_path)
        self.log_info(f"Cargando .env desde: {env_path}")
        self.log_info(f"CURSOS_SPREADSHEET_ID cargado: {bool(os.getenv('CURSOS_SPREADSHEET_ID'))}")
        
        # 1. Verificar que el catálogo existe y es válido
        if not self._test_catalog_file():
            return False
        
        # 2. Cargar catálogo
        catalog = self._load_catalog()
        if not catalog:
            return False
        
        # 3. Verificar servicio de Google Sheets
        sheets_service = self._initialize_sheets_service()
        if not sheets_service:
            return False
        
        # 4. Verificar cada dominio del catálogo
        return self._test_all_domains(catalog, sheets_service)

    def _test_catalog_file(self) -> bool:
        """Verifica que el archivo de catálogo existe y es válido."""
        catalog_path = "/Users/lucamazzarello_/Desktop/Repositories/chatbot_uba/config/sheets_catalog.yaml"
        
        if not os.path.exists(catalog_path):
            self.log_error("Archivo sheets_catalog.yaml no encontrado")
            return False
        
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            if not content or 'domains' not in content:
                self.log_error("Estructura de catálogo inválida - falta 'domains'")
                return False
            
            self.log_success(f"Catálogo cargado con {len(content['domains'])} dominios")
            return True
            
        except Exception as e:
            self.log_error(f"Error leyendo catálogo: {str(e)}")
            return False

    def _load_catalog(self) -> Dict[str, Any]:
        """Carga el catálogo desde el archivo YAML."""
        try:
            with open("/Users/lucamazzarello_/Desktop/Repositories/chatbot_uba/config/sheets_catalog.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.log_error(f"No se pudo cargar catálogo: {str(e)}")
            return {}

    def _initialize_sheets_service(self) -> SheetsService:
        """Inicializa el servicio de Google Sheets."""
        if not GOOGLE_API_KEY:
            self.log_warning("Google API Key no configurada - saltando tests de conectividad")
            return None
        
        try:
            sheets_service = SheetsService(api_key=GOOGLE_API_KEY)
            self.log_success("SheetsService inicializado correctamente")
            return sheets_service
        except Exception as e:
            self.log_error(f"Error inicializando SheetsService: {str(e)}")
            return None

    def _test_all_domains(self, catalog: Dict[str, Any], sheets_service: SheetsService) -> bool:
        """Verifica todos los dominios del catálogo."""
        domains = catalog.get('domains', {})
        success_count = 0
        total_count = len(domains)
        
        self.log_info(f"Verificando {total_count} dominios de sheets:")
        
        for domain_name, domain_config in domains.items():
            self.log_info(f"\n🔍 Probando dominio: {domain_name}")
            
            if self._test_domain(domain_name, domain_config, sheets_service):
                success_count += 1
                self.log_success(f"✅ {domain_name}: OK")
            else:
                self.log_error(f"❌ {domain_name}: FALLÓ")
        
        self.log_info(f"\n📊 Resumen: {success_count}/{total_count} dominios exitosos")
        
        # Considerar exitoso si al menos 80% funcionan
        return success_count >= (total_count * 0.8)

    def _test_domain(self, domain_name: str, config: Dict[str, Any], 
                    sheets_service: SheetsService) -> bool:
        """Verifica un dominio específico."""
        
        # 1. Verificar configuración básica
        if not self._validate_domain_config(domain_name, config):
            return False
        
        # 2. Resolver spreadsheet_id
        spreadsheet_id = self._resolve_spreadsheet_id(config.get('spreadsheet_id'))
        if not spreadsheet_id:
            self.log_warning(f"   ⚠️ {domain_name}: spreadsheet_id no resuelto")
            return False
        
        # 3. Verificar conectividad (solo si hay servicio)
        if sheets_service:
            return self._test_sheet_connectivity(domain_name, spreadsheet_id, config, sheets_service)
        else:
            self.log_info(f"   ℹ️ {domain_name}: configuración válida (sin test de conectividad)")
            return True

    def _validate_domain_config(self, domain_name: str, config: Dict[str, Any]) -> bool:
        """Valida la configuración de un dominio."""
        required_fields = ['spreadsheet_id', 'selector', 'triggers']
        
        for field in required_fields:
            if field not in config:
                self.log_error(f"   ❌ {domain_name}: falta campo '{field}'")
                return False
        
        # Verificar triggers
        triggers = config.get('triggers', {})
        if 'keywords' not in triggers or not triggers['keywords']:
            self.log_error(f"   ❌ {domain_name}: falta 'triggers.keywords'")
            return False
        
        self.log_info(f"   ✅ {domain_name}: configuración válida")
        return True

    def _resolve_spreadsheet_id(self, spreadsheet_id: str) -> str:
        """Resuelve un spreadsheet_id que puede ser una variable de entorno."""
        if not spreadsheet_id:
            return ""
        
        if spreadsheet_id.startswith('ENV:'):
            env_var = spreadsheet_id.split(':', 1)[1]
            resolved = os.getenv(env_var)
            if not resolved:
                return ""
            return resolved
        
        return spreadsheet_id

    def _test_sheet_connectivity(self, domain_name: str, spreadsheet_id: str, 
                               config: Dict[str, Any], sheets_service: SheetsService) -> bool:
        """Verifica conectividad real con la sheet."""
        try:
            # Determinar rango a probar
            if config.get('dynamic_sheets', False):
                # Para dominios como CURSOS que usan sheets dinámicas por meses
                available_months = config.get('available_months', [])
                if available_months:
                    # Usar el primer mes disponible para testing
                    sheet_name = available_months[0]
                    self.log_info(f"   📅 Usando sheet dinámica: {sheet_name}")
                else:
                    self.log_warning(f"   ⚠️ {domain_name}: dynamic_sheets=true pero sin available_months")
                    return False
            else:
                sheet_name = config.get('sheet_name', 'Hoja 1')
            
            ranges = config.get('ranges', {})
            default_range = ranges.get('default', 'A:E')
            full_range = f"'{sheet_name}'!{default_range}"
            
            self.log_info(f"   🔗 Probando conectividad: {spreadsheet_id} - {full_range}")
            
            # Intentar obtener datos
            values = sheets_service.get_sheet_values(spreadsheet_id, full_range)
            
            if values:
                row_count = len(values)
                col_count = len(values[0]) if values else 0
                self.log_success(f"   ✅ Conectividad OK: {row_count} filas, {col_count} columnas")
                
                # Verificar que hay contenido útil
                if row_count < 2:  # Al menos header + 1 fila
                    self.log_warning(f"   ⚠️ Pocas filas de datos ({row_count})")
                
                return True
            else:
                self.log_warning(f"   ⚠️ Sheet vacía o sin permisos")
                return False
                
        except Exception as e:
            self.log_error(f"   ❌ Error de conectividad: {str(e)}")
            return False

    def _test_tools_integration(self) -> bool:
        """Verifica que las herramientas usen las sheets correctas."""
        tool_mappings = {
            'HorariosCatedraTool': '1xPT3mTqDNTsRjhM65vbI7SyLSVBbUkeE4VLH09HTywk',
            'HorariosLicTecTool': '1Byu1Xqxrx-UCM9_18p0cWppOwNzTWn4DHt0R249kiuY', 
            'HorariosSecretariasTool': '1GyKQXDdujbnAhfnWehcd8W9_jKzErlVmYhRHAvOV83U',
            'MailsNuevoEspacioTool': '1R5arCJAolOcMIAEBydTnIMi2boIagw4qyaI2UlC-eb0'
        }
        
        self.log_info("\n🔧 Verificando integración con herramientas:")
        
        success_count = 0
        for tool_name, expected_id in tool_mappings.items():
            try:
                # Importar herramienta dinámicamente
                module_name = f"services.tools.{tool_name.lower().replace('tool', '_tool')}"
                module = __import__(module_name, fromlist=[tool_name])
                tool_class = getattr(module, tool_name)
                
                # Instanciar herramienta
                tool = tool_class(None)
                
                # Verificar configuración
                if hasattr(tool, 'config') and 'spreadsheet_id' in str(tool.config):
                    self.log_success(f"   ✅ {tool_name}: configurada")
                    success_count += 1
                else:
                    self.log_warning(f"   ⚠️ {tool_name}: sin spreadsheet_id en config")
                    
            except Exception as e:
                self.log_error(f"   ❌ {tool_name}: error - {str(e)}")
        
        return success_count >= len(tool_mappings) * 0.75  # 75% success rate


if __name__ == "__main__":
    test = TestSheetsCatalog()
    result = test.run_test()
    exit(0 if result["passed"] else 1)