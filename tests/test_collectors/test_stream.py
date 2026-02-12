import pytest
import asyncio
from src.collectors import stream
from src.config import settings

@pytest.mark.asyncio
async def test_stream_collector_import_and_config():
    """
    Testa a configuração inicial e importação do módulo de stream.

    Este teste serve como um 'Smoke Test' (Teste de Fumaça) para garantir que:
    1. O módulo `src.collectors.stream` pode ser importado sem erros de sintaxe ou dependências faltando.
    2. As configurações do ambiente (`settings.ENV`) estão carregadas corretamente.
    
    Se este teste falhar, indica problemas fundamentais na configuração do projeto ou ambiente virtual.
    """
    assert stream is not None
    assert settings.ENV in ["development", "production", "testing"]
    print(f"\n[Smoke Test] Environment: {settings.ENV}")