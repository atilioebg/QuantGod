import os
import sys
from pathlib import Path
import textwrap

def create_file(path: Path, content: str = ""):
    """Create a file with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

def setup_project():
    """Sets up the DeepSwing project structure."""
    
    # Base directory (Current Working Directory)
    base_dir = Path.cwd()
    print(f"Setting up project in: {base_dir}")

    # --- Configuration Content ---
    
    gitignore_content = textwrap.dedent("""
        # Python
        __pycache__/
        *.py[cod]
        *$py.class
        
        # Environments
        .env
        .venv/
        venv/
        env/
        
        # Data
        data/
        !data/.gitkeep
        
        # IDE / Editor
        .vscode/
        .idea/
        *.swp
        
        # Logs
        *.log
    """).strip()

    pytest_ini_content = textwrap.dedent("""
        [pytest]
        pythonpath = src
        asyncio_mode = auto
        testpaths = tests
        addopts = -v
    """).strip()

    requirements_content = textwrap.dedent("""
        polars>=0.20.0
        pytest>=8.0.0
        pytest-asyncio>=0.23.0
        python-dotenv>=1.0.0
        pydantic>=2.6.0
        torch>=2.2.0
        websockets>=12.0
        aiohttp>=3.9.0
        pyarrow>=15.0.0
    """).strip()

    readme_content = textwrap.dedent("""
        # DeepSwing
        
        Crypto Swing Trade AI focusing on TDD and Data Engineering.
        
        ## Setup
        1. Create virtual environment: `python -m venv .venv`
        2. Activate: `. .venv/bin/activate` (Linux/Mac) or `.\\.venv\\Scripts\\Activate` (Windows)
        3. Install deps: `pip install -r requirements.txt`
        4. Run tests: `pytest`
    """).strip()

    env_example_content = textwrap.dedent("""
        # API Keys
        EXCHANGE_API_KEY=your_key_here
        EXCHANGE_SECRET=your_secret_here
        
        # Config
        ENV=development
    """).strip()

    # --- Source Code Content ---

    src_config_content = textwrap.dedent("""
        import os
        from pydantic import BaseSettings
        
        class Settings(BaseSettings):
            ENV: str = "development"
            
            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"

        settings = Settings()
    """).strip()
    
    # Smoke test content
    test_stream_content = textwrap.dedent("""
        import pytest
        import asyncio
        from src.collectors import stream
        from src.config import settings

        @pytest.mark.asyncio
        async def test_stream_collector_import_and_config():
            \"\"\"
            Smoke test to verify:
            1. 'src.collectors.stream' can be imported.
            2. Environment config is accessible.
            \"\"\"
            assert stream is not None
            assert settings.ENV in ["development", "production", "testing"]
            print(f"\\n[Smoke Test] Environment: {settings.ENV}")
    """).strip()

    # --- Directory Tree Definition ---
    # format: (path, content) - content is optional
    
    files_to_create = [
        # Root configs
        (".env.example", env_example_content),
        (".gitignore", gitignore_content),
        ("pytest.ini", pytest_ini_content),
        ("requirements.txt", requirements_content),
        ("README.md", readme_content),
        
        # Notebooks
        ("notebooks/.gitkeep", ""),
        
        # Data (Directories created implicitly by file path, but we want empty dirs)
        # We'll create .gitkeep files to enforce directory creation
        ("data/raw/historical/.gitkeep", ""),
        ("data/raw/stream/.gitkeep", ""),
        ("data/processed/.gitkeep", ""),
        
        # Source - Collectors
        ("src/__init__.py", ""),
        ("src/config.py", src_config_content),
        ("src/collectors/__init__.py", ""),
        ("src/collectors/historical.py", "# Historical data collector"),
        ("src/collectors/stream.py", "# Real-time data collector via WebSockets"),
        
        # Source - Processing
        ("src/processing/__init__.py", ""),
        ("src/processing/normalization.py", "# Data normalization logic"),
        ("src/processing/labeling.py", "# Data labeling logic"),
        
        # Source - Models
        ("src/models/__init__.py", ""),
        ("src/models/vivit.py", "# ViViT (Video Vision Transformer) Model Architecture"),
        
        # Source - Utils
        ("src/utils/__init__.py", ""),
        ("src/utils/logger.py", "# Logging configuration"),
        
        # Tests
        ("tests/__init__.py", ""),
        ("tests/conftest.py", "# Global fixtures"),
        ("tests/test_collectors/__init__.py", ""),
        ("tests/test_collectors/test_stream.py", test_stream_content),
        ("tests/test_processing/__init__.py", ""),
        ("tests/test_processing/test_normalization.py", "def test_normalization():\n    assert True"),
    ]

    print("Starting scaffolding...")
    
    for path_str, content in files_to_create:
        create_file(base_dir / path_str, content)
        
    print("\\nDeepSwing Project Structure Created Successfully! 🚀")

if __name__ == "__main__":
    setup_project()
