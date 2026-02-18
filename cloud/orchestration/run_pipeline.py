import yaml
import logging
from pathlib import Path
import pandas as pd
from cloud.etl.extract import DataExtractor
from cloud.etl.transform import L2Transformer
from cloud.etl.load import DataLoader
from cloud.etl.validate import DataValidator
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    # 1. Load Config
    config_path = Path("cloud/configs/cloud_config.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize Modules
    extractor = DataExtractor(config['paths']['rclone_mount'])
    transformer = L2Transformer(
        levels=config['etl']['orderbook_levels'],
        sampling_ms=config['etl']['sampling_interval_ms']
    )
    loader = DataLoader(config['paths']['processed_output'])
    validator = DataValidator()

    # 3. Execution
    zip_files = extractor.list_zips()
    if not zip_files:
        logger.error("No data to process.")
        return

    all_processed_dfs = []

    for zip_path in tqdm(zip_files, desc="Processing ZIPs"):
        transformer.reset_book()
        sampled_rows = []
        
        for name, file_obj in extractor.stream_zip_content(zip_path):
            # We assume one file per zip as per legacy logic
            # If it's a large file, we iterate line by line to save RAM
            for line in file_obj:
                if not line: continue
                try:
                    msg = json.loads(line)
                    row = transformer.process_message(msg)
                    if row:
                        sampled_rows.append(row)
                except Exception as e:
                    continue
        
        if sampled_rows:
            df_sampled = pd.DataFrame(sampled_rows)
            # Feature Engineering & Resampling
            df_final = transformer.apply_feature_engineering(df_sampled)
            
            # Normalization (Z-Score) if enabled
            if config['features']['apply_zscore']:
                df_final = transformer.apply_zscore(df_final, config['paths']['scaler_path'])
            
            # Validation
            validator.validate_integrity(df_final, name=zip_path.name)
            
            # Load (per ZIP to avoid large memory accumulation, but we can also concat)
            output_name = zip_path.with_suffix(".parquet").name
            loader.save_parquet(df_final, output_name, config['etl']['compression'])
            
            all_processed_dfs.append(df_final)

    logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    run_pipeline()
