from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    compressed_filepath: str = '../lichess_db_eval.jsonl.zst'
    output_filepath: str = '../lichess_db_eval.jsonl'
    max_eval: int = 15
    line_limit: int = 5000000
    # flake8: noqa
    svg_base_url: str = 'https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/' 
    train_size: float = 0.85
    val_size: float = 0.10


data_settings = DataSettings()
