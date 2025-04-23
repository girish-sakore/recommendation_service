import os

class Config:
    # MySQL Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "33061")
    DB_NAME = os.getenv("DB_NAME", "food_clube_production")
    DB_USER = os.getenv("DB_USER", "mysqldeveloper")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "Password@123")
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "saved_models")
    EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 50))
    TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", 10))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

config = Config()