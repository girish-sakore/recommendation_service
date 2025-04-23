import os

class Settings:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "foodclub_production")
    DB_USER = os.getenv("DB_USER", "mysqldeveloper")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "Password@123")
    
    MODEL_PATH = os.getenv("MODEL_PATH", "saved_models")
    EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 50))
    TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", 10))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

settings = Settings()
