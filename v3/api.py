from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import MySQLDatabase
from trainer import RecommendationTrainer
import os
import uvicorn
from config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database connection
db = MySQLDatabase()
app.state.db = db
app.state.recommender = None

def load_recommender():
    try:
        logger.info("Importing Recommender...")
        from recommender import Recommender
        logger.info("Instantiating Recommender with database...")
        # Pass the database instance to Recommender
        app.state.recommender = Recommender(db)
        logger.info("Recommender loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load recommender: {str(e)}", exc_info=True)
        app.state.recommender = None
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Checking if model exists...")
    if not os.path.exists(f"{Config.MODEL_PATH}/model.h5"):
        logger.info("No trained model found. Training...")
        await train_model()

    logger.info("Loading recommender...")
    loaded = load_recommender()
    logger.info(f"Recommender loaded: {loaded}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    if hasattr(app.state, 'db') and app.state.db:
        app.state.db.close()

@app.get("/recommend/user/{user_id}")
async def user_recommendations(user_id: int, cart_item_id: int, limit: int = 3):
    logger.info(f"Getting recommendations for user {user_id} with item {cart_item_id}")
    
    if app.state.recommender is None:
        logger.error("Recommender not loaded")
        raise HTTPException(status_code=500, detail="Recommender not loaded.")
    
    try:
        recommendations = app.state.recommender.recommend_for_cart(user_id, cart_item_id, limit)
        
        if not recommendations:
            logger.info(f"No personalized recommendations, falling back to popular items")
            recommendations = app.state.db.get_popular_items(limit)
            
        return {
            "user_id": user_id,
            "cart_item_id": cart_item_id,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/similar/{item_id}")
async def similar_items(item_id: int, limit: int = 5):
    if app.state.recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not loaded.")
    
    try:
        similar = app.state.recommender.similar_items(item_id, limit)
        if not similar:
            similar = app.state.db.get_popular_items(limit)
        return {"item_id": item_id, "similar_items": similar}
    except Exception as e:
        logger.error(f"Similar items error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model():
    try:
        logger.info("Starting model training...")
        trainer = RecommendationTrainer(app.state.db)
        history = trainer.train()
        
        # Reload recommender after training
        load_recommender()
        
        return {
            "status": "success",
            "epochs": len(history.history['loss']),
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1]
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "recommender_loaded": app.state.recommender is not None,
        "database_connected": app.state.db.connection.is_connected() if hasattr(app.state, 'db') else False
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )