from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import MySQLDatabase
from recommender import Recommender
from trainer import RecommendationTrainer
import uvicorn
from config import Config

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = MySQLDatabase()
recommender = Recommender()

@app.get("/recommend/user/{user_id}")
async def user_recommendations(user_id: int, limit: int = 5):
    try:
        recommendations = recommender.recommend_for_user(user_id, limit)
        if recommendations is None:
            # New user - return popular items
            recommendations = db.get_popular_items(limit)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/similar/{item_id}")
async def similar_items(item_id: int, limit: int = 5):
    try:
        similar = recommender.similar_items(item_id, limit)
        if similar is None:
            similar = db.get_popular_items(limit)
        return {"item_id": item_id, "similar_items": similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model():
    try:
        trainer = RecommendationTrainer(db)
        history = trainer.train()
        return {"status": "success", "epochs": len(history.history['loss'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)