from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.database import Database
from models.recommender import Recommender
from models.trainer import RecommendationTrainer
import uvicorn
from config.settings import settings

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
db = Database()
recommender = Recommender()

@app.get("/recommend/user/{user_id}")
async def recommend_for_user(user_id: int, limit: int = 5):
    return {
        "user_id": user_id,
        "recommendations": recommender.recommend_for_user(user_id, limit)
    }

@app.get("/recommend/items/")
async def recommend_for_items(item_ids: str, limit: int = 5):
    item_ids = [int(id) for id in item_ids.split(",")]
    return {
        "item_ids": item_ids,
        "recommendations": recommender.recommend_for_items(item_ids, limit)
    }

@app.post("/train")
async def train_model():
    trainer = RecommendationTrainer(db)
    history = trainer.train()
    return {"status": "success", "epochs": len(history.history['loss'])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)