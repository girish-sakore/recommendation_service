import numpy as np
import pickle
from tensorflow.keras.models import load_model
from config.settings import settings

class Recommender:
    def __init__(self):
        # Load model and encoders
        self.model = load_model(f"{settings.MODEL_PATH}/recommendation_model.h5")
        
        with open(f"{settings.MODEL_PATH}/encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.item_encoder = encoders['item_encoder']
    
    def recommend_for_user(self, user_id, top_n=5):
        """Get personalized recommendations for a user"""
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            # New user - return popular items
            return self.get_popular_items(top_n)
        
        # Get all item indices
        all_item_indices = np.arange(len(self.item_encoder.classes_))
        
        # Predict scores for all items
        user_indices = np.array([user_idx] * len(all_item_indices))
        predictions = self.model.predict([user_indices, all_item_indices])
        predictions = predictions.flatten()
        
        # Get top N items
        top_indices = predictions.argsort()[-top_n:][::-1]
        recommended_items = self.item_encoder.inverse_transform(top_indices)
        
        return recommended_items.tolist()
    
    def recommend_for_items(self, item_ids, top_n=5):
        """Get recommendations based on items in cart"""
        # Find similar users who ordered these items
        # Then recommend what those users also ordered
        
        # This is a simplified version - you could implement item-item similarity
        return self.get_popular_items(top_n)
    
    def get_popular_items(self, top_n=5):
        """Fallback to most popular items"""
        # In practice, you'd query the DB for most ordered items
        # This is just a placeholder
        return self.item_encoder.classes_[:top_n].tolist()