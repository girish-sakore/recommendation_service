import numpy as np
import pickle
from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import MeanAbsoluteError
from config import Config

class Recommender:
    def __init__(self):
        self.config = Config()
        self.model = load_model(f"{self.config.MODEL_PATH}/model.h5")
        
        with open(f"{self.config.MODEL_PATH}/encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.item_encoder = encoders['item_encoder']
    
    def recommend_for_user(self, user_id, top_n=5):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return None  # New user
        
        all_item_indices = np.arange(len(self.item_encoder.classes_))
        user_indices = np.array([user_idx] * len(all_item_indices))
        
        predictions = self.model.predict([user_indices, all_item_indices])
        predictions = predictions.flatten()
        
        top_indices = predictions.argsort()[-top_n:][::-1]
        return self.item_encoder.inverse_transform(top_indices).tolist()
    
    def similar_items(self, item_id, top_n=5):
        try:
            item_idx = self.item_encoder.transform([item_id])[0]
        except ValueError:
            return None
        
        # Get item embedding weights
        item_emb_layer = self.model.get_layer('item_embedding')
        item_emb_weights = item_emb_layer.get_weights()[0]
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(item_emb_weights[item_idx].reshape(1, -1), item_emb_weights)
        
        # Get top N similar items (excluding itself)
        similar_indices = similarities.argsort()[0][-top_n-1:-1][::-1]
        return self.item_encoder.inverse_transform(similar_indices).tolist()