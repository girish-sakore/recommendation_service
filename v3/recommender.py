import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from config import Config
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, db):
        self.config = Config()
        self.db = db  # Store the injected database connection
        
        try:
            # Load model and encoders
            self.model = load_model(f"{self.config.MODEL_PATH}/model.h5")
            
            with open(f"{self.config.MODEL_PATH}/encoders.pkl", 'rb') as f:
                encoders = pickle.load(f)
                self.user_encoder = encoders['user_encoder']
                self.item_encoder = encoders['item_encoder']
            
            # Load item-restaurant mapping if exists
            mapping_path = f"{self.config.MODEL_PATH}/item_restaurant_mapping.pkl"
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.item_restaurant_mapping = pickle.load(f)
            else:
                self.item_restaurant_mapping = None
                
            logger.info("Recommender initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {str(e)}", exc_info=True)
            raise RuntimeError(f"Recommender initialization failed: {str(e)}")

    def recommend_for_cart(self, user_id: int, cart_item_id: int, top_n: int = 5) -> Optional[List[int]]:
        """Recommend items to add to cart based on current item"""
        try:
            # Get restaurant ID for the cart item
            restaurant_id = self.db.get_restaurant_for_item(cart_item_id)
            if not restaurant_id:
                logger.warning(f"No restaurant found for item {cart_item_id}")
                return self._fallback_recommendations(cart_item_id, restaurant_id, top_n)
            
            # Get user index (handle new users)
            try:
                user_idx = self.user_encoder.transform([user_id])[0]
            except ValueError:
                logger.info(f"New user detected: {user_id}")
                return self._fallback_recommendations(cart_item_id, restaurant_id, top_n)
            
            # Get all items from same restaurant
            restaurant_items = self.db.get_menu_items_by_restaurant(restaurant_id)
            restaurant_item_ids = [item['id'] for item in restaurant_items]
            
            # Filter out current item and items without embeddings
            candidate_items = [
                i for i in restaurant_item_ids 
                if i != cart_item_id and i in self.item_encoder.classes_
            ]
            
            if not candidate_items:
                logger.warning(f"No valid candidate items for restaurant {restaurant_id}")
                return self._fallback_recommendations(cart_item_id, restaurant_id, top_n)
            
            # Prepare model inputs
            user_indices = np.array([user_idx] * len(candidate_items))
            item_indices = np.array([self.item_encoder.transform([i])[0] for i in candidate_items])
            
            # Get predictions
            predictions = self.model.predict([user_indices, item_indices]).flatten()
            
            # Get top recommendations
            ranked_items = sorted(zip(candidate_items, predictions), 
                                key=lambda x: x[1], reverse=True)
            
            return [item[0] for item in ranked_items[:top_n]]
            
        except Exception as e:
            logger.error(f"Error in recommend_for_cart: {str(e)}", exc_info=True)
            return self._fallback_recommendations(cart_item_id, None, top_n)

    def similar_items(self, item_id: int, top_n: int = 5) -> Optional[List[int]]:
        """Find similar items in the same restaurant without user history"""
        try:
            # 1. Get restaurant ID for the target item
            restaurant_id = self.db.get_restaurant_for_item(item_id)
            if not restaurant_id:
                logger.warning(f"No restaurant found for item {item_id}")
                return None

            # 2. Get all items from the same restaurant
            restaurant_items = self.db.get_menu_items_by_restaurant(restaurant_id)
            restaurant_item_ids = [item['id'] for item in restaurant_items]
            
            # 3. Filter out current item and items without embeddings
            candidate_items = [
                i for i in restaurant_item_ids 
                if i != item_id and i in self.item_encoder.classes_
            ]
            
            if not candidate_items:
                logger.warning(f"No valid candidate items in restaurant {restaurant_id}")
                return None

            # 4. Get item embedding (handle unknown items)
            try:
                item_idx = self.item_encoder.transform([item_id])[0]
            except ValueError:
                logger.warning(f"Item {item_id} not in model vocabulary")
                return None

            # 5. Calculate similarities
            item_emb_layer = self.model.get_layer('item_embedding')
            item_emb_weights = item_emb_layer.get_weights()[0]
            
            # Get embeddings for all candidate items
            candidate_indices = [self.item_encoder.transform([i])[0] for i in candidate_items]
            candidate_embeddings = item_emb_weights[candidate_indices]

            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            target_embedding = item_emb_weights[item_idx].reshape(1, -1)
            similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]

            # 6. Combine and sort results
            ranked_items = sorted(zip(candidate_items, similarities), 
                            key=lambda x: x[1], reverse=True)
            
            return [item[0] for item in ranked_items[:top_n]]
            
        except Exception as e:
            logger.error(f"Error in similar_items: {str(e)}", exc_info=True)
            return None

    def _is_from_same_restaurant(self, item_id: int, restaurant_id: int) -> bool:
        """Check if item belongs to the specified restaurant"""
        if self.item_restaurant_mapping is not None:
            return item_id in self.item_restaurant_mapping[
                self.item_restaurant_mapping['restaurant_id'] == restaurant_id
            ]['menu_item_id'].values
        else:
            # Fallback to database query if mapping not available
            item_restaurant = self.db.get_restaurant_for_item(item_id)
            return item_restaurant == restaurant_id

    def _fallback_recommendations(self, cart_item_id: int, restaurant_id: Optional[int], top_n: int) -> List[int]:
        """Fallback recommendation strategy"""
        try:
            # 1. Try to get commonly paired items
            if restaurant_id:
                paired_items = self.db.get_common_pairs_for_item(cart_item_id, restaurant_id, top_n)
                if len(paired_items) >= top_n:
                    return paired_items[:top_n]
                
                # 2. Get popular items from same restaurant
                popular_items = self.db.get_popular_items_by_restaurant(restaurant_id, top_n)
                combined = list(dict.fromkeys(paired_items + popular_items))
                return combined[:top_n]
            
            # 3. Final fallback to global popular items
            return self.db.get_popular_items(top_n)
            
        except Exception as e:
            logger.error(f"Fallback recommendation failed: {str(e)}")
            return self.db.get_popular_items(top_n) if hasattr(self.db, 'get_popular_items') else []