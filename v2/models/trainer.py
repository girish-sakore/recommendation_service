import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import pickle
import os
from config.settings import settings

class RecommendationTrainer:
    def __init__(self, db):
        self.db = db
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
    
    def prepare_data(self):
        data = self.db.get_orders_data()
        df = pd.DataFrame(data)
        
        # Encode user and item IDs
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['menu_item_id'])
        
        return df
    
    def build_model(self, num_users, num_items):
        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=num_users, 
            output_dim=settings.EMBEDDING_SIZE, 
            name='user_embedding'
        )(user_input)
        user_vec = Flatten()(user_embedding)
        
        # Item embedding
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(
            input_dim=num_items, 
            output_dim=settings.EMBEDDING_SIZE, 
            name='item_embedding'
        )(item_input)
        item_vec = Flatten()(item_embedding)
        
        # Dot product
        dot = Dot(axes=1)([user_vec, item_vec])
        
        # Combined model
        model = Model(inputs=[user_input, item_input], outputs=dot)
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train(self):
        df = self.prepare_data()
        
        X = df[['user_idx', 'item_idx']].values
        y = df['order_count'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Build model
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.item_encoder.classes_)
        model = self.build_model(num_users, num_items)
        
        # Train
        history = model.fit(
            [X_train[:, 0], X_train[:, 1]],
            y_train,
            batch_size=64,
            epochs=settings.TRAIN_EPOCHS,
            validation_data=([X_test[:, 0], X_test[:, 1]], y_test)
        )
        
        # Save model and encoders
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        model.save(f"{settings.MODEL_PATH}/recommendation_model.h5")
        
        with open(f"{settings.MODEL_PATH}/encoders.pkl", 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder
            }, f)
        
        return history