import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pickle
import os
from config import Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RecommendationTrainer:
    def __init__(self, db):
        self.db = db
        self.config = Config()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
    
    def prepare_data(self):
        # Get order data with restaurant information
        data = self.db.get_orders_data_with_restaurants()
        df = pd.DataFrame(data)
        
        # Encode users and items
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['menu_item_id'])
        
        # Create co-occurrence features (items bought together)
        df = self._add_cooccurrence_features(df)
        
        # Store restaurant mapping
        self.item_restaurant_mapping = df[['menu_item_id', 'restaurant_id']].drop_duplicates()
        
        return df

    def _add_cooccurrence_features(self, df):
        """Add features about items commonly ordered together"""
        # Get items commonly ordered together in the same restaurant
        cooccurrence = self.db.get_cooccurrence_stats()
        cooccurrence_df = pd.DataFrame(cooccurrence)
        
        # Merge with main dataframe
        df = df.merge(
            cooccurrence_df,
            on=['menu_item_id', 'restaurant_id'],
            how='left'
        )
        
        # Fill NA for items without co-occurrence data
        df['cooccurrence_score'] = df['cooccurrence_score'].fillna(0)
        
        return df

    def build_model(self, num_users, num_items):
        print('Running build_model')
        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=num_users, 
            output_dim=self.config.EMBEDDING_SIZE, 
            name='user_embedding'
        )(user_input)
        user_vec = Flatten()(user_embedding)
        
        # Item embedding
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(
            input_dim=num_items, 
            output_dim=self.config.EMBEDDING_SIZE, 
            name='item_embedding'
        )(item_input)
        item_vec = Flatten()(item_embedding)
        
        # Dot product
        dot = Dot(axes=1)([user_vec, item_vec])
        
        model = Model(inputs=[user_input, item_input], outputs=dot)
        # model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        model.compile(
            optimizer=Adam(0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        
        return model
    
    def train(self):
        print('Running train')
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
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.TRAIN_EPOCHS,
            validation_data=([X_test[:, 0], X_test[:, 1]], y_test)
        )
        
        # Save model and encoders
        os.makedirs(self.config.MODEL_PATH, exist_ok=True)
        save_model(model, f"{self.config.MODEL_PATH}/model.h5")
        
        with open(f"{self.config.MODEL_PATH}/encoders.pkl", 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder
            }, f)
        
        # Save item-restaurant mapping
        with open(f"{self.config.MODEL_PATH}/item_restaurant_mapping.pkl", 'wb') as f:
            pickle.dump(self.item_restaurant_mapping, f)
        
        return history