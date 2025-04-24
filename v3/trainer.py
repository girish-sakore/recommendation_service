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
        print('Running prepare_data')
        data = self.db.get_orders_data()
        df = pd.DataFrame(data)
        
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['menu_item_id'])
        
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
        
        return history