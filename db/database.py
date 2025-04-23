import mysql.connector
from mysql.connector import pooling
from config.settings import settings

class Database:
    def __init__(self):
        self.pool = pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=5,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )

    def get_connection(self):
        return self.pool.get_connection()

    def get_orders_data(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT 
                    o.user_id, 
                    oi.menu_item_id, 
                    COUNT(*) as order_count
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                GROUP BY o.user_id, oi.menu_item_id
            """)
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def get_menu_items(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT id, name, category_id 
                FROM menu_items
            """)
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def get_user_orders(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT menu_item_id 
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE o.user_id = %s
            """, (user_id,))
            return [item['menu_item_id'] for item in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
