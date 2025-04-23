import mysql.connector
from mysql.connector import Error
from config import Config

class MySQLDatabase:
    def __init__(self):
        self.config = Config()
        print("Host:", self.config)
        # print("Port:", self.config.DB_PORT)
        self.connection = self._create_connection()
    
    def _create_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=33061,
                database=self.config.DB_NAME,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD
            )
            return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise
    
    def get_orders_data(self):
        query = """
        SELECT 
            o.user_id, 
            oi.menu_item_id, 
            COUNT(*) as order_count
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        GROUP BY o.user_id, oi.menu_item_id
        """
        
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_menu_items(self):
        query = "SELECT id, name FROM menu_items"
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_user_orders(self, user_id):
        query = """
        SELECT menu_item_id 
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        WHERE o.user_id = %s
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (user_id,))
        result = [item['menu_item_id'] for item in cursor.fetchall()]
        cursor.close()
        return result
    
    def get_popular_items(self, limit=5):
        query = """
        SELECT menu_item_id, COUNT(*) as order_count
        FROM order_items
        GROUP BY menu_item_id
        ORDER BY order_count DESC
        LIMIT %s
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (limit,))
        result = [item['menu_item_id'] for item in cursor.fetchall()]
        cursor.close()
        return result
    
    def close(self):
        if self.connection.is_connected():
            self.connection.close()