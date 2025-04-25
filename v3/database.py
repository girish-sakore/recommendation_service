from typing import List
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

    def get_orders_data_with_restaurants(self):
        query = """
        SELECT 
            o.user_id,
            oi.menu_item_id,
            b.restaurant_id,
            COUNT(*) as order_count
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        JOIN branches b ON mi.menu_category_id = b.id
        GROUP BY o.user_id, oi.menu_item_id, b.restaurant_id
        """

        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def get_popular_items_by_restaurant(self, restaurant_id, limit=5):
        query = """
        SELECT oi.menu_item_id, COUNT(*) as order_count
        FROM order_items oi
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        JOIN branches b ON mi.menu_category_id = b.id
        WHERE b.restaurant_id = %s
        GROUP BY oi.menu_item_id
        ORDER BY order_count DESC
        LIMIT %s
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (restaurant_id, limit))
        result = [item['menu_item_id'] for item in cursor.fetchall()]
        cursor.close()
        return result

    def get_cooccurrence_stats(self):
        """Get stats on items commonly ordered together"""
        query = """
        SELECT 
            a.menu_item_id,
            b.menu_item_id AS paired_item_id,
            br.restaurant_id,
            COUNT(*) AS pair_count,
            COUNT(*) / (
                SELECT COUNT(*) 
                FROM order_items 
                WHERE menu_item_id = a.menu_item_id
            ) AS cooccurrence_score
        FROM order_items a
        JOIN order_items b ON a.order_id = b.order_id AND a.menu_item_id != b.menu_item_id
        JOIN menu_items mi_a ON a.menu_item_id = mi_a.id
        JOIN menu_items mi_b ON b.menu_item_id = mi_b.id
        JOIN branches br ON mi_a.menu_category_id = br.id
        GROUP BY a.menu_item_id, b.menu_item_id, br.restaurant_id
        HAVING pair_count > 1  -- Only significant pairs
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def get_restaurant_for_item(self, cart_item_id):
        query = """
        SELECT DISTINCT b.restaurant_id
        FROM order_items oi
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        JOIN branches b ON mi.menu_category_id = b.id
        WHERE oi.id = %s
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (cart_item_id,))
        result = cursor.fetchone()
        cursor.close()
        return result['restaurant_id'] if result else None   

    def get_common_pairs_for_item(self, cart_item_id: int, restaurant_id: int, limit: int = 5) -> List[int]:
        query = """
        SELECT 
            other_items.menu_item_id,
            COUNT(*) AS pair_count
        FROM order_items AS main_item
        JOIN order_items AS other_items ON 
            main_item.order_id = other_items.order_id
            AND main_item.menu_item_id != other_items.menu_item_id
        JOIN menu_items AS other_menu_items ON 
            other_items.menu_item_id = other_menu_items.id
        JOIN branches ON 
            other_menu_items.menu_category_id = branches.id
        WHERE 
            main_item.id = %s
            AND branches.restaurant_id = %s
        GROUP BY other_items.menu_item_id
        ORDER BY pair_count DESC
        LIMIT %s
        """
        
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (cart_item_id, restaurant_id, limit))
        result = [item['menu_item_id'] for item in cursor.fetchall()]
        cursor.close()
        return result

    def get_menu_items_by_restaurant(self, restaurant_id):
        query = """
        SELECT menu_items.id
        FROM menu_items
        JOIN menu_categories ON menu_items.menu_category_id = menu_categories.id
        JOIN branches ON menu_categories.branch_id = branches.id
        JOIN restaurants ON branches.restaurant_id = restaurants.id
        WHERE restaurants.id=%s
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query, (restaurant_id,))
        result = cursor.fetchall()
        cursor.close()
        return result

    def close(self):
        if self.connection.is_connected():
            self.connection.close()