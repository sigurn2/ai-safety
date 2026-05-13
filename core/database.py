from core.config import MYSQL_DATABASE,MYSQL_CHARSET,MYSQL_HOST,MYSQL_PASSWORD,MYSQL_PORT,MYSQL_USER
import mysql.connector

def _check_connection():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        port=MYSQL_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        print(row)
    
    cursor.close()
    conn.close()






if __name__ == "__main__":
    _check_connection()