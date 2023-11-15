import mariadb

# 로컬
def db_connector():
    # Connect to MariaDB Platform
    conn = mariadb.connect(
        user="root",
        password="mariadb2022",
        host="127.0.0.1",
        port=3306,
        database="book_analysis"
    )
    return conn
