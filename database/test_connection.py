from database.connection import init_pool, get_connection, release_connection

try:
    init_pool()
    conn = get_connection()

    cur = conn.cursor()

    cur.execute("SELECT CURRENT_TIMESTAMP;")

    result = cur.fetchone()

    print("Connected Successfully")
    print(result[0])

    release_connection(conn)

except Exception as e:
    print("Connection Failed")
    print(e)