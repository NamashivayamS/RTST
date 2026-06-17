from database.connection import get_connection

try:
    conn = get_connection()

    cur = conn.cursor()

    cur.execute("SELECT NOW();")

    result = cur.fetchone()

    print("Connected Successfully")
    print(result)

    conn.close()

except Exception as e:
    print("Connection Failed")
    print(e)