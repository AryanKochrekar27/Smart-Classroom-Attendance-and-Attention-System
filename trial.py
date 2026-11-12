import sqlite3

# Connect to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Query to fetch all rows from the attendance table
table_name = "attendance"
cursor.execute(f"SELECT * FROM {table_name};")
rows = cursor.fetchall()

# Fetch column names for the header
cursor.execute(f"PRAGMA table_info({table_name});")
columns = [col[1] for col in cursor.fetchall()]

# Print table contents in a clean format
print(f"\nContents of the '{table_name}' table:\n")
print(" | ".join(columns))  # Print column headers
print("-" * (len(columns) * 15))  # Separator
for row in rows:
    print(" | ".join(map(str, row)))

# Close the connection
conn.close()
