import sqlite3
import pandas as pd

# ... (Assume 'df' is the DataFrame you just built from the MAL API) ...

# 1. Create a connection to a local SQLite database file
# If 'anime_database.db' doesn't exist, Python will create it automatically.
conn = sqlite3.connect('anime_database.db')

# 2. Write the DataFrame directly into a SQL table named 'user_ratings'
# 'if_exists="append"' ensures that when you loop through new users, 
# it adds their data to the bottom of the table instead of overwriting it.
df.to_sql(name='user_ratings', con=conn, if_exists='append', index=False)

print("Data successfully saved to the database!")

# 3. Always close the connection when done
conn.close()