import psycopg2
import os
from pathlib import Path

# Read DATABASE_URL from .env file
db_url = None
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('DATABASE_URL='):
            db_url = line.split('=', 1)[1].strip()
            break

if not db_url:
    print("❌ DATABASE_URL not found in .env file")
    exit(1)

print(f"📡 Connecting to database...")

# Read SQL file
sql_file_path = Path('../database/init.sql')
sql_content = sql_file_path.read_text(encoding='utf-8')

print(f"📄 Read {len(sql_content)} characters from init.sql")

# Connect to database
try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    print("✅ Connected to database!")
    print("🔄 Running SQL script...")
    
    # Execute the SQL
    cursor.execute(sql_content)
    conn.commit()
    
    print("✅ Database initialized successfully!")
    print("📊 Tables created:")
    
    # List tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    for table in tables:
        print(f"   ✓ {table[0]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)