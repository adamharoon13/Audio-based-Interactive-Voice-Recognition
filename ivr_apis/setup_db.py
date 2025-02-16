import sqlite3

# # Connect to (or create) the database
# conn = sqlite3.connect("patients.db")
# cursor = conn.cursor()

# # Create the patients table with a due_amount field
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS patients (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     account_number TEXT UNIQUE NOT NULL,
#     first_name TEXT NOT NULL,
#     last_name TEXT NOT NULL,
#     dob DATE NOT NULL,
#     due_amount REAL NOT NULL  -- Due amount for the patient
# );
# """)

# # Insert a test account number with due amount
# cursor.execute("INSERT OR IGNORE INTO patients (account_number, first_name,last_name, dob, due_amount) VALUES ('343469420', 'Adam', 'Johnson', '13/07/2000', 150.75)")

# cursor.execute("INSERT OR IGNORE INTO patients (account_number, first_name,last_name, dob, due_amount) VALUES ('4204206969', 'Maria', 'Collin', '19/12/1992', 55.00)")

# # Commit and close
# conn.commit()
# conn.close()

# print("Database setup complete.")


# import sqlite3

# conn = sqlite3.connect("patients.db")  # Ensure the path is correct
# cursor = conn.cursor()

# # Check if the "patients" table exists
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients';")
# table_exists = cursor.fetchone()

# if table_exists:
#     print("✅ Table 'patients' exists!")
# else:
#     print("❌ Table 'patients' does NOT exist!")

# conn.close()
# import sqlite3

# Database file
DB_FILE = "patients.db"

# Function to fetch and print all patient records
def fetch_all_patients():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # cursor.execute("INSERT INTO patients (account_number, first_name,last_name, dob, due_amount) VALUES ('4204206969', 'Maria', 'Collins', '19/12/1992', 55.00)")
        cursor.execute("SELECT * FROM patients")
        rows = cursor.fetchall()

        if not rows:
            print("No records found in the database.")
        else:
            print("\n--- Patients in Database ---")
            for row in rows:
                print(row)

    except sqlite3.Error as e:
        print(f"Database error: {e}")

    finally:
        conn.close()

# Run the test
if __name__ == "__main__":
    fetch_all_patients()
