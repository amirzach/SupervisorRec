from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
import mysql.connector
import os
import random

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  
    'database': 'project_supervisor_rec'
}

# Database connection function
def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn, conn.cursor(dictionary=True)

# Generate 11-digit random ID
def generate_random_id():
    return random.randint(100, 999)

# Create table if it doesn't exist
def initialize_db():
    conn, cursor = get_db_connection()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student (
        StudentID BIGINT PRIMARY KEY,
        StdName VARCHAR(100) UNIQUE NOT NULL,
        StdPassword VARCHAR(255) NOT NULL,
        StdEmail VARCHAR(100) UNIQUE NOT NULL
    )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

# Initialize database on app startup
@app.before_first_request
def before_first_request():
    initialize_db()

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('homepage'))
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return "", 204  # Return no content status code

def get_unique_student_id():
    conn, cursor = get_db_connection()
    try:
        while True:
            student_id = generate_random_id()
            cursor.execute("SELECT StudentID FROM student WHERE StudentID = %s", (student_id,))
            if not cursor.fetchone():
                return student_id
    finally:
        cursor.close()
        conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Handle POST request
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Check if username or email already exists
    conn, cursor = get_db_connection()
    try:
        cursor.execute("SELECT StdName, StdEmail FROM student WHERE StdName = %s OR StdEmail = %s", 
                      (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            error_message = ""
            if existing_user['StdName'] == username:
                error_message = "Username already exists. Please choose a different username."
            elif existing_user['StdEmail'] == email:
                error_message = "Email already registered. Please use a different email."
            return render_template('register.html', error=error_message)
        
        # If no existing user, proceed with registration
        student_id = get_unique_student_id()
        cursor.execute(
            "INSERT INTO student (StudentID, StdName, StdEmail, StdPassword) VALUES (%s, %s, %s, %s)",
            (student_id, username, email, password)
        )
        conn.commit()
        return redirect(url_for('index'))
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return render_template('register.html', error="Registration failed. Please try again.")
    finally:
        cursor.close()
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn, cursor = get_db_connection()
    try:
        cursor.execute("SELECT * FROM student WHERE StdName = %s", (username,))
        user = cursor.fetchone()
        
        # Debug print
        print(f"User: {user}")
        
        if user:
            # Debug print
            print(f"Checking password for {username}")
            print(f"Stored password: {user['StdPassword']}")
            
            # Direct password comparison
            password_correct = (user['StdPassword'] == password)
            print(f"Password correct: {password_correct}")
            
            if password_correct:
                session['username'] = username
                session['user_id'] = user['StudentID']
                return redirect(url_for('homepage'))
        
        error_message = "Invalid username or password. Please try again."
        return render_template('index.html', error=error_message)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        error_message = "Login failed. Please try again later."
        return render_template('index.html', error=error_message)
    finally:
        cursor.close()
        conn.close()

@app.route('/homepage')
def homepage():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('homepage.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)