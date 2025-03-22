from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory, jsonify
import mysql.connector
import os
import random
from AiEngine import get_recommender

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
        
        if user and user['StdPassword'] == password:
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

@app.route('/api/supervisors', methods=['GET'])
def get_all_supervisors():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    conn, cursor = get_db_connection()
    try:
        cursor.execute("""
            SELECT s.SupervisorID, s.SvName, GROUP_CONCAT(e.Expertise SEPARATOR ', ') as expertise_areas
            FROM supervisor s
            LEFT JOIN expertise e ON s.SupervisorID = e.SupervisorID
            GROUP BY s.SupervisorID
            ORDER BY s.SvName
        """)
        
        supervisors = cursor.fetchall()
        return jsonify({"supervisors": supervisors})
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return jsonify({"error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()
        
@app.route('/supervisor_list.html')
def supervisor_list():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('supervisor_list.html')

@app.route('/profiles.html')
def profiles():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('profiles.html')

@app.route('/past_fyp.html')
def past_fyp():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get FYP projects from database
    conn, cursor = get_db_connection()
    try:
        cursor.execute("""
            SELECT ProjectID, Title, Author, Abstract, Year 
            FROM past_fyp
            ORDER BY year DESC, Author
        """)
        
        fyp_projects = cursor.fetchall()
        return render_template('past_fyp.html', projects=fyp_projects)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return render_template('past_fyp.html', error="Failed to load projects", projects=[])
    finally:
        cursor.close()
        conn.close()

# New route for handling supervisor search
@app.route('/api/search_supervisors', methods=['GET'])
def search_supervisors():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    query = request.args.get('query', '')
    min_score = float(request.args.get('min_score', 0.1))
    top_n = int(request.args.get('top_n', 5))
    
    try:
        recommender = get_recommender()
        results = recommender.search_supervisors(query, min_score, top_n)
        return jsonify({"results": results})
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

# Return supervisor details for a specific supervisor
@app.route('/api/supervisor/<int:supervisor_id>', methods=['GET'])
def get_supervisor(supervisor_id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    conn, cursor = get_db_connection()
    try:
        # Get supervisor details
        cursor.execute("""
            SELECT s.SupervisorID, s.SvName, s.SvEmail, GROUP_CONCAT(e.Expertise SEPARATOR ', ') as expertise_areas
            FROM supervisor s
            LEFT JOIN expertise e ON s.SupervisorID = e.SupervisorID
            WHERE s.SupervisorID = %s
            GROUP BY s.SupervisorID
        """, (supervisor_id,))
        
        supervisor = cursor.fetchone()
        if not supervisor:
            return jsonify({"error": "Supervisor not found"}), 404
            
        return jsonify(supervisor)
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return jsonify({"error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)