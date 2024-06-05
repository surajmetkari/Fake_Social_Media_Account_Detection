from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import joblib
import hashlib
from functools import wraps
from tinydb import TinyDB, Query
import json

# Importing the necessary libraries for model loading and prediction
import pandas as pd
import numpy as np
import re
import gender_guesser.detector as gender
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load('model.ckpt')

# Load the trained model
model2 = joblib.load('classifier_model.pkl')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Database initialization for user management
db = TinyDB('database.json')
users_table = db.table('users')

def load_data():
    # Load data from profile.json
    with open('profile.json', 'r') as file:
        data = json.load(file)
    return data['_default']['1']

@app.route('/analyze')
def analyze():
    # Load data from profile.json
    data = load_data()
    sex = int(data['sex'])
    statuses_count = int(data['statuses_count'])
    followers = int(data['followers'])
    friends = int(data['friends'])
    fav = int(data['favourites'])
    listed_count = int(data['listed_count'])
    lang = int(data['lang'])

    # Making prediction
    prediction = model.predict([[sex, statuses_count, followers, friends, fav, listed_count, lang]])
        
    # Processing prediction result
    result = "Fake" if prediction > 0.5 else "Real"

    return render_template('analyze.html', data=data, result=result)

# Decorator to check if the user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Route to the home page
@app.route('/')
@login_required
def home():
    return render_template('index.html')

# Route to handle user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hashing the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Inserting user data into the database
        users_table.insert({
            'username': username,
            'password': hashed_password,
        })
        
        return redirect(url_for('login'))
    return render_template('signup.html')

# Route to handle user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        User = Query()
        user = users_table.get(User.username == username)
        if user and user['password'] == hashed_password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error=True)
    return render_template('login.html', error=False)

# Route to handle user logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    predicted_label = None
    if request.method == 'POST':
        text_input = request.form['text']
        text_input_vectorized = vectorizer.transform([text_input])
        predicted_label = model2.predict(text_input_vectorized)[0]
        
        # Store data in label.json using TinyDB
        db = TinyDB('label.json')
        db.insert({'text': text_input, 'predicted_label': predicted_label})

    return render_template('chat.html', predicted_label=predicted_label)

@app.route('/save_profile', methods=['GET', 'POST'])
def save_profile():
    if request.method == 'POST':
        # Extracting form data
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        bio = request.form['bio']
        sex = request.form['sex']
        statuses_count = int(request.form['statuses_count'])
        followers = int(request.form['followers'])
        friends = int(request.form['friends'])
        favourites = int(request.form['favourites'])
        listed_count = int(request.form['listed_count'])
        lang = request.form['lang']
        
        # Clearing existing data in the database
        db_profile = TinyDB('profile.json')
        db_profile.truncate()
        
        # Inserting profile data into the database
        db_profile.insert({
            'name': name,
            'username': username,
            'email': email,
            'mobile': mobile,
            'bio': bio,
            'sex': sex,
            'statuses_count': statuses_count,
            'followers': followers,
            'friends': friends,
            'favourites': favourites,
            'listed_count': listed_count,
            'lang': lang
        })
        
        return redirect(url_for('home'))
    
    # Return a response for GET requests
    return render_template('profile.html')



if __name__ == '__main__':
    app.run(debug=True)
