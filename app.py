from flask import Flask , redirect , url_for , request , render_template, session, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, username, password):
        self.username = username
        self.password = password
# Load trained model
with open("place_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("cleaned_dataset.csv") 

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Review"])



@app.route('/', methods=['GET','POST'])
@app.route('/home')
def index():
    if session.get('logged_in'):
        return redirect('recom')
    else:

        return render_template('index.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            db.session.add(User(username=request.form['username'], password=request.form['password']))
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return render_template('register.html', message="User Already Exists")
    else:
        return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            return redirect('recom')
        return render_template('login.html', message="Incorrect Details")

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))



@app.route("/recom")
def recom():
    return render_template("recom.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    city = request.args.get("city")
    review_keyword = request.args.get("review")

    # Filter the dataset by city
    city_df = df[df["City"].str.lower() == city.lower()].copy()

    if city_df.empty:
        return jsonify({"message": "No places found in the given city"}), 404

    # Apply vectorizer only on the filtered city dataset
    city_tfidf_matrix = vectorizer.transform(city_df["Review"])

    # Transform the user input keyword
    keyword_vector = vectorizer.transform([review_keyword])

    # Compute similarity scores
    scores = city_tfidf_matrix.dot(keyword_vector.T).toarray().flatten()

    # Ensure scores match the city_df length
    city_df["Review Score"] = scores

    # Sort by review score and rating
    city_df = city_df.sort_values(by=["Review Score", "Rating"], ascending=[False, False])

    # Return top 5 recommendations
    recommendations = city_df.head(5)[["Place Name", "Rating"]].to_dict(orient="records")

    return jsonify(recommendations)


if __name__ == "__main__":
    app.secret_key = "ThisIsNotASecret:p"
    with app.app_context():
        db.create_all()
        app.run(debug=True)