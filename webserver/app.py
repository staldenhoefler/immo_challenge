from flask import Flask, render_template, request
import numpy as np
import joblib
import pgeocode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd

class KMeansClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=50):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        cluster_labels = self.kmeans.predict(X)
        return pd.DataFrame(cluster_labels, columns=["region_group"], index=X.index)

# Geo-utility for converting ZIP codes to lat/lon
geo = pgeocode.Nominatim('CH')

def zip_to_lat_lon(zip_code):
    """
    Convert ZIP code to latitude and longitude using pgeocode.
    """
    if isinstance(zip_code, int):
        zip_code = str(zip_code)
    location = geo.query_postal_code(zip_code)
    if location is not None and not location.isnull().any():
        return location.longitude, location.latitude
    return None, None



# Load preprocessing pipeline and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("linear_model.pkl")
kmeans_pipeline = joblib.load("kmeans_clusterer.pkl")

# Flask app initialization
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


def preprocess_and_predict(input_data):
    # Convert ZIP code to lon and lat
    zip_code = input_data.pop("postleithzahl")
    lon, lat = zip_to_lat_lon(zip_code)
    if lon is None or lat is None:
        raise ValueError("Ungültige Postleitzahl.")

    # Add lon and lat to input data
    input_data["lon"] = lon
    input_data["lat"] = lat

    # Create DataFrame for input data
    df_input = pd.DataFrame([input_data])
    print(df_input.info())

    region_group2 = kmeans_pipeline.transform(df_input[['lon', 'lat']])
    df_input['region_group'] = region_group2.values

    # Preprocess the data
    preprocessed_data = preprocessor.transform(df_input)

    # Predict using the model
    prediction = model.predict(preprocessed_data)

    # Reverse log transformation
    return np.exp(prediction)[0]


@app.route("/", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        form_data = request.form.to_dict()
        print("Formulardaten:", form_data)

        try:
            result = preprocess_and_predict(form_data)
            print("Vorhersage:", result)
        except ValueError as e:
            print("Fehler:", e)
            return render_template("home.html", error=str(e))

        return render_template("home.html", result=result)

    return render_template("home.html")




if __name__ == "__main__":
    app.run(port=5000, debug=True)
