import streamlit as st
from streamlit_option_menu import option_menu
from sklearn import datasets
from sklearn. tree import DecisionTreeClassifier
import numpy as np
from sklearn import datasets
from math import e
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.clfs = []
        self.clfs_weights = []

    def calc_weighted_error(self, y_true, y_pred):
        return np.sum(y_true != y_pred) / len(y_pred)

    def calc_error(self, y_true, y_pred):
        res = np.array([y_true != y_pred])
        res = res.astype(int)
        return res

    def update_weights(self, weights, pred_weight, error):
        new_weights = weights * (np.exp(pred_weight * error))
        new_weights = new_weights / np.sum(new_weights)
        return new_weights

    def calc_pred_weight(self, learning_rate, weighted_error):
        EPS = 1e-10
        return learning_rate * np.log((1 - weighted_error + EPS) / (weighted_error + EPS))

    def fit(self, X, y):
        m, n = X.shape
        weights = np.full(len(X), 1 / len(X))

        for i in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=2)
            clf.fit(X, y)
            self.clfs.append(clf)

            y_pred = clf.predict(X)
            weighted_error = self.calc_weighted_error(y, y_pred)
            pred_weight = self.calc_pred_weight(self.learning_rate, weighted_error)
            self.clfs_weights.append(pred_weight)

            error = self.calc_error(y, y_pred)
            weights = self.update_weights(weights, pred_weight, error)

            new_indices = np.random.choice(m, m, p=weights.ravel())
            X = X[new_indices]
            y = y[new_indices]

        # Normalisasi clfs_weights
        self.clfs_weights = self.clfs_weights / np.sum(self.clfs_weights)

    def predict(self,X):
      n_estimators = len(self.clfs)
      y_pred = np.zeros(len(X))
      pred_weights = np.zeros(len(X))

      for i in range(n_estimators):
          clf = self.clfs[i]
          pred_weight = self.clfs_weights[i]
          pred = clf.predict(X)

          # Menambahkan bobot prediksi pada posisi yang sesuai
          y_pred += pred_weight * pred
          pred_weights[pred == 1] += pred_weight
          pred_weights[pred == 0] -= pred_weight

      # Menentukan kelas prediksi berdasarkan jumlah bobot
      y_pred = np.sign(y_pred)
      y_pred[pred_weights >= 0] = 1
      y_pred[pred_weights < 0] = 0

      return y_pred

st.set_page_config(
    page_title="Implemntasi Adabooost",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">IMPLEMENTASI ADABOOST MEACHINE LEARNING</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Dr. Indah Agustien Siradjuddin, S.Kom., M.Kom",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset", "Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://asset.kompas.com/crops/LlS_K6YXiqlztK08GKshvg_m15U=/152x0:997x563/750x500/data/photo/2022/05/24/628c83ebd499d.jpg" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    
    if selected =="Dataset" :

        from sklearn.datasets import load_breast_cancer
        # Memuat dataset breast cancer
        data = load_breast_cancer()

        # Membuat DataFrame dari data dan target
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write(df.head())                 
    if selected == "Implementation":
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score


        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train AdaBoostClassifier
        adaboost = AdaBoostClassifier(n_estimators=5, learning_rate=0.1)
        adaboost.fit(X_train, y_train)

        # adaboost.clfs_weights
        np.sum(adaboost.clfs_weights)
        # Perform predictions on the test set
        y_pred = adaboost.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        with st.form("my_form"):
            st.subheader("Implementasi")
            mean_radius = st.number_input('Masukkan Mean radius')
            mean_tektstur = st.number_input('Masukkan Mean texture')
            mean_perimeter = st.number_input('Masukkan Mean perimeter')
            mean_area = st.number_input('Masukkan Mean area')
            mean_smoothness = st.number_input('Masukkan Mean smoothness')
            mean_compactness = st.number_input('Masukkan Mean compactness')
            mean_compacity = st.number_input('Masukkan Mean concavity')
            mean_concapoints = st.number_input('Masukkan Mean concave points')
            mean_simmetry = st.number_input('Masukkan Mean symmetry')
            mean_fratical_dimension = st.number_input('Masukkan Mean fractal dimension')
            err_radius = st.number_input('Masukkan radius error')
            err_tektstur = st.number_input('Masukkan texture error')
            err_perimeter = st.number_input('Masukkan perimeter error')
            err_area = st.number_input('Masukkan area error')
            err_smoothness = st.number_input('Masukkan smoothness error')
            err_compactness = st.number_input('Masukkan compactness error')
            err_compacity = st.number_input('Masukkan concavity error')
            err_concapoints = st.number_input('Masukkan concave points error')
            err_simmetry = st.number_input('Masukkan symmetry error')
            err_fratical_dimension = st.number_input('Masukkan fractal dimension error')
            worst_radius = st.number_input('Masukkan worst radius')
            worst_tektstur = st.number_input('Masukkan worst texture')
            worst_perimeter = st.number_input('Masukkan worst perimeter')
            worst_area = st.number_input('Masukkan worst area')
            worst_smoothness = st.number_input('Masukkan worst smoothness')
            worst_compactness = st.number_input('Masukkan worst compactness')
            worst_compacity = st.number_input('Masukkan worst concavity')
            worst_concapoints = st.number_input('Masukkan worst concave points')
            worst_simmetry = st.number_input('Masukkan worst symmetry')
            worst_fratical_dimension = st.number_input('Masukkan worst fractal dimension') 
            submit = st.form_submit_button("submit")
            inputs = np.array([mean_radius,mean_tektstur,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_compacity,mean_concapoints,mean_simmetry,mean_fratical_dimension,err_radius,err_tektstur,err_perimeter,err_area,err_smoothness,err_compactness,err_compacity,err_concapoints,err_simmetry,err_fratical_dimension,worst_radius,worst_tektstur,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_compacity,worst_concapoints,worst_simmetry,worst_fratical_dimension])
            input_norm = np.array(inputs).reshape(1, -1)
            input_pred = adaboost.predict(input_norm)
            if submit:
                st.subheader('Hasil Prediksi')
            # Menampilkan hasil prediksi
                
                st.success(input_pred[0])
                st.write("Accuracy:", accuracy)

        
          


        
