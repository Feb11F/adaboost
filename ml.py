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
    def __init__(self, n_estimators, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.clfs = []
        self.clfs_weights = []
        self.bobot_awal = []
        self.bobot_data_baru = []

    def inisialisasi_bobot(self,X):
        weights = np.full(len(X), 1 / len(X))
        self.bobot_awal.append(weights)
        return weights

    def calc_weight_error(self,y,y_pred):
       return np.sum(y != y_pred)/len(y_pred)

    def  calc_error(self, y, y_pred) :
        #1 ketika y != pred
        err = np.array([y != y_pred])
        err = err.astype(int)
        return err

    def update_weight_error(self, weights, pred_weight, error):
        new_weights = weights * (e**(pred_weight * error))
        new_weights = new_weights / np.sum(new_weights)
        return new_weights

    def pred_weight(self, learning_rate, weighted_error):
        EPS = 1e-10
        bobot = learning_rate * np.log((1 - weighted_error + EPS) / (weighted_error + EPS))
        return bobot

    def select_data_indices(self,weights):
        flattened_weights = np.ravel(weights)  # Meratakan array bobot

        # Memilih data dengan probabilitas berdasarkan bobot
        data_indices = np.random.choice(len(flattened_weights), size=len(flattened_weights), replace=True, p=flattened_weights)

        return data_indices

    def fit(self, X, y):
        weights = self.inisialisasi_bobot(X)
        # print("ini bobot data awal",weights)
        # print("target asli",y)
        for i in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=2)
            clf.fit(X, y)
            self.clfs.append(clf)

            # y_pred = clf.predict(X)
            y_pred = [1, 1, 0, 0, 0, 1, 0, 1, 1, 0]
            print("ini prediksi",y_pred)

            rj = self.calc_weight_error(y, y_pred)
            # print("weight error",rj)

            alpha_j = self.pred_weight(self.learning_rate,rj)
            self.clfs_weights.append(alpha_j)

            # print("ini bobot predictor",self.clfs_weights)

            error = self.calc_error(y, y_pred)

            print("ini error nya",error)

            update_bobot_data = self.update_weight_error(weights, alpha_j, error)
            self.bobot_data_baru.append(update_bobot_data)

            print("ini bobot data baru",update_bobot_data)

            new_indices = self.select_data_indices(update_bobot_data)
            print("new_indices",new_indices)

            # X =[X[index] for index in new_indices]
            # y =[y[index] for index in new_indices]

            X = X[new_indices]
            y = y[new_indices]

            print(X)
            print(y)

        # Normalisasi clfs_weights
        self.clfs_weights = self.clfs_weights / np.sum(self.clfs_weights)
        return self.clfs, self.clfs_weights

    def predict(self, X):
        m = X.shape[0]  # Jumlah data dalam X
        predictions = np.zeros((m, len(self.clfs)))  # Array untuk menyimpan prediksi dari setiap classifier

        # Melakukan prediksi dari setiap classifier
        for i in range(len(self.clfs)):
            clf = self.clfs[i]
            for j in range(m):
                predictions[j, i] = clf.predict([X[j]])
        print(predictions)

        final_predictions = np.zeros(m)  # Array untuk menyimpan prediksi akhir

        for i in range(m):  # Iterasi melalui setiap data
            pred_weights = np.zeros(len(self.clfs_weights))  # Array untuk menyimpan bobot prediktor dengan hasil prediksi yang sama
            for j in range(len(self.clfs_weights)):  # Iterasi melalui setiap prediktor
                clf_weight = self.clfs_weights[j]
                if np.array_equal(predictions[i], predictions[i, j]):# Membandingkan semua elemen menggunakan np.array_equal
                    pred_weights[j] = clf_weight

            max_weight_idx = np.argmax(pred_weights)
            final_predictions[i] = predictions[i, max_weight_idx]  # Memilih hasil prediksi dengan bobot prediktor terbesar

        print(final_predictions)

        return final_predictions


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

        X_train = X_train[:,:10]

        # Create and train AdaBoostClassifier
        adaboost = AdaBoostClassifier(n_estimators=3, learning_rate=0.1)
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
            submit = st.form_submit_button("submit")
            inputs = np.array([mean_radius,mean_tektstur,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_compacity,mean_concapoints,mean_simmetry,mean_fratical_dimension])
            input_norm = np.array(inputs).reshape(-1, 10)
            input_pred = adaboost.predict(input_norm)
            if submit:
                st.subheader('Hasil Prediksi')
            # Menampilkan hasil prediksi
                
                st.success(input_pred[0])
                st.write("Accuracy:", accuracy)

        
          


        
