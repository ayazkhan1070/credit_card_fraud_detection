import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn

from joblib import load

try:
    model1 = load(r"D:\ds_intern\iso_forest.pkl")
    model2 = load(r"D:\ds_intern\lof.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
scaler = load(r"D:\ds_intern\scaler.pkl")
pca=load(r"D:\ds_intern\pca.pkl")
 # Load your model
# with open("D:/ds_intern/Section02.pkl", 'rb') as file:
#     model = pickle.load(file)


df1=pd.read_csv('./credit_cardtest.csv')

def detect_fraudulent_transactions(new_data: pd.DataFrame, model):
    columns=new_data.columns
    for col in columns:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    new_data.dropna(inplace=True)
    new_data[['Time', 'Amount']] = scaler.transform(new_data[['Time', 'Amount']])
    pca_features = new_data.values
    pca_transformed = pca.transform(pca_features)
    pca_df = pd.DataFrame(data=pca_transformed, columns=['PC1', 'PC2'])
    X = pca_df.values
    y_pred_model = model.predict(X)
    y_pred_model = [1 if pred == -1 else 0 for pred in y_pred_model]
    #fraudulent_transactions = pca_df[y_pred_model == 1]

    return y_pred_model
print(model2)
print(detect_fraudulent_transactions(df1, model2))
