import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn import tree

# lendo o excel
file_path = "data/Machine_Learning_Datasets.xlsx"
excel_file = pd.ExcelFile(file_path)

# lendo cada planilha separada
df1 = pd.read_excel(excel_file, sheet_name="DiaBD_A Diabetes Dataset for En")
df2 = pd.read_excel(excel_file, sheet_name="diabetes_prediction_dataset")
df3 = pd.read_excel(excel_file, sheet_name="diabetes_binary_5050split_healt")

# limpreza e padronização ------

# função para padronizar nomes de colunas
def padronizar_colunas(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

df1 = padronizar_colunas(df1)
df2 = padronizar_colunas(df2)
df3 = padronizar_colunas(df3)

# converter targets para formato numérico consistente (0 = não diabético, 1 = diabético) antes (No, Yes)
df1['diabetic'] = df1['diabetic'].map({'No': 0, 'Yes': 1})
# no dataset 3, converter escala 0–10 para 0–1
df3['diabetes_binary'] = df3['diabetes_binary'].apply(lambda x: 1 if x >= 5 else 0)

# corrigir valores muito altos de idade e BMI no dataset 2
df2['age'] = df2['age'].apply(lambda x: x / 10 if x > 120 else x)
df2['bmi'] = df2['bmi'].apply(lambda x: x / 10 if x > 100 else x)

# criando o decodificador ------
le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['diabetic'] = df1['diabetic'].astype(int)

df2['gender'] = le.fit_transform(df2['gender'])
df2['smoking_history'] = le.fit_transform(df2['smoking_history'])

# criando o normalizador ------
scaler = MinMaxScaler()

df1_scaled = df1.copy()
df1_scaled[df1.columns] = scaler.fit_transform(df1)

df2_scaled = df2.copy()
df2_scaled[df2.columns] = scaler.fit_transform(df2)

df3_scaled = df3.copy()
df3_scaled[df3.columns] = scaler.fit_transform(df3)

# seleção e padroniação das colunas de interesse ------
df1_selected = df1[['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'pulse_rate', 'diabetic']].copy()
df1_selected.rename(columns={'diabetic': 'target'}, inplace=True)

df2_selected = df2[['age', 'bmi', 'hba1c_level', 'smoking_history', 'diabetes']].copy()
df2_selected.rename(columns={'diabetes': 'target'}, inplace=True)

df3_selected = df3[['age', 'bmi', 'physactivity', 'genhlth', 'income', 'diabetes_binary']].copy()
df3_selected.rename(columns={'diabetes_binary': 'target'}, inplace=True)

# mudando as idades para float ------
df1_selected['age'] = df1_selected['age'].astype(float)
df2_selected['age'] = df2_selected['age'].astype(float)
df3_selected['age'] = df3_selected['age'].astype(float)

# arrumando o BMI que está com valores maiores que o padrão
df1_selected['bmi'] = df1_selected['bmi'] / 100.0
df2_selected['bmi'] = df2_selected['bmi'] / 10.0
df3_selected['bmi'] = df3_selected['bmi'] / 10.0

# mudando os bmi para float
df1_selected['bmi'] = df1_selected['bmi'].astype(float)
df2_selected['bmi'] = df2_selected['bmi'].astype(float)
df3_selected['bmi'] = df3_selected['bmi'].astype(float)

# garantindo que todas as colunas estejam com o mesmo tipo
df1_selected = df1_selected.astype(float)
df2_selected = df2_selected.astype(float)
df3_selected = df3_selected.astype(float)

# concatenando os três datasets ------
df_final = pd.concat([df1_selected, df2_selected, df3_selected], ignore_index=True)

# preencher NaNs com a mediana (para colunas numéricas)
for col in df_final.columns:
    if df_final[col].dtype in ['float64', 'int64']:
        df_final[col] = df_final[col].fillna(df_final[col].median())

# separar X e y
X = df_final.drop('target', axis=1)
y = df_final['target']

# criar o modelo de árvore de decisão
modelo = DecisionTreeClassifier(random_state=42)

# treinamento com profundidade limitada
modelo = DecisionTreeClassifier(
    max_depth=3,       # limita a profundidade da árvore
    criterion="gini",  # pode trocar por "entropy" se quiser comparar
    random_state=42
)
modelo.fit(X, y)

# visualização da árvore
plt.figure(figsize=(14, 8))
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=["Não diabético", "Diabético"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão (profundidade limitada a 3 níveis)", fontsize=14, pad=20)
plt.show()