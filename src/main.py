import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# LEITURA E PADRONIZAÇÃO
# ============================
file_path = "data/Machine_Learning_Datasets.xlsx"
excel = pd.ExcelFile(file_path)

df1 = pd.read_excel(excel, "DiaBD_A Diabetes Dataset for En")
df2 = pd.read_excel(excel, "diabetes_prediction_dataset")
df3 = pd.read_excel(excel, "diabetes_binary_5050split_healt")

def padronizar(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

df1, df2, df3 = padronizar(df1), padronizar(df2), padronizar(df3)

# ============================
# TRATAMENTO E CORREÇÕES
# ============================
df1["diabetic"] = df1["diabetic"].map({"No": 0, "Yes": 1})
df3["diabetes_binary"] = df3["diabetes_binary"].apply(lambda x: 1 if x >= 5 else 0)

if "hypertension" in df2.columns:
    df2 = df2.rename(columns={"hypertension": "hypertensive"})

for col in ["hvyalcoholconsump", "smoker"]:
    if col in df3.columns:
        df3[col] = df3[col].map({0: 0, 10: 1})

df1 = df1.rename(columns={"diabetic": "target"})
df2 = df2.rename(columns={"diabetes": "target"})
df3 = df3.rename(columns={"diabetes_binary": "target"})

def corrigir_bmi(x): return x / 10 if x > 60 else x
for df in [df1, df2, df3]:
    df["bmi"] = df["bmi"].apply(corrigir_bmi)

# ============================
# FEATURES DE CADA DATASET
# ============================
df1 = df1[["age", "bmi", "hypertensive", "target"]]
df2 = df2[["age", "bmi", "hba1c_level", "heart_disease", "hypertensive", "target"]]
df3 = df3[["age", "bmi", "physactivity", "income", "hvyalcoholconsump", "smoker", "target"]]

# ============================
# UNIFICAÇÃO E NORMALIZAÇÃO
# ============================
df = pd.concat([df1, df2, df3], ignore_index=True)

numeric = df.drop("target", axis=1).columns
df[numeric] = df[numeric].fillna(df[numeric].median())
df[numeric] = MinMaxScaler().fit_transform(df[numeric])

# ============================
# TREINO E MODELO
# ============================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
modelo.fit(X_train, y_train)

# ============================
# AJUSTE AUTOMÁTICO DO THRESHOLD
# ============================
y_proba = modelo.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    f1 = classification_report(
        y_test, preds, output_dict=True, zero_division=0
    )["weighted avg"]["f1-score"]

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# ============================
# PREDIÇÃO FINAL COM THRESHOLD AJUSTADO
# ============================
y_pred_final = (y_proba >= best_threshold).astype(int)

print("\n=== Threshold selecionado ===")
print(best_threshold)

importances = modelo.feature_importances_
feature_names = X.columns

# Gráficos
feat_imp = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values(by="importance", ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x="importance", y="feature")
plt.title("Importância das Variáveis - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusão (Threshold Ajustado)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()