import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Carregar os dados
file_path = 'LLM.csv'
data = pd.read_csv(file_path)

# Remover linhas com valores nulos
data.dropna(inplace=True)

# Pré-processamento: Vetorização dos textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['Label']

# Divisão dos dados em treinamento (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização dos classificadores
classifiers = {
    'LDA': LDA(),
    'QDA': QDA(),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

# Treinamento e avaliação dos classificadores
roc_auc_scores = {}
plt.figure(figsize=(12, 8))
for name, clf in classifiers.items():
    try:
        clf.fit(X_train.toarray(), y_train)
        y_pred_proba = clf.predict_proba(X_test.toarray())[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='ai')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_auc_scores[name] = roc_auc
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    except Exception as e:
        print(f"Erro com o classificador {name}: {e}")

plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('/app/roc_curves.png')

# Ordenação dos classificadores pela AUC
sorted_classifiers = sorted(roc_auc_scores.items(), key=lambda x: x[1], reverse=True)
print("Classificadores ordenados pela AUC:")
for clf, auc in sorted_classifiers:
    print(f"{clf}: AUC = {auc:.2f}")
