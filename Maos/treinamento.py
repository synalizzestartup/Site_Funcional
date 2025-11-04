import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# --- Configurações ---
DATA_PATH = "dados_gestos.csv"
MODEL_PATH = "modelo_gestos.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# The number of landmarks collected per hand
NUM_LANDMARKS = 21 

# 1. Carregar os dados
print(f"Carregando dados de '{DATA_PATH}'...")
try:
    # Load the CSV, using the first row as the header by default
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_PATH}' não encontrado. Execute 'coleta_dados.py' primeiro.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    print("Verifique se o arquivo 'dados_gestos.csv' não está corrompido ou vazio.")
    exit()

# Check if the 'label' column exists. If not, the CSV might not have a header.
if 'label' not in df.columns:
    print("Erro: A coluna 'label' não foi encontrada no CSV.")
    print("Por favor, delete o arquivo 'dados_gestos.csv' e execute 'coleta_dados.py' novamente para recriar o arquivo com o cabeçalho correto.")
    exit()

if df.empty:
    print("Erro: O arquivo de dados está vazio. Colete os dados primeiro.")
    exit()

print("Dados carregados com sucesso!")
print(f"Total de amostras: {len(df)}")
print(f"Gestos encontrados: {df['label'].unique()}")

def extrair_e_normalizar_features(row):
    """
    Extrai e normaliza as features de uma linha do DataFrame.
    Esta função é IDÊNTICA à lógica em 'reconhecimento.py'.
    """
    # As coordenadas já estão relativas ao pulso pelo script de coleta.
    # A linha contém as 42 features (21 x,y pares).
    temp_features = row.values

    # 1. Normaliza pela escala (tamanho) da mão
    # Calcula a distância euclidiana máxima a partir do pulso (origem)
    max_dist = 0.0
    for i in range(0, len(temp_features), 2):
        dist = np.sqrt(temp_features[i]**2 + temp_features[i+1]**2)
        if dist > max_dist:
            max_dist = dist
    
    # Evita divisão por zero
    if max_dist == 0:
        max_dist = 1

    # 2. Divide todas as features pela distância máxima
    normalized_features = [f / max_dist for f in temp_features]
    return normalized_features

# 2. Preparar os dados
# X são as features (coordenadas), y é o label (gesto)
X_raw = df.drop('label', axis=1)
y = df['label']

print("\nNormalizando os dados de treinamento...")
# Aplica a função de normalização a cada linha do DataFrame
X_normalized_list = X_raw.apply(extrair_e_normalizar_features, axis=1)
# Converte a lista de listas resultante de volta para um DataFrame
X = pd.DataFrame(X_normalized_list.tolist(), index=X_raw.index, columns=X_raw.columns)
print("Dados normalizados com sucesso!")

# Codificar os labels (transformar 'mao_aberta', 'punho_fechado' em 0, 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Verificar se alguma classe tem menos de 2 amostras
class_counts = pd.Series(y_encoded).value_counts()
if (class_counts < 2).any():
    print("\nErro: Pelo menos uma classe de gesto tem menos de 2 amostras.")
    print("Isso impede a divisão estratificada dos dados para treino e teste.")
    print("\nContagem de amostras por gesto:")
    for label_id, count in class_counts.items():
        print(f"- Gesto '{le.inverse_transform([label_id])[0]}': {count} amostra(s)")
    print("\nPor favor, colete mais dados para os gestos com poucas amostras e tente novamente.")
    exit()

# Salvar o LabelEncoder para uso posterior na predição
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)
print(f"LabelEncoder salvo em '{LABEL_ENCODER_PATH}'.")

# 3. Dividir em treino e teste
# 80% para treino, 20% para teste. `stratify` garante uma divisão proporcional dos gestos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# 4. Treinar o modelo Random Forest
print("\nIniciando o treinamento do modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Treinamento concluído!")

# 5. Salvar o modelo treinado
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"Modelo salvo em '{MODEL_PATH}'.")

# 6. Avaliar o modelo
print("\nAvaliando o modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# 7. (Opcional) Visualizar a Matriz de Confusão
print("Gerando a matriz de confusão...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusão')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Previsto')
plt.tight_layout()
plt.savefig('matriz_confusao.png') # Salva a figura da matriz
print("Matriz de confusão salva como 'matriz_confusao.png'.")
plt.show() # Exibe a matriz
