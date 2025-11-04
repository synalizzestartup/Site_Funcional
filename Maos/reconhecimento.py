import cv2
import mediapipe as mp
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configurações e Carregamento do Modelo ---
MODEL_PATH = "modelo_gestos.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
NUM_LANDMARKS = 21

print("Carregando modelo...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    print("Erro: Arquivos de modelo não encontrados. Execute 'treinamento.py' primeiro.")
    exit()

print("Modelo carregado com sucesso!")

def extrair_features_from_request(landmarks_data):
    """
    Extrai e normaliza as features da mão a partir dos dados recebidos na requisição.
    Esta função é IDÊNTICA à lógica em 'treinamento.py'.
    """
    temp_features = []
    # Pega as coordenadas do pulso (ponto 0) como referência
    pulso_x = landmarks_data[0]['x']
    pulso_y = landmarks_data[0]['y']

    # 1. Calcula as coordenadas relativas ao pulso (como no script de coleta)
    for landmark in landmarks_data:
        landmark_x = landmark['x']
        landmark_y = landmark['y']
        temp_features.extend([landmark_x - pulso_x, landmark_y - pulso_y])

    # 2. Normaliza pela escala (tamanho) da mão
    # Calcula a distância euclidiana máxima a partir do pulso para usar como fator de escala
    max_dist = 0.0
    for i in range(0, len(temp_features), 2):
        dist = np.sqrt(temp_features[i]**2 + temp_features[i+1]**2)
        if dist > max_dist:
            max_dist = dist
    
    # Evita divisão por zero se a mão não for detectada corretamente
    if max_dist == 0: max_dist = 1

    # 3. Divide todas as features pela distância máxima
    features = [f / max_dist for f in temp_features]
    return features

# --- Configuração do Servidor Flask ---
app = Flask(__name__)
CORS(app) # Permite que o frontend (em outro domínio/porta) acesse esta API

@app.route('/recognize', methods=['POST'])
def recognize_gesture():
    """
    Endpoint da API para receber os landmarks e retornar a predição.
    """
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({"error": "Dados de landmarks ausentes"}), 400

    try:
        # Extrai as features da mesma forma que no treinamento
        features = extrair_features_from_request(data['landmarks'])
        
        # Faz a predição com o modelo carregado
        prediction_numeric = model.predict([features])[0]
        predicted_label = le.inverse_transform([prediction_numeric])[0]
        
        # Retorna o resultado como JSON
        return jsonify({"prediction": predicted_label})

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return jsonify({"error": "Falha ao processar os landmarks"}), 500

if __name__ == '__main__':
    # Inicia o servidor Flask na porta 5000
    # O host '0.0.0.0' permite que ele seja acessível na sua rede local
    print("Iniciando servidor Flask em http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)