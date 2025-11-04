import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# --- Configurações ---
DATA_PATH = "dados_gestos.csv"  # Nome do arquivo para salvar os dados
NUM_LANDMARKS = 21  # Número de landmarks da mão

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extrair_features(hand_landmarks):
    """
    Converte os landmarks da mão em uma lista simples de coordenadas normalizadas.
    """
    features = []
    # Pega as coordenadas do pulso (ponto 0) como referência
    pulso_x = hand_landmarks.landmark[0].x
    pulso_y = hand_landmarks.landmark[0].y

    # Adiciona as coordenadas de todos os pontos relativas ao pulso
    for i in range(NUM_LANDMARKS):
        landmark_x = hand_landmarks.landmark[i].x
        landmark_y = hand_landmarks.landmark[i].y
        features.append(landmark_x - pulso_x)
        features.append(landmark_y - pulso_y)

    return features

def salvar_dados(dados, nome_arquivo):
    """
    Salva a lista de dados em um arquivo CSV.
    """
    file_exists = os.path.isfile(nome_arquivo)
    with open(nome_arquivo, 'a', newline='') as f:
        writer = csv.writer(f)
        # Escreve o cabeçalho apenas se o arquivo for novo
        if not file_exists:
            header = []
            for i in range(NUM_LANDMARKS):
                header += [f'x{i}', f'y{i}']
            header.append('label')
            writer.writerow(header)
        
        # Escreve os dados
        writer.writerows(dados)
    print(f"\n{len(dados)} amostras salvas em '{nome_arquivo}'!")

# --- Coleta de Dados ---
if __name__ == "__main__":
    # Pede informações ao usuário
    nome_gesto_left = input("Digite o nome do gesto para a MÃO ESQUERDA (ex: A_esq): ")
    nome_gesto_right = input("Digite o nome do gesto para a MÃO DIREITA (ex: B_dir): ")
    num_amostras = int(input("Digite o número de amostras a coletar (ex: 100): "))

    dados_coletados = []
    amostras_capturadas = 0
    cap = cv2.VideoCapture(0)

    print("\nPosicione as DUAS MÃOS na câmera. Pressione 's' para iniciar a coleta.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Inverte a imagem para um efeito de espelho, facilitando o posicionamento
        # frame = cv2.flip(frame, 1)
        
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Coleta de Dados - Pressione 's' para iniciar", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            # Contagem regressiva
            for i in range(3, 0, -1):
                print(f"Iniciando em {i}...")
                time.sleep(1)
            
            print("Coletando amostras...")
            while amostras_capturadas < num_amostras:
                ret, frame_coleta = cap.read()
                if not ret: break
                
                frame_coleta = cv2.flip(frame_coleta, 1)
                frame_rgb_coleta = cv2.cvtColor(frame_coleta, cv2.COLOR_BGR2RGB)
                results_coleta = hands.process(frame_rgb_coleta)

                # Garante que estamos coletando dados apenas quando AMBAS as mãos são detectadas
                if results_coleta.multi_hand_landmarks and len(results_coleta.multi_hand_landmarks) == 2:
                    for i, hand_landmarks in enumerate(results_coleta.multi_hand_landmarks):
                        # Identifica se a mão é esquerda ou direita
                        handedness = results_coleta.multi_handedness[i].classification[0].label
                        
                        features = extrair_features(hand_landmarks)
                        if handedness == 'Left':
                            dados_coletados.append(features + [nome_gesto_left])
                        elif handedness == 'Right':
                            dados_coletados.append(features + [nome_gesto_right])
                    amostras_capturadas += 1
                    print(f"Amostra {amostras_capturadas}/{num_amostras} coletada.", end='\r')

            salvar_dados(dados_coletados, DATA_PATH)
            break # Sai do loop principal após a coleta
        elif key == 27: # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()