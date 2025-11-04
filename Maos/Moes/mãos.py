
import time
import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
contador = 0
width, height = 250, 250
capturando_crops = False
crops_restantes = 0
hand1_pos_inicial = None
hand2_pos_inicial = None

# MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh   

WINDOW_NAME = "Mao - Reconhecimento"
cap = cv2.VideoCapture(0)

is_fullscreen = True
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands, mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,  # até 2 rostos
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe usa RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        results_face = face_mesh.process(img_rgb)

        # Desenhar Face Mesh (468 pontos + conexões)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,   # malha completa
                    landmark_drawing_spec=None,          # sem desenhar os pontos pequenos
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,165,255), thickness=1, circle_radius=1)
                )
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os 21 pontos + conexões
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para sair
            break
        elif key == ord('f'): # Alterna tela cheia
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('s'):  # tecla "s" para iniciar captura de 60 crops
            time.sleep(0.5)
            # Cria os diretórios se não existirem
            os.makedirs("diretorio_hand1", exist_ok=True)
            os.makedirs("diretorio_hand2", exist_ok=True)
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Pega a posição inicial do ponto 9 das duas mãos detectadas
                hand_landmarks1 = results.multi_hand_landmarks[0]
                ponto9_1 = hand_landmarks1.landmark[9]
                x_init1 = int(ponto9_1.x * w)
                y_init1 = int(ponto9_1.y * h)
                hand1_pos_inicial = (x_init1, y_init1)
                if len(results.multi_hand_landmarks) > 1:
                    hand_landmarks2 = results.multi_hand_landmarks[1]
                    ponto9_2 = hand_landmarks2.landmark[9]
                    x_init2 = int(ponto9_2.x * w)
                    y_init2 = int(ponto9_2.y * h)
                    hand2_pos_inicial = (x_init2, y_init2)
                else:
                    hand2_pos_inicial = None
                capturando_crops = True
                crops_restantes = 60

        if capturando_crops and crops_restantes > 0:
            if results.multi_hand_landmarks and hand1_pos_inicial is not None:
                h, w, _ = frame.shape
                # Encontra a mão mais próxima da posição inicial de hand1
                min_dist1 = None
                hand1_landmarks = None
                min_dist2 = None
                hand2_landmarks = None
                for hand_landmarks in results.multi_hand_landmarks:
                    ponto9 = hand_landmarks.landmark[9]
                    x = int(ponto9.x * w)
                    y = int(ponto9.y * h)
                    dist1 = (x - hand1_pos_inicial[0])**2 + (y - hand1_pos_inicial[1])**2
                    if (min_dist1 is None) or (dist1 < min_dist1):
                        min_dist1 = dist1
                        hand1_landmarks = hand_landmarks
                    if hand2_pos_inicial is not None:
                        dist2 = (x - hand2_pos_inicial[0])**2 + (y - hand2_pos_inicial[1])**2
                        if (min_dist2 is None) or (dist2 < min_dist2):
                            min_dist2 = dist2
                            hand2_landmarks = hand_landmarks
                # Faz o crop da mão mais próxima da posição inicial de hand1
                if hand1_landmarks is not None:
                    ponto9 = hand1_landmarks.landmark[9]
                    x = int(ponto9.x * w)
                    y = int(ponto9.y * h)
                    x1 = max(x - width//2, 0)
                    y1 = max(y - height//2, 0)
                    x2 = min(x + width//2, w)
                    y2 = min(y + height//2, h)
                    mao_crop = frame[y1:y2, x1:x2]
                    nome_arquivo = f"diretorio_hand1/hand1_mao_{contador}.png"
                    cv2.imwrite(nome_arquivo, mao_crop)
                    print(f"Screenshot salva: {nome_arquivo}")
                # Faz o crop da mão mais próxima da posição inicial de hand2
                if hand2_landmarks is not None and hand2_pos_inicial is not None:
                    ponto9 = hand2_landmarks.landmark[9]
                    x = int(ponto9.x * w)
                    y = int(ponto9.y * h)
                    x1 = max(x - width//2, 0)
                    y1 = max(y - height//2, 0)
                    x2 = min(x + width//2, w)
                    y2 = min(y + height//2, h)
                    mao_crop = frame[y1:y2, x1:x2]
                    nome_arquivo = f"diretorio_hand2/hand2_mao_{contador}.png"
                    cv2.imwrite(nome_arquivo, mao_crop)
                    print(f"Screenshot salva: {nome_arquivo}")
                contador += 1
            crops_restantes -= 1
            if crops_restantes == 0:
                capturando_crops = False
                hand1_pos_inicial = None
                hand2_pos_inicial = None
cap.release()
cv2.destroyAllWindows()
