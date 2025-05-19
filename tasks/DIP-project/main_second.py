import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.transform import resize
from skimage.filters import sobel, gaussian, threshold_otsu
from skimage.morphology import closing, square, remove_small_objects
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import dump, load
import time
import warnings

# Suprimir avisos
warnings.filterwarnings("ignore")

# Configurações
DATA_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(DATA_DIR, 'lego')  # Diretório das imagens
if not os.path.exists(IMAGES_DIR):
    IMAGES_DIR = os.path.join(DATA_DIR)  # Fallback para diretório atual
    
RANDOM_STATE = 42
MODEL_DIR = os.path.join(DATA_DIR, 'models_improved')
os.makedirs(MODEL_DIR, exist_ok=True)

# Carregar os dados
print("Carregando dados...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"Total de imagens de treino: {len(train_df)}")
print(f"Total de imagens de teste: {len(test_df)}")

# Lista de defeitos
defect_columns = ['has_deffect', 'no_hat', 'no_face', 'no_head', 'no_leg', 'no_body', 'no_hand', 'no_arm']

# Mostrar distribuição de classes
print("\nDistribuição de defeitos no conjunto de treino:")
for col in defect_columns:
    counts = train_df[col].value_counts()
    print(f"{col}: {counts.to_dict()}")

# Função para carregar imagem com tratamento de erros
def load_image(image_id, resize_dim=(224, 224)):
    try:
        # Tentar vários formatos de arquivo
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return cv2.resize(img, resize_dim)
        
        print(f"Imagem não encontrada: {image_id}")
        return np.zeros((resize_dim[0], resize_dim[1], 3), dtype=np.uint8)
    except Exception as e:
        print(f"Erro ao carregar imagem {image_id}: {e}")
        return np.zeros((resize_dim[0], resize_dim[1], 3), dtype=np.uint8)

# Funções de pré-processamento avançado
def preprocess_image(img):
    # Converter para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Equalização de histograma para melhorar contraste
    gray_eq = cv2.equalizeHist(gray)
    
    # Redução de ruído
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    # Binarização adaptativa para lidar com diferentes iluminações
    binary = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Operações morfológicas para remover ruído
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return {
        'original': img,
        'gray': gray,
        'gray_eq': gray_eq,
        'binary': binary,
        'morphed': morphed
    }

# Extração de características aprimorada
def extract_advanced_features(img):
    # Pré-processar a imagem
    processed = preprocess_image(img)
    features = []
    
    # 1. Histogramas de cores (RGB) - com mais bins para capturar mais detalhes
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [64], [0, 256])
        hist = hist / hist.sum()  # Normalizar
        features.extend(hist.flatten())
    
    # 2. Local Binary Patterns com múltiplos raios
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(processed['gray'], n_points, radius, method='uniform')
        hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist_lbp = hist_lbp.astype('float') / (hist_lbp.sum() + 1e-10)  # Evitar divisão por zero
        features.extend(hist_lbp)
    
    # 3. Características de forma baseadas em contornos
    contours, _ = cv2.findContours(processed['binary'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Área, perímetro, número de contornos, etc.
    total_area = sum(cv2.contourArea(cnt) for cnt in contours) if contours else 0
    features.append(total_area / (img.shape[0] * img.shape[1]))
    
    total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours) if contours else 0
    features.append(total_perimeter / (2 * (img.shape[0] + img.shape[1])))  # Normalizado
    
    features.append(len(contours))
    
    # Contornos grandes (possíveis partes do Lego)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100] if contours else []
    features.append(len(large_contours))
    
    # Razão de aspecto dos maiores contornos
    if large_contours:
        aspect_ratios = []
        for cnt in sorted(large_contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratios.append(float(w) / (h + 1e-10))
        
        features.extend(aspect_ratios + [0] * (3 - len(aspect_ratios)))  # Pad to 3
    else:
        features.extend([0, 0, 0])
    
    # 4. HOG (Histogram of Oriented Gradients) para capturar formas
    hog_image = processed['gray']
    hog_image = resize(hog_image, (64, 64))  # Redimensionar para acelerar
    hog_features, _ = hog(
        hog_image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys'
    )
    # Reduzir dimensionalidade escolhendo apenas alguns recursos
    features.extend(hog_features[::10])  # Pegar cada 10º valor
    
    # 5. Características de textura GLCM com múltiplos ângulos
    gray_scaled = (processed['gray'] / 16).astype(np.uint8)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray_scaled, distances, angles, levels=16, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        features.extend(graycoprops(glcm, prop).flatten())
    
    # 6. Análise de simetria (Legos geralmente são simétricos)
    h, w = processed['gray'].shape
    left_half = processed['gray'][:, :w//2]
    right_half = np.fliplr(processed['gray'][:, w//2:])
    
    # Ajustar tamanho se necessário
    min_w = min(left_half.shape[1], right_half.shape[1])
    h_sym_diff = np.abs(left_half[:, :min_w] - right_half[:, :min_w]).mean()
    features.append(h_sym_diff)
    
    top_half = processed['gray'][:h//2, :]
    bottom_half = np.flipud(processed['gray'][h//2:, :])
    
    # Ajustar tamanho se necessário
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    v_sym_diff = np.abs(top_half[:min_h, :] - bottom_half[:min_h, :]).mean()
    features.append(v_sym_diff)
    
    # 7. Divida a imagem em 4x4 regiões para capturar características locais
    rows, cols = 4, 4
    cell_height, cell_width = img.shape[0] // rows, img.shape[1] // cols
    
    for i in range(rows):
        for j in range(cols):
            cell = img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            cell_gray = processed['gray'][i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            
            # Média e desvio padrão de cor para cada canal RGB
            for channel in range(3):
                features.append(np.mean(cell[:,:,channel]))
                features.append(np.std(cell[:,:,channel]))
            
            # Média e desvio padrão da intensidade em escala de cinza
            features.append(np.mean(cell_gray))
            features.append(np.std(cell_gray))
            
            # Energia da textura (soma dos quadrados)
            features.append(np.sum(cell_gray ** 2))
    
    # 8. Medidas baseadas em bordas
    sobel_edges = sobel(processed['gray'])
    features.append(np.mean(sobel_edges))
    features.append(np.std(sobel_edges))
    
    return np.array(features)

# Função para criar variações simples das imagens (data augmentation)
def augment_image(img, label):
    augmented_images = [img]
    augmented_labels = [label]
    
    # Flip horizontal
    flipped_h = cv2.flip(img, 1)
    augmented_images.append(flipped_h)
    augmented_labels.append(label)
    
    # Pequenas rotações (apenas se a imagem tiver defeito)
    if label[0] == 1:  # se has_deffect=1
        for angle in [5, -5]:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)
            augmented_labels.append(label)
    
    return augmented_images, augmented_labels

# Função para treinar um modelo com validação cruzada
def train_with_cv(X, y, defect_idx, defect_name):
    print(f"\nTreinando modelo para {defect_name} (índice {defect_idx})...")
    
    # Preparar target para este defeito específico
    y_defect = y[:, defect_idx].copy()
    
    # Verificar o tipo dos dados e converter para inteiro de forma segura
    if np.issubdtype(y_defect.dtype, np.floating):
        # Se for float, arredondar e converter para inteiro
        y_defect = np.round(y_defect).astype(int)
    else:
        # Se já for inteiro ou outro tipo, apenas converter diretamente
        y_defect = y_defect.astype(int)
    
    # Verificar se há classes únicas suficientes para estratificação
    unique_classes = np.unique(y_defect)
    if len(unique_classes) < 2:
        print(f"AVISO: Apenas uma classe encontrada para {defect_name}. Usando divisão aleatória.")
        # Usar validação cruzada não estratificada se houver apenas uma classe
        skf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        # Configurar validação cruzada estratificada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Calcular pesos das classes para balanceamento
    class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_defect)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Inicializar modelos
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    
    # Criar ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    # Criar pipeline com SMOTE para balanceamento
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', ensemble)
    ])
    
    # Inicializar métricas para acompanhamento
    cv_scores = []
    
    # Iterar pelas divisões da validação cruzada
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_defect)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_defect[train_idx], y_defect[val_idx]
        
        print(f"Fold {fold+1}/5 - Treinando...")
        
        # Treinar o modelo
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Avaliar
        y_pred = pipeline.predict(X_val_fold)
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        accuracy = accuracy_score(y_val_fold, y_pred)
        
        print(f"Fold {fold+1} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        cv_scores.append(f1)
    
    # Treinar modelo final com todos os dados
    print("Treinando modelo final com todos os dados...")
    pipeline.fit(X, y_defect)
    
    # Salvar o modelo
    model_path = os.path.join(MODEL_DIR, f"{defect_name}_model.joblib")
    dump(pipeline, model_path)
    
    print(f"Média das pontuações F1 nas validações: {np.mean(cv_scores):.4f}")
    print(f"Modelo para {defect_name} salvo em {model_path}")
    
    return pipeline

# Carregar e extrair características avançadas das imagens de treino
print("\nCarregando e extraindo características avançadas das imagens de treino...")
X_train = []
y_train = []
augmented = False  # Flag para controlar se já fizemos augmentation

# Verificar se já existem características extraídas
features_file = os.path.join(DATA_DIR, 'advanced_features_train.npz')
if os.path.exists(features_file):
    print("Carregando características pré-processadas...")
    loaded = np.load(features_file, allow_pickle=True)
    X_train = loaded['X']
    y_train = loaded['y']
    augmented = True
else:
    for idx, row in train_df.iterrows():
        img_id = row['example_id']
        print(f"Processando imagem {idx+1}/{len(train_df)}: {img_id}")
        
        img = load_image(img_id)
        features = extract_advanced_features(img)
        
        label = row[defect_columns].values
        
        # Aplicar data augmentation apenas para imagens com defeitos
        if row['has_deffect'] == 1:
            aug_images, aug_labels = augment_image(img, label)
            
            for aug_img, aug_label in zip(aug_images, aug_labels):
                if aug_img is not img:  # Não duplicar a imagem original
                    aug_features = extract_advanced_features(aug_img)
                    X_train.append(aug_features)
                    y_train.append(aug_label)
        
        X_train.append(features)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Salvar as características extraídas para uso futuro
    np.savez(features_file, X=X_train, y=y_train)
    augmented = True

print(f"Características extraídas. Shape: {X_train.shape}")
if augmented:
    print("Dados aumentados com técnicas de data augmentation")

# Treinar modelos para cada tipo de defeito
models = {}
print("\nTreinando modelos para cada tipo de defeito...")

# Verificar se já temos modelos treinados
all_models_exist = True
for i, defect in enumerate(defect_columns):
    model_path = os.path.join(MODEL_DIR, f"{defect}_model.joblib")
    if not os.path.exists(model_path):
        all_models_exist = False
        break

if all_models_exist:
    print("Carregando modelos pré-treinados...")
    for i, defect in enumerate(defect_columns):
        model_path = os.path.join(MODEL_DIR, f"{defect}_model.joblib")
        models[defect] = load(model_path)
        print(f"Modelo para {defect} carregado.")
else:
    for i, defect in enumerate(defect_columns):
        models[defect] = train_with_cv(X_train, y_train, i, defect)

# Processar imagens de teste
print("\nProcessando imagens de teste...")
X_test = []

# Verificar se já existem características de teste extraídas
test_features_file = os.path.join(DATA_DIR, 'advanced_features_test.npz')
if os.path.exists(test_features_file):
    print("Carregando características de teste pré-processadas...")
    X_test = np.load(test_features_file, allow_pickle=True)['X']
else:
    for idx, row in test_df.iterrows():
        img_id = row['example_id']
        print(f"Processando imagem de teste {idx+1}/{len(test_df)}: {img_id}")
        
        img = load_image(img_id)
        features = extract_advanced_features(img)
        X_test.append(features)
    
    X_test = np.array(X_test)
    
    # Salvar as características extraídas para uso futuro
    np.savez(test_features_file, X=X_test)

print(f"Características de teste extraídas. Shape: {X_test.shape}")

# Fazer previsões
print("\nRealizando previsões...")
predictions = np.zeros((len(test_df), len(defect_columns)))

for i, defect in enumerate(defect_columns):
    model = models[defect]
    # Usar predict_proba para obter probabilidades
    try:
        proba = model.predict_proba(X_test)
        # Obter probabilidade da classe positiva (1)
        predictions[:, i] = proba[:, 1]
    except:
        # Fallback para predict caso o modelo não suporte predict_proba
        predictions[:, i] = model.predict(X_test).astype(float)

# Criar DataFrame de submissão
submission = test_df.copy()
for i, col in enumerate(defect_columns):
    submission[col] = predictions[:, i]

# # Salvar a submissão
# submission.to_csv('submission_second.csv', index=False)
# print("\nSubmissão salva em 'submission_second.csv'")

# Convertendo probabilidades para classificações binárias (0 ou 1)
print("\nConvertendo probabilidades para classificações binárias...")
for i, col in enumerate(defect_columns):
    # Você pode ajustar o threshold (0.5 é o padrão)
    threshold = 0.5
    submission[col] = (predictions[:, i] >= threshold).astype(int)

# Salvar a submissão com valores binários
submission.to_csv('submission_second.csv', index=False)
print("\nSubmissão binária salva em 'submission_second.csv'")

# Visualizar algumas imagens de teste e suas previsões
def visualize_test_predictions(num_samples=5):
    plt.figure(figsize=(15, 5 * num_samples))
    
    # Pegar amostras aleatórias
    samples = test_df.sample(min(num_samples, len(test_df))).index
    
    for i, idx in enumerate(samples):
        img_id = test_df.iloc[idx]['example_id']
        img = load_image(img_id)
        
        # Mostrar imagem
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"ID de Teste: {img_id}")
        plt.axis('off')
        
        # Mostrar previsões
        pred_values = predictions[idx]
        defect_preds = {defect: prob for defect, prob in zip(defect_columns, pred_values)}
        
        # Ordenar defeitos por probabilidade
        sorted_defects = sorted(defect_preds.items(), key=lambda x: x[1], reverse=True)
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.axis('off')
        
        text = "Probabilidades de Defeitos:\n"
        for defect, prob in sorted_defects:
            text += f"{defect}: {prob:.4f}\n"
        
        plt.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('visualizacao_previsoes_teste.png')
    print("\nVisualização de previsões de teste salva em 'visualizacao_previsoes_teste.png'")

# Visualizar imagens de validação com ground truth vs. previsão
def visualize_validation(num_samples=3):
    # Separar um pequeno conjunto de validação
    X_val_sample = X_train[-num_samples:]
    y_val_sample = y_train[-num_samples:]
    
    # Fazer previsões neste conjunto
    val_predictions = np.zeros((len(X_val_sample), len(defect_columns)))
    for i, defect in enumerate(defect_columns):
        model = models[defect]
        try:
            proba = model.predict_proba(X_val_sample)
            val_predictions[:, i] = proba[:, 1]
        except:
            val_predictions[:, i] = model.predict(X_val_sample).astype(float)
    
    plt.figure(figsize=(15, 7 * num_samples))
    
    for i in range(num_samples):
        # Encontrar o exemplo correspondente no DataFrame original
        # (Isso é uma simplificação; na prática você precisaria de uma forma mais robusta de mapear)
        
        # Mostrar previsões vs. ground truth
        plt.subplot(num_samples, 1, i+1)
        plt.axis('off')
        
        text = "Ground Truth vs. Previsão:\n"
        for j, defect in enumerate(defect_columns):
            text += f"{defect}: {y_val_sample[i][j]:.1f} vs {val_predictions[i][j]:.4f}\n"
        
        plt.text(0.1, 0.5, text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('visualizacao_validacao.png')
    print("\nVisualização de validação salva em 'visualizacao_validacao.png'")

try:
    visualize_test_predictions()
    visualize_validation()
except Exception as e:
    print(f"Erro ao criar visualizações: {e}")

print("\nProcesso concluído! Verifique o arquivo 'submission_second.csv' para submissão.")