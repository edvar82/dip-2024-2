import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops, label
import time
from joblib import dump, load

# Configurações
DATA_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(DATA_DIR, 'lego')  # Ajuste para o diretório correto onde estão as imagens
RANDOM_STATE = 42

# Verificar se o diretório existe
if not os.path.exists(IMAGES_DIR):
    print(f"AVISO: Diretório de imagens não encontrado: {IMAGES_DIR}")
    IMAGES_DIR = os.path.join(DATA_DIR)  # Tenta usar o diretório atual
    print(f"Tentando usar diretório alternativo: {IMAGES_DIR}")

# Carregar os dados
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"Total de imagens de treino: {len(train_df)}")
print(f"Total de imagens de teste: {len(test_df)}")

# Mostrar distribuição de classes
print("\nDistribuição de defeitos no conjunto de treino:")
defect_columns = ['has_deffect', 'no_hat', 'no_face', 'no_head', 'no_leg', 'no_body', 'no_hand', 'no_arm']
for col in defect_columns:
    counts = train_df[col].value_counts()
    print(f"{col}: {counts.to_dict()}")

# Função para carregar e redimensionar imagem
def load_image(image_id, resize=(128, 128)):
    try:
        # Tentar vários formatos de arquivo
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return cv2.resize(img, resize)
        
        # Se não encontrar a imagem
        print(f"Imagem não encontrada: {image_id}")
        return np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
    except Exception as e:
        print(f"Erro ao carregar imagem {image_id}: {e}")
        return np.zeros((resize[0], resize[1], 3), dtype=np.uint8)

# Função para extrair características da imagem
def extract_features(img):
    features = []
    
    # 1. Histogramas de cores (RGB)
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # 2. Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 3. Local Binary Patterns para textura
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype('float') / sum(hist_lbp)
    features.extend(hist_lbp)
    
    # 4. Medidas de forma baseadas em bordas
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Área total dos contornos
    total_area = sum(cv2.contourArea(cnt) for cnt in contours) if contours else 0
    features.append(total_area / (img.shape[0] * img.shape[1]))
    
    # Número de contornos
    features.append(len(contours))
    
    # 5. Características de textura GLCM
    if gray.size > 0:
        # Reduzir a escala de cinza para acelerar o cálculo da GLCM
        gray_scaled = (gray / 16).astype(np.uint8)
        glcm = graycomatrix(gray_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
        
        # Calcular propriedades da GLCM
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in props:
            features.extend(graycoprops(glcm, prop).flatten())
    
    # 6. Divida a imagem em regiões e calcule estatísticas para cada região
    rows, cols = 2, 2
    cell_height, cell_width = img.shape[0] // rows, img.shape[1] // cols
    
    for i in range(rows):
        for j in range(cols):
            # Extrair região
            cell = img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            
            # Calcular média e desvio padrão de cor para cada canal
            for channel in range(3):
                features.append(np.mean(cell[:,:,channel]))
                features.append(np.std(cell[:,:,channel]))
    
    return np.array(features)

# Carregar imagens e extrair características
print("\nExtraindo características das imagens de treino...")
X_train = []
for img_id in train_df['example_id']:
    img = load_image(img_id)
    features = extract_features(img)
    X_train.append(features)

X_train = np.array(X_train)
y_train = train_df[defect_columns].values

# Dividir em conjunto de treinamento e validação
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
)

# Definir parâmetros para GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

# Otimização de hiperparâmetros usando GridSearchCV
models = {}
best_params = {}
model_dir = os.path.join(DATA_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

# Verificar se já existem modelos treinados
model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
if len(model_files) == len(defect_columns):
    print("\nCarregando modelos treinados...")
    for i, defect in enumerate(defect_columns):
        model_path = os.path.join(model_dir, f"{defect}_model.joblib")
        if os.path.exists(model_path):
            models[defect] = load(model_path)
            print(f"Modelo para {defect} carregado.")
else:
    print("\nTreinando modelos com GridSearchCV...")
    for i, defect in enumerate(defect_columns):
        print(f"\nOtimizando hiperparâmetros para {defect}...")
        start_time = time.time()
        
        # Criar o GridSearchCV
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid=param_grid,
            cv=3,  # Número de folds
            scoring='f1',  # Métrica de avaliação
            n_jobs=-1  # Usar todos os núcleos disponíveis
        )
        
        # Treinar o modelo
        grid_search.fit(X_train_split, y_train_split[:, i])
        
        # Guardar melhor modelo e parâmetros
        best_model = grid_search.best_estimator_
        models[defect] = best_model
        best_params[defect] = grid_search.best_params_
        
        # Salvar o modelo
        model_path = os.path.join(model_dir, f"{defect}_model.joblib")
        dump(best_model, model_path)
        
        # Mostrar resultados
        print(f"Melhores parâmetros para {defect}: {grid_search.best_params_}")
        print(f"Melhor pontuação: {grid_search.best_score_:.4f}")
        print(f"Tempo de treinamento: {time.time() - start_time:.2f} segundos")
        
        # Avaliar no conjunto de validação
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val[:, i], y_pred)
        print(f"Acurácia na validação: {accuracy:.4f}")
        print("Relatório de classificação:")
        print(classification_report(y_val[:, i], y_pred))

# Extrair características das imagens de teste
print("\nExtraindo características das imagens de teste...")
X_test = []
for img_id in test_df['example_id']:
    img = load_image(img_id)
    features = extract_features(img)
    X_test.append(features)

X_test = np.array(X_test)

# Fazer previsões
print("\nFazendo previsões...")
predictions = np.zeros((len(test_df), len(defect_columns)), dtype=int)

for i, defect in enumerate(defect_columns):
    model = models[defect]
    # Usar predict diretamente para obter valores binários (0 ou 1)
    predictions[:, i] = model.predict(X_test)

# Criar DataFrame de submissão
submission = test_df.copy()
for i, col in enumerate(defect_columns):
    submission[col] = predictions[:, i]

# Salvar a submissão
submission.to_csv('submission_first.csv', index=False)
print("\nSubmissão salva em 'submission.csv'")

# Visualizar algumas imagens e previsões
def visualize_predictions(num_samples=3):
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(min(num_samples, len(train_df))):
        # Selecionar aleatoriamente uma imagem com defeitos
        with_defect = train_df[train_df['has_deffect'] == 1]
        sample = with_defect.sample(1).iloc[0]
        img_id = sample['example_id']
        img = load_image(img_id)
        
        # Mostrar imagem e defeitos
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"ID: {img_id}")
        plt.axis('off')
        
        # Mostrar defeitos
        defects = [defect for defect, val in sample[defect_columns].items() if val == 1]
        defect_str = ", ".join(defects)
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.axis('off')
        plt.text(0.1, 0.5, f"Defeitos: {defect_str}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizacao_defeitos.png')
    print("\nVisualização salva em 'visualizacao_defeitos.png'")

try:
    visualize_predictions()
except Exception as e:
    print(f"Erro ao criar visualização: {e}")

print("\nProcesso concluído!")