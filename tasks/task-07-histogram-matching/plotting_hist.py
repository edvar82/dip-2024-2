import cv2 as cv
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

# Carrega as imagens
source_img = cv.imread("source.jpg")
reference_img = cv.imread("reference.jpg")

# Converte para RGB
source_rgb = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
reference_rgb = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)

# Aplica histogram matching (Scikit-Image)
matched_rgb = exposure.match_histograms(source_rgb, reference_rgb, channel_axis=2)

# Retorna a imagem igualada para BGR
output_image = cv.cvtColor(matched_rgb, cv.COLOR_RGB2BGR)

# Salva o resultado
cv.imwrite("output_achieved.jpg", output_image)

# Carrega as imagens
source_img = cv.imread("source.jpg")
reference_img = cv.imread("reference.jpg")
output_image = cv.imread("output_achieved.jpg")
output_expected = cv.imread("output.jpg")

# Separa canais das imagens
b_r, g_r, r_r = cv.split(reference_img)
b_o, g_o, r_o = cv.split(output_image)
b_s, g_s, r_s = cv.split(source_img)
b_oe, g_oe, r_oe = cv.split(output_expected)

# Configura figura e eixos
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
fig.suptitle('Histogramas de Imagens', fontsize=16)

# Ajusta o espaço entre os subplots
fig.subplots_adjust(hspace=0.5, wspace=0.4)

# Histograma da imagem source
axes[0,0].hist(b_s.ravel(), 256, [0,256], color='b')
axes[0,0].set_title('Source - B')
axes[0,1].hist(g_s.ravel(), 256, [0,256], color='g')
axes[0,1].set_title('Source - G')
axes[0,2].hist(r_s.ravel(), 256, [0,256], color='r')
axes[0,2].set_title('Source - R')

# Histograma da imagem reference
axes[1,0].hist(b_r.ravel(), 256, [0,256], color='b')
axes[1,0].set_title('Reference - B')
axes[1,1].hist(g_r.ravel(), 256, [0,256], color='g')
axes[1,1].set_title('Reference - G')
axes[1,2].hist(r_r.ravel(), 256, [0,256], color='r')
axes[1,2].set_title('Reference - R')

# Histograma do output_achieved
axes[2,0].hist(b_o.ravel(), 256, [0,256], color='b')
axes[2,0].set_title('Output_achieved - B')
axes[2,1].hist(g_o.ravel(), 256, [0,256], color='g')
axes[2,1].set_title('Output_achieved - G')
axes[2,2].hist(r_o.ravel(), 256, [0,256], color='r')
axes[2,2].set_title('Output_achieved - R')

# Histograma do output_expected
axes[3,0].hist(b_oe.ravel(), 256, [0,256], color='b')
axes[3,0].set_title('Output_expected - B')
axes[3,1].hist(g_oe.ravel(), 256, [0,256], color='g')
axes[3,1].set_title('Output_expected - G')
axes[3,2].hist(r_oe.ravel(), 256, [0,256], color='r')
axes[3,2].set_title('Output_expected - R')

# Salvar a imagem
plt.savefig("histogramas_comparacao_1.png")

plt.tight_layout()
plt.show()