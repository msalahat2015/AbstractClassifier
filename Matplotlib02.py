import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# 1. Load learning curve data
# -------------------------------
# ملفات CSV يجب أن تحتوي: epoch,accuracy,loss,val_accuracy,val_loss
mlp_csv = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/base/mlp_learning_curves.csv"
cnn_csv = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/Target/cnn_learning_curves.csv"

mlp_df = pd.read_csv(mlp_csv)
cnn_df = pd.read_csv(cnn_csv)

# -------------------------------
# 2. Plot learning curves (Accuracy)
# -------------------------------
plt.figure(figsize=(10,5))

# Training Accuracy
plt.plot(mlp_df['epoch'], mlp_df['accuracy'], label='MLP Train Acc', color='tab:blue', linestyle='--')
plt.plot(cnn_df['epoch'], cnn_df['accuracy'], label='CNN Train Acc', color='tab:orange', linestyle='--')

# Validation Accuracy
plt.plot(mlp_df['epoch'], mlp_df['val_accuracy'], label='MLP Val Acc', color='tab:blue')
plt.plot(cnn_df['epoch'], cnn_df['val_accuracy'], label='CNN Val Acc', color='tab:orange')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/learning_curves_mlp_cnn.png")
plt.close()

# -------------------------------
# 3. Plot learning curves (Loss)
# -------------------------------
plt.figure(figsize=(10,5))

# Training Loss
plt.plot(mlp_df['epoch'], mlp_df['loss'], label='MLP Train Loss', color='tab:blue', linestyle='--')
plt.plot(cnn_df['epoch'], cnn_df['loss'], label='CNN Train Loss', color='tab:orange', linestyle='--')

# Validation Loss
plt.plot(mlp_df['epoch'], mlp_df['val_loss'], label='MLP Val Loss', color='tab:blue')
plt.plot(cnn_df['epoch'], cnn_df['val_loss'], label='CNN Val Loss', color='tab:orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/loss_curves_mlp_cnn.png")
plt.close()

# -------------------------------
# 4. Load merged Classification Reports (F1-scores)
# -------------------------------
# ملف CSV الناتج عن دمج تقارير MLP وCNN
# يجب أن يحتوي الأعمدة: category,f1_mlp,f1_cnn
f1_csv = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/classification_reports_merged.csv"
f1_df = pd.read_csv(f1_csv)

# استخدم فقط الأعمدة المطلوبة للـ F1-score
f1_df = f1_df[['category', 'f1_mlp', 'f1_cnn']]

# -------------------------------
# 5. Plot F1-score Heatmap
# -------------------------------
plt.figure(figsize=(12,6))
f1_matrix = f1_df.set_index('category')[['f1_mlp','f1_cnn']]
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("F1-score Comparison: MLP vs CNN")
plt.ylabel("Class")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/matplotlib/f1_heatmap_mlp_vs_cnn.png")
plt.close() 
