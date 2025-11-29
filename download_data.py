from sklearn.datasets import load_wine
import pandas as pd
import os

# Загружаем датасет Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Сохраняем
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/wine.csv', index=False)

print('='*60)
print('✓ Датасет загружен и сохранен в data/raw/wine.csv')
print('='*60)
print(f'Размер: {df.shape}')
print(f'\nПервые строки:\n{df.head()}')
print(f'\nРаспределение классов:\n{df["target"].value_counts()}')
