import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import pickle

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

prepare_params = params['prepare']

# Загружаем данные
print('Загрузка данных...')
df = pd.read_csv('data/raw/wine.csv')

# Разделяем на признаки и целевую переменную
X = df.drop('target', axis=1)
y = df['target']

# Разделяем на train/test
print(f'Разделение данных (test_size={prepare_params[\"test_size\"]})...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=prepare_params['test_size'],
    random_state=prepare_params['random_state'],
    stratify=y
)

# Нормализация данных
print('Нормализация данных...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создаем директорию для обработанных данных
os.makedirs('data/processed', exist_ok=True)

# Сохраняем обработанные данные
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    'data/processed/X_train.csv', index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    'data/processed/X_test.csv', index=False
)
pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

# Сохраняем scaler
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Сохраняем метаданные
metadata = {
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': X_train.shape[1],
    'n_classes': len(y.unique())
}

with open('data/processed/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('\n' + '='*60)
print('Подготовка завершена!')
print('='*60)
print(f'Train samples: {metadata[\"train_samples\"]}')
print(f'Test samples: {metadata[\"test_samples\"]}')
print(f'Features: {metadata[\"n_features\"]}')
print(f'Classes: {metadata[\"n_classes\"]}')
