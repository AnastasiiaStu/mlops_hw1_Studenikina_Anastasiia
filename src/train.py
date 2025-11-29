import pandas as pd
import yaml
import pickle
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

train_params = params['train']

# Загружаем подготовленные данные
print('Загрузка подготовленных данных...')
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Загружаем метаданные
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Train samples: {len(X_train)}')
print(f'Test samples: {len(X_test)}')

# Настраиваем MLflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('wine_classification')

# Начинаем эксперимент MLflow
with mlflow.start_run():
    
    # Логируем параметры
    mlflow.log_params(train_params)
    mlflow.log_params(metadata)
    
    # Создаем модель
    print(f'\nОбучение модели: {train_params[\"model_type\"]}')
    
    if train_params['model_type'] == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=train_params['n_estimators'],
            max_depth=train_params['max_depth'],
            random_state=train_params['random_state'],
            n_jobs=-1
        )
    elif train_params['model_type'] == 'LogisticRegression':
        model = LogisticRegression(
            random_state=train_params['random_state'],
            max_iter=1000
        )
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Делаем предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Вычисляем метрики
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f'\nТочность на train: {train_accuracy:.4f}')
    print(f'Точность на test: {test_accuracy:.4f}')
    print(f'F1-score на test: {test_f1:.4f}')
    
    # Логируем метрики
    mlflow.log_metric('train_accuracy', train_accuracy)
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)
    
    # Создаем и сохраняем confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Логируем график
    mlflow.log_artifact('confusion_matrix.png')
    
    # Сохраняем модель локально
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Логируем модель в MLflow
    mlflow.sklearn.log_model(model, 'model')
    
    # Логируем артефакт
    mlflow.log_artifact('model.pkl')
    
    # Сохраняем отчет о классификации
    report = classification_report(y_test, y_test_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    
    print('\n' + '='*50)
    print('Classification Report:')
    print(report)
    print('='*50)
    
    print('\nМодель обучена и сохранена!')
    print('Артефакты залогированы в MLflow')
