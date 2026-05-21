import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Загрузка данных...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")


# ========== ПРОСТЫЕ ПРИЗНАКИ ==========
def prepare_features(df, is_train=True):
    df = df.copy()

    # Пол
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Возраст (заполняем пропуски медианой)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Цена билета (заполняем пропуски медианой)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Порт посадки (заполняем самым частым)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Размер семьи (SibSp + Parch)
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Один ли плыл?
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    # Выбираем только нужные признаки
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

    return df[features]


print("Подготовка признаков...")
X_train = prepare_features(train)
X_test = prepare_features(test)
y_train = train['Survived']

print(f"Признаков: {X_train.shape[1]}")

# ========== ОБУЧЕНИЕ ==========
print("Обучение Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ========== ПРЕДСКАЗАНИЕ ==========
test_pred = model.predict(X_test)

# ========== СОЗДАНИЕ ФАЙЛА ДЛЯ KAGGLE ==========
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('titanic_submission.csv', index=False)

print("✅ Готово! titanic_submission.csv создан")
print(submission.head())

# Проверка точности на тренировочных данных
train_pred = model.predict(X_train)
train_acc = (train_pred == y_train).mean()
print(f"Точность на обучении: {train_acc:.2%}")