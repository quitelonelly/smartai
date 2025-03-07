import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Исправлено: feature_extlection -> feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from nltk.stem import SnowballStemmer

# Загрузка данных
data = pd.read_csv('dialogues.csv')

# Предобработка текста
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Разделение на обучающую и тестовую выборки
X = data['cleaned_text']
y = data['role']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизация текста
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Балансировка данных с помощью SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Обучение модели (Random Forest вместо Logistic Regression)
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Оценка модели
y_pred = model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# Функция для предсказания роли
def predict_role(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]