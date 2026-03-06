from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample author dataset
docs = [
    "Freedom and liberty must guide society",
    "The most violent element in society is ignorance",
    "True freedom requires courage and independence",

    "The Arctic expedition was long and dangerous",
    "Exploration requires determination and teamwork",
    "The frozen land tested human endurance",

    "Diplomacy maintains peace between nations",
    "International cooperation strengthens relations",
    "Countries must work together for stability"
]

labels = [
    "Emma Goldman",
    "Emma Goldman",
    "Emma Goldman",

    "Matthew Henson",
    "Matthew Henson",
    "Matthew Henson",

    "TingFang Wu",
    "TingFang Wu",
    "TingFang Wu"
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    docs, labels, test_size=0.3, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train classifier
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Evaluate model
predictions = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Mystery text prediction
mystery_text = input("\nEnter a mystery message:\n> ")

mystery_vector = vectorizer.transform([mystery_text])

prediction = model.predict(mystery_vector)[0]
probabilities = model.predict_proba(mystery_vector)

print("\nPredicted Author:", prediction)
print("Prediction Probabilities:", probabilities)


