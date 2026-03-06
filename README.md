Author Identification NLP 
This project uses Natural Language Processing and Machine Learning to identify the most likely author of a piece of text. The model is trained on writing samples from different authors and predicts which author wrote a mystery message.

## Project Overview

The system converts text into numerical vectors using the **Bag-of-Words model** and then trains a **Multinomial Naive Bayes classifier** to recognize writing patterns.

This type of problem is known as **author attribution** and is commonly used in stylometry and forensic linguistics.

## Technologies Used

- Python
- scikit-learn
- CountVectorizer (Bag of Words)
- Multinomial Naive Bayes

## Machine Learning Workflow

1. Collect writing samples from different authors
2. Convert text into numerical vectors using Bag of Words
3. Train a Naive Bayes classifier
4. Predict the author of a mystery text

## Example

Input:

"Freedom and liberty must guide society."

Output:

Predicted Author: Emma Goldman

## Skills Demonstrated

- Natural Language Processing
- Text Vectorization
- Machine Learning Classification
- scikit-learn pipeline

## Future Improvements

- Use TF-IDF instead of Bag-of-Words
- Add more authors and training data
- Use more advanced NLP models
