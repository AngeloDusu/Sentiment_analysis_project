# 🧠 Sentiment Analysis Project

This project focuses on classifying tweets into sentiment categories using both traditional machine learning and deep learning techniques.  
It was designed to strengthen my understanding of data preprocessing, SQL integration, and model development for Natural Language Processing (NLP).

---

## 📊 1. Data Collection

The dataset used in this project comes from Hugging Face:  
➡️ [Sentiment Analysis Tweet Dataset](https://huggingface.co/datasets/LYTinn/sentiment-analysis-tweet)

To enhance data handling skills, I uploaded the dataset into a **SQL database** and interacted with it through Python.  
This approach helped me understand how to manage and query large datasets effectively.

---

## 🧹 2. Data Cleaning

Both the **train** and **test** datasets were cleaned using several Python scripts.  
Key cleaning steps included:

- Removing duplicates and irrelevant entries
- Handling missing values
- Text normalization (lowercasing, punctuation removal, stopwords, etc.)
- Label encoding for sentiment categories

---

## 🔍 3. Exploratory Data Analysis (EDA)

The following libraries were used for visualization and exploration:

- **Pandas**
- **Matplotlib**
- **Seaborn**

Through EDA, I gained insights into:

- The sentiment distribution in the dataset
- Word frequency and most common terms
- Data imbalance
- Potential data leaks (which were identified and fixed)

---

## 🤖 4. Modeling / Machine Learning

### 🔧 Libraries Used:

- **Scikit-learn**
- **TensorFlow / Keras**
- **FastText**
- **Pandas, NumPy**

### 🧩 Models Developed:

1. **Logistic Regression (Baseline)**  
   A simple ML model used as a performance benchmark.  
   → Result: Weak performance, even after tuning.

2. **LSTM (Random Embedding)**  
   LSTM model using randomly initialized embeddings.  
   → Result: Decent accuracy, but could be improved.

3. **LSTM (Pretrained FastText Embedding)**  
   LSTM model with pretrained FastText word embeddings, then fine-tuned with GridSearchCV.  
   → Result: Best overall performance (~0.49 accuracy).  
   However, excessive tuning caused overfitting.

---

## 📈 5. Results Summary

| Model                      | Type             | Accuracy | Notes                                 |
| -------------------------- | ---------------- | -------- | ------------------------------------- |
| Logistic Regression        | Machine Learning | ~0.35    | Weak baseline                         |
| LSTM (Random Embedding)    | Deep Learning    | ~0.48    | Moderate performance                  |
| LSTM (Pretrained FastText) | Deep Learning    | ~0.49    | Best performance; risk of overfitting |

---

## 🧠 6. Key Learnings

Throughout this project, I learned how to:

- Connect and query **SQL databases** from Python
- Build and compare **machine learning vs deep learning** models for sentiment analysis
- Use **GridSearchCV** for hyperparameter tuning
- Integrate **FastText** embeddings for better text representation

This project also showed me that LSTM isn’t only useful for forecasting — it can also handle **text classification** effectively when combined with proper embeddings.

---

## 🚀 7. Future Improvements

Some potential improvements for this project:

- Experiment with **Bidirectional LSTM (BiLSTM)** or **GRU**
- Implement **transformer-based models** like BERT
- Improve preprocessing by using **lemmatization** and **domain-specific stopwords**
- Use **cross-validation** for more stable results
- Add a small **Streamlit dashboard** for real-time sentiment prediction

---

## 📁 8. Project Files

You can access the full project folder (data, notebooks, and models) here:  
🔗 [Google Drive Folder](https://drive.google.com/drive/folders/1SmLIWPqP3YJjJ6HCfrEtY2vQIGtiySlJ?usp=sharing)

---

## 🧑‍💻 9. Author

**Syahrul Angelo Aria Dusu**  
📍 Data Science & Analytics Enthusiast  
📫 [GitHub Portfolio](https://angelodusu.github.io/Portofolio/)  
💼 [LinkedIn](https://www.linkedin.com/in/syahrul-angelo-aria-dusu-2610ilo/)  
📧 2610angelo@gmail.com

---

⭐ _If you found this project interesting, feel free to give it a star on GitHub!_ ⭐
