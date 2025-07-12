# JobGenie
---

# 💼 Job Recommendation System

A smart, AI-powered Job Recommendation System that matches candidates with suitable job listings based on their resume, skills, experience, and job descriptions using advanced Natural Language Processing techniques.

---

## 🚀 Features

- Resume and job description parsing using NLP
- Semantic matching using **BERT-based embeddings**
- Skill, role, and experience-level alignment
- Ranked job recommendations based on fit score
- Interactive interface built with **Streamlit**

---

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data handling
- **NLTK**, **spaCy**, **transformers** – NLP & text preprocessing
- **Sentence-BERT** – Semantic text similarity
- **scikit-learn** – Feature extraction & similarity metrics
- **Streamlit** – Web app UI
- **Git/GitHub** – Version control

---

## 📁 Project Structure

job-recommendation-system/
│
├── data/ # Sample resumes and job listings
├── models/ # Pretrained and fine-tuned models
├── app/ # Streamlit frontend
│ ├── main.py # Streamlit app
│ └── utils.py # Helper functions
├── notebooks/ # Jupyter notebooks for development
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## 🛠️ How It Works

1. **Input:** User uploads a resume or enters skill details  
2. **Preprocessing:** Text is cleaned, tokenized, and embedded using Sentence-BERT  
3. **Matching:** Each resume is compared to job descriptions using cosine similarity  
4. **Ranking:** Jobs are ranked by relevance score and displayed to the user

---

## 📊 Use Cases

- Job portals (e.g., LinkedIn, Indeed, Internshala)
- Campus placement tools
- Career guidance platforms
- Resume shortlisting automation

---

## 🧪 Future Improvements

- Add support for multiple job categories  
- Integrate real-time job scraping (from LinkedIn or Indeed APIs)  
- Include candidate feedback loop for model tuning  
- Extend model with multi-language support

---

## 👨‍💻 Author

**Yash Bagga**  
[LinkedIn](https://linkedin.com/in/yash-bagga-a32b1a256) | [GitHub](https://github.com/yashbagga5)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
