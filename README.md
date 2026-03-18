# 🚀 NIBRAS AI Recommendation System : 

An intelligent AI-powered recommendation system designed for the NIBRAS academic platform to help Computer Science students choose the most suitable specialization track based on their skills and capabilities.

---

## 🎯 Project Goal : 

This project helps students make **data-driven decisions** when choosing their specialization instead of relying on random choices or peer influence.

The system analyzes a student's capabilities and recommends the most suitable academic track using Machine Learning and AI explanations.

---

## 🧠 How It Works : 

The system uses a **hybrid approach**:

### 1️⃣ Capability-Based Input
Students provide their skill levels (0 → 1) in areas like:
- Programming
- Algorithms
- Math
- Theory
- Data
- Systems
- Hardware
- AI
- UX
- Security
- Graphics
- Biology

---

### 2️⃣ Feature Engineering
- Converts student capabilities into numerical vectors  
- Computes **cosine similarity** with predefined track profiles  
- Combines:Capabilities + Similarity Features
  
---

### 3️⃣ Machine Learning Model
- Model: XGBoost Classifier  
- Output: Probability distribution over tracks  
- Top 3 tracks are selected based on highest probabilities  

---

### 4️⃣ AI Explanation (LLM)
- Uses OpenAI API to generate:
- Why a track fits the student  
- Key strengths influencing the decision  
- Personalized advice  

---
## 🛠️ Technologies Used : 

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost   
- OpenAI API  

---

## 📊 Dataset : 

- Simulated dataset of student capabilities  
- Each student represented by normalized skill values (0 → 1)  
- Includes labeled specialization tracks  

---
