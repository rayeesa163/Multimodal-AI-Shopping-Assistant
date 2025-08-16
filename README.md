 🛍️ Multimodal AI Shopping Assistant

A cloud-deployed AI web application that delivers personalized shopping recommendations by analyzing user emotions and textual queries in real-time.

🚀 Overview

The Multimodal AI Shopping Assistant is designed to enhance online shopping experiences by integrating multimodal inputs—text and emotion. Leveraging cutting-edge AI technologies, this assistant provides tailored product suggestions based on the user's emotional state and intent.

🧠 Key Features

Emotion Detection: Utilizes DeepFace to capture and interpret user emotions via webcam.

Intent Understanding: Employs BERT-based Sentence Transformers to comprehend user queries.

Product Recommendations: Generates personalized suggestions by combining emotional and textual analysis.

Real-Time Interaction: Offers instantaneous responses for an engaging user experience.

Cloud Deployment: Ensures accessibility and scalability through cloud hosting.

🧩 Technologies Used

DeepFace: For facial emotion recognition.

Sentence Transformers (BERT): To process and understand user input.

Flask: Framework for building the web application.

HTML/CSS/JavaScript: For frontend development.

Python: Backend logic and AI model integration.
GitHub
+1
Medium
GitHub
+4
Medium
+4
activeloop.ai
+4

📦 Installation & Setup

Clone the Repository:

git clone https://github.com/rayeesa163/Multimodal-AI-Shopping-Assistant.git
cd Multimodal-AI-Shopping-Assistant


Set Up Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Dependencies:

pip install -r requirements.txt


Run the Application:

python app.py


Access the app at http://127.0.0.1:5000 in your browser.

📄 File Descriptions

app.py: Main application file that runs the Flask server.

emotion_detector.py: Handles real-time emotion detection from webcam input.

recommender.py: Generates product recommendations based on processed inputs.

voice_to_text.py: Converts spoken language into text for analysis.

products.csv: Dataset containing product information for recommendations.
arXiv

🎥 Demo

Experience the assistant in action: https://youtu.be/djZrt2Tf4zI?si=kI71eEwAWVbrJjHa

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
