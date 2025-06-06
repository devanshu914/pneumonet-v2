# PneumoNet-V2 🔬🫁

PneumoNet-V2 is an AI-powered pneumonia detection system built using deep learning. It analyzes chest X-ray images and predicts the likelihood of pneumonia with the help of a pre-trained EfficientNetB0 model. The system is designed as a Flask web application with a clean frontend and easy-to-use interface.

## 🌐 Live Demo

🚀 [Click here to try PneumoNet-V2](https://pneumonet-v2-2.onrender.com/)

## 📸 Features

- ✅ Upload chest X-ray images
- 🧠 Deep Learning-based pneumonia detection using EfficientNetB0
- 📊 Real-time prediction results
- 🌐 Flask backend with HTML/CSS/JS frontend
- 📁 Easily extendable and customizable

---

## 🛠️ Tech Stack

| Layer        | Tech Used                        |
|-------------|-----------------------------------|
| Frontend     | HTML, CSS, JavaScript            |
| Backend      | Python, Flask                    |
| AI Model     | TensorFlow / Keras (EfficientNetB0) |
| Deployment   | Render (free tier)               |

---̨

## 📂 Project Structure

```bash
PneumoNet_V2_Project/
│
├── app.py                  # Main Flask app
├── model/
│   └── my_model_checkpoint.keras  # Trained model
├── static/                 # CSS, JS, images
├── templates/              # HTML templates
├── requirements.txt        # Python dependencies
├── Procfile                # For Render deployment
└── README.md               # Project info
