# AI Task-Management System

An intelligent task prioritization system that uses machine learning to automatically analyze and categorize tasks based on their content, urgency, and importance.

## Author
**Faizan Ayoub Hajam**  
Student at Amity University Noida

## Overview
This project implements a machine learning-based system that automatically analyzes task descriptions and assigns priority levels. It uses Natural Language Processing (NLP) and machine learning techniques to understand task context and determine appropriate priority levels.

## Features
- 🤖 **AI-Powered Analysis**: Uses advanced machine learning algorithms to analyze task descriptions
- ⚡ **Real-time Processing**: Get instant priority suggestions as you type
- 📊 **Smart Prioritization**: Considers multiple factors including urgency, importance, and complexity
- 🎯 **Accurate Predictions**: Trained on thousands of task descriptions for better accuracy
- 🔒 **Secure & Scalable**: Built with security and scalability in mind

## Technical Stack
- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, TF-IDF Vectorization
- **API**: RESTful API with CORS support
- **Deployment**: Docker, Gunicorn
- **Frontend**: HTML, CSS, JavaScript

## Project Structure
```
backend/
├── app.py              # Main Flask application
├── model_processor.py  # ML model handling
├── models/            # Trained model files
├── templates/         # HTML templates
├── requirements.txt   # Python dependencies
├── Dockerfile        # Docker configuration
└── docker-compose.yml # Docker compose configuration
```

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

### Local Development Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Task-Manager-UI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

### Docker Deployment
1. Build and run using Docker Compose:
   ```bash
   cd backend
   docker-compose up --build
   ```

2. Access the application at `http://localhost:8080`

## API Documentation

### Predict Task Priority
```http
POST /api/predict
Content-Type: application/json

{
    "task_text": "Your task description here"
}
```

Response:
```json
{
    "prediction": "Predicted Priority: [HIGH/MEDIUM/LOW]"
}
```

## Model Details
The system uses a combination of:
- TF-IDF Vectorization for text processing
- Machine Learning classifier for priority prediction
- Trained on a diverse dataset of task descriptions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Amity University Noida for academic support
- Open source community for various tools and libraries used in this project

## Contact
For any queries or suggestions, please reach out to:
- Email: [faizanayub16@gmail.com]
- LinkedIn: [www.linkedin.com/in/faizan-hajam]
- GitHub: [https://github.com/faizanHajam120]
