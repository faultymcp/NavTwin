# NavTwin 🧭

An AI-powered navigation application designed specifically for neurodivergent users experiencing navigation anxiety. NavTwin uses a dual digital twin framework with integrated AI components to provide personalized, stress-reducing route guidance.

## 🎯 Project Overview

NavTwin is a passion project that addresses the unique navigation challenges faced by neurodivergent individuals. By combining sensory profile analysis, personalized route planning, and gamification elements, the app transforms navigation from a source of anxiety into a manageable, even enjoyable experience.

## ✨ Key Features

- **🧠 AI-Powered Personalization**: Five integrated AI components analyze user preferences and sensory sensitivities
- **🗺️ Smart Route Planning**: Personalized routes based on individual sensory profiles
- **🎮 Gamification**: Engagement elements designed to reduce travel stress
- **🔍 Natural Address Search**: Intuitive search functionality for easy destination input
- **🔐 Secure Authentication**: User profile management and data protection
- **📱 Cross-Platform**: Built with React Native for iOS and Android

## 🛠️ Tech Stack

### Frontend
- React Native
- JavaScript/TypeScript
- Google Maps API integration

### Backend
- Python
- FastAPI
- Machine Learning models for personalization

### Architecture
- Dual Digital Twin Framework
- RESTful API design
- Real-time route optimization

## 📋 Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- Google Maps API key
- npm or yarn

## 🚀 Installation

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

## ⚙️ Configuration

Create a `.env` file in the root directory:
```env
GOOGLE_MAPS_API_KEY=your_api_key_here
BACKEND_URL=http://localhost:8000
```

## 📱 Usage

1. **Create Profile**: Set up your sensory preferences and navigation anxiety triggers
2. **Enter Destination**: Use natural language or address search
3. **Review Route**: See personalized route options tailored to your comfort
4. **Navigate**: Follow turn-by-turn guidance with anxiety-reducing features
5. **Track Progress**: Earn rewards and build confidence through gamification

## 🏗️ Architecture

NavTwin implements a dual digital twin framework:

1. **User Digital Twin**: Models user preferences, sensory profiles, and behavioral patterns
2. **Environment Digital Twin**: Represents real-world navigation contexts and conditions

These twins work together with five AI components to deliver personalized navigation experiences.

## 🎓 Academic Context

This project is submitted as part of a Masters in Artificial Intelligence thesis, exploring how AI can create more inclusive and accessible navigation tools for neurodivergent populations.

**Submission Date**: December 1st, 2024

## 🤝 Contributing

This is an academic thesis project. Feedback and suggestions are welcome!


## 👤 Author

**Vyshnavi Muniganti** (@faultymcp)
- AI Masters Student
- Focus: Accessibility AI & Healthcare Technology

## 🙏 Acknowledgments

- Thesis supervisor and academic advisors
- Neurodivergent community for invaluable feedback
- Google Maps Platform for mapping services

---

*Built with ❤️ for a more accessible world*
