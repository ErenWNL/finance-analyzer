# Finance Analyzer

A comprehensive financial analysis tool that provides transaction categorization, budget optimization, and spending pattern predictions using machine learning.

## Features

- Transaction categorization using NLP (Natural Language Processing)
- Budget optimization with reinforcement learning (RL)
- Spending pattern prediction using LSTM (Long Short-Term Memory) neural networks
- Interactive visualizations using Recharts
- Real-time analysis and insights
- Firebase authentication and data storage
- MongoDB for transaction data management

## Technical Stack

### Frontend
- React 18 with Vite
- Material-UI (MUI) for UI components
- Recharts for data visualization
- Firebase for authentication and storage
- TailwindCSS for styling
- Axios for API communication

### Backend
- Python 3.8+
- Flask for REST API
- PyTorch for ML models
- Scikit-learn for data processing
- NLTK for NLP tasks
- MongoDB for database
- Firebase Admin SDK

## Prerequisites

### For Windows
- Node.js v18 or higher
- Python 3.8 or higher
- MongoDB Community Edition
- Git Bash (recommended for running shell scripts)
- Visual Studio Build Tools (for Python packages)

### For macOS
- Node.js v18 or higher
- Python 3.8 or higher
- MongoDB Community Edition
- Xcode Command Line Tools (for Python packages)

## Installation

### Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Create a `.env.local` file in the root directory with your Firebase configuration:
```
VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_auth_domain
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_storage_bucket
VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
VITE_FIREBASE_APP_ID=your_app_id
```

### Backend Setup

#### Windows
1. Open Command Prompt or Git Bash and navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory:
```
MONGODB_URI=mongodb://localhost:27017/finance_analyzer
FIREBASE_CREDENTIALS=path/to/your/firebase-credentials.json
MODEL_PATH=./models/
LOG_LEVEL=INFO
```

#### macOS
1. Open Terminal and navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

4. Create a `.env` file in the backend directory:
```
MONGODB_URI=mongodb://localhost:27017/finance_analyzer
FIREBASE_CREDENTIALS=path/to/your/firebase-credentials.json
MODEL_PATH=./models/
LOG_LEVEL=INFO
```

## Running the Application

### Start MongoDB

#### Windows
1. Open Command Prompt as Administrator
2. Navigate to MongoDB bin directory:
```bash
cd "C:\Program Files\MongoDB\Server\<version>\bin"
```
3. Start MongoDB service:
```bash
mongod --dbpath="C:\data\db"
```

#### macOS
1. Open Terminal
2. Start MongoDB service:
```bash
brew services start mongodb-community
```

### Start the Backend

#### Windows
1. Open Git Bash and run:
```bash
./start_backend.sh
```
Or manually:
```bash
cd backend
python app.py
```

#### macOS
1. Open Terminal and run:
```bash
chmod +x start_backend.sh
./start_backend.sh
```
Or manually:
```bash
cd backend
python3 app.py
```

### Start the Frontend

#### Windows
1. Open a new Command Prompt or Git Bash
2. From the project root:
```bash
npm run dev
```

#### macOS
1. Open a new Terminal
2. From the project root:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Project Structure

- `backend/`: Python backend with ML models and API
  - `app.py`: Main Flask application (REST API endpoints)
  - `transaction_categorizer.py`: NLP-based transaction categorization using NLTK
  - `rl_budget_optimizer.py`: Budget optimization using PyTorch RL
  - `lstm_predictor.py`: Spending pattern prediction using PyTorch LSTM
  - `models/`: Trained ML models and weights
  - `logs/`: Application logs
- `src/`: React frontend application
  - `components/`: Reusable React components
  - `pages/`: Application pages
  - `services/`: API services
  - `utils/`: Utility functions
- `public/`: Static assets
- `node_modules/`: Frontend dependencies
- `venv/`: Python virtual environment

## Development

### Frontend Development
- Development server: `npm run dev`
- Build: `npm run build`
- Linting: `npm run lint`
- Preview: `npm run preview`

### Backend Development
- Development server: `python app.py` (Windows) or `python3 app.py` (macOS)
- Logs: Located in `backend/logs/`
- API Documentation: Available at `http://localhost:5000/docs` when running

## Troubleshooting

### Common Issues

#### Windows
- If `start_backend.sh` fails, ensure Git Bash is installed and the script has proper line endings (LF)
- If Python packages fail to install, ensure Visual Studio Build Tools are installed
- If MongoDB fails to start, ensure the data directory exists at `C:\data\db`

#### macOS
- If MongoDB fails to start, ensure it's properly installed via Homebrew
- If Python packages fail to install, ensure Xcode Command Line Tools are installed
- If permission errors occur, use `sudo` with appropriate commands

## License

This project is licensed under the terms of the LICENSE file in the root directory.

## Support

For any issues or questions, please open an issue in the repository.
