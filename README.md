# Fantasy Weapon Generator

An AI-powered tool that generates fantasy weapon names, descriptions, and attributes for tabletop RPGs and creative writing.

## Overview

This project uses a fine-tuned GPT-2 model to generate creative and detailed fantasy weapons based on user prompts. The system consists of three main components:

1. **Data Collection**: Scrapes weapon data from D&D API and generates synthetic fantasy weapon data
2. **Model Training**: Fine-tunes a GPT-2 model on the collected weapon data
3. **Web API**: Provides a user-friendly interface and API endpoints to interact with the model

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- CUDA-compatible GPU (recommended for training, not required for inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fantasy-weapon-generator.git
cd fantasy-weapon-generator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
fantasy-weapon-generator/
├── data/                       # Generated data storage
├── models/                     # Trained model storage
├── data_collection.py          # Data collection script
├── train_model.py              # Model training script
├── api.py                      # FastAPI web interface
├── templates/                  # HTML templates (auto-generated)
├── static/                     # Static assets (auto-generated)
│   └── css/                    # CSS stylesheets
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### Step 1: Data Collection

Run the data collection script to gather weapon data from D&D API and generate synthetic data:

```bash
python data_collection.py
```

This will:
- Fetch weapon data from the D&D 5e API
- Generate additional synthetic fantasy weapons
- Save the combined dataset to `data/weapons_raw.json`
- Format the data for training and save to `data/weapons_training.json` and `data/weapons_training.csv`

### Step 2: Model Training

Train the model using the collected data:

```bash
python train_model.py
```

This will:
- Prepare the training and validation datasets
- Fine-tune a DistilGPT-2 model on the weapon data
- Save the trained model to `models/gpt2-fantasy-weapons`
- Generate sample outputs to compare with the baseline model
- Save comparison results to `models/comparison_results.json`

### Step 3: Running the Web API

Start the FastAPI server:

```bash
python api.py
```

This will:
- Load the fine-tuned model
- Create necessary templates and static files
- Start a web server at http://localhost:8000

You can now:
- Access the web interface at http://localhost:8000
- Generate weapons using the API endpoints
- View system status at http://localhost:8000/status
- View request logs at http://localhost:8000/logs

For production deployment, use a proper ASGI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /`: Web interface
- `GET /status`: Check API status and model information
- `GET /generate?prompt=YOUR_PROMPT`: Generate weapon based on prompt with default parameters
- `POST /generate`: Generate weapon with customizable parameters

Example API call:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Generate a legendary frost bow", "temperature": 0.8, "max_length": 200}'
```

## Advanced Configuration

### Model Parameters

When generating weapons, you can adjust:

- `temperature`: Controls randomness (0.1-1.5, higher = more random)
- `max_length`: Maximum length of generated text (10-1000)
- `top_p`: Nucleus sampling parameter (0.1-1.0)
- `num_return_sequences`: Number of generations to return (1-5)

### Training Parameters

If you want to modify the training process, edit `train_model.py`:

- Adjust batch sizes
- Change learning rate
- Modify number of training epochs
- Change model architecture (default is DistilGPT-2)

## Troubleshooting

- **CUDA out of memory**: Reduce batch sizes or use a smaller model
- **API fails to start**: Check if the model was trained correctly
- **Poor generation quality**: Try adjusting temperature or re-train with more data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
