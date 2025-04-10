import os
import json
import time
import uuid
import torch
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import platform
import psutil
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/gpt2-fantasy-weapons"

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "distilgpt2"
    logger.warning(f"Fine-tuned model not found, using {MODEL_PATH} instead")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {MODEL_PATH}")
    app.state.start_time = time.time()
    app.state.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    
    if MODEL_PATH != "distilgpt2":
        special_tokens = {
            'additional_special_tokens': ['<|prompt|>', '<|completion|>', '<|endoftext|>']
        }
        app.state.tokenizer.add_special_tokens(special_tokens)
    
    app.state.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    app.state.model.to(device)
    app.state.model.eval()
    
    app.state.request_count = 0
    app.state.request_log = []
    
    logger.info("Model loaded successfully")
    
    yield
    
    logger.info("Shutting down API")
    del app.state.model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(
    title="Fantasy Weapon Generator API",
    description="API for generating fantasy weapon names and descriptions",
    version="1.0.0",
    lifespan=lifespan
)

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

css_dir = os.path.join(static_dir, "css")
os.makedirs(css_dir, exist_ok=True)

with open(os.path.join(css_dir, "style.css"), "w") as f:
    f.write("""
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 0;
    background-color: #f0f2f5;
}

.container {
    width: 90%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 0;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.8rem;
    font-weight: bold;
}

.tagline {
    font-size: 1rem;
    opacity: 0.8;
}

h1, h2, h3 {
    color: #2c3e50;
}

.card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 20px;
    margin-bottom: 20px;
}

.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="text"], select, textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

.result {
    margin-top: 30px;
    white-space: pre-wrap;
}

.weapon-title {
    color: #e74c3c;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.weapon-description {
    font-style: italic;
    color: #555;
}

.weapon-properties {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 15px 0;
}

.property-tag {
    background-color: #ecf0f1;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.9rem;
    color: #7f8c8d;
}

.log-entry {
    border-bottom: 1px solid #eee;
    padding: 10px 0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.stat-card {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-title {
    font-weight: bold;
    color: #7f8c8d;
    margin-bottom: 5px;
}

.stat-value {
    font-size: 1.5rem;
    color: #2c3e50;
}

footer {
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 1rem 0;
    margin-top: 4rem;
}

@media (max-width: 768px) {
    .container {
        width: 95%;
    }
    
    header .container {
        flex-direction: column;
        text-align: center;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
}
""")

with open(os.path.join(templates_dir, "index.html"), "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fantasy Weapon Generator</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">Fantasy Weapon Generator</div>
            <div class="tagline">AI-powered legendary items for your game</div>
        </div>
    </header>

    <div class="container">
        <div class="card">
            <h2>Generate Weapon</h2>
            <form id="generation-form" method="post" action="/generate-web">
                <div class="form-group">
                    <label for="prompt">Prompt:</label>
                    <input type="text" id="prompt" name="prompt" placeholder="e.g., Generate a legendary ice dagger" required>
                </div>
                <div class="form-group">
                    <label for="temperature">Temperature (0.1-1.5):</label>
                    <input type="range" id="temperature" name="temperature" min="0.1" max="1.5" step="0.1" value="0.8">
                    <span id="temperature-value">0.8</span>
                </div>
                <div class="form-group">
                    <label for="max_length">Max Length:</label>
                    <input type="range" id="max_length" name="max_length" min="50" max="500" step="10" value="200">
                    <span id="max_length-value">200</span>
                </div>
                <button type="submit">Generate</button>
            </form>
        </div>

        {% if result %}
        <div class="card result">
            <h2>Result</h2>
            <div class="weapon-result">
                {{ result | safe }}
            </div>
        </div>
        {% endif %}
        
        <div class="card">
            <h2>API Information</h2>
            <p>Use our API endpoints to generate weapons programmatically:</p>
            
            <h3>Endpoints:</h3>
            <ul>
                <li><code>GET /status</code> - Get API status and model information</li>
                <li><code>GET /generate?prompt=YOUR_PROMPT</code> - Generate weapon based on prompt</li>
                <li><code>POST /generate</code> - Generate weapon with additional parameters</li>
                <li><code>GET /logs</code> - View request logs (admin only)</li>
            </ul>
            
            <h3>Example:</h3>
            <pre>curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Generate a legendary frost bow", "temperature": 0.8, "max_length": 200}'</pre>
        </div>
    </div>

    <script>
        // Update slider values
        document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('temperature-value').textContent = this.value;
        });
        
        document.getElementById('max_length').addEventListener('input', function() {
            document.getElementById('max_length-value').textContent = this.value;
        });
    </script>
</body>
</html>
""")

with open(os.path.join(templates_dir, "logs.html"), "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Logs - Fantasy Weapon Generator</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">Fantasy Weapon Generator</div>
            <div class="tagline">Admin Dashboard</div>
        </div>
    </header>

    <div class="container">
        <div class="card">
            <h2>System Information</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Uptime</div>
                    <div class="stat-value">{{ uptime }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Total Requests</div>
                    <div class="stat-value">{{ request_count }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Model</div>
                    <div class="stat-value">{{ model_name }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">CPU Usage</div>
                    <div class="stat-value">{{ cpu_usage }}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Memory Usage</div>
                    <div class="stat-value">{{ memory_usage }}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Platform</div>
                    <div class="stat-value">{{ platform }}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Request Logs</h2>
            {% if logs %}
                {% for log in logs %}
                <div class="log-entry">
                    <strong>Time:</strong> {{ log.timestamp }}<br>
                    <strong>ID:</strong> {{ log.request_id }}<br>
                    <strong>Prompt:</strong> {{ log.prompt }}<br>
                    <strong>Status:</strong> {{ log.status }}<br>
                    {% if log.error %}
                    <strong>Error:</strong> {{ log.error }}<br>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p>No logs available.</p>
            {% endif %}
        </div>
        
        <div class="card">
            <a href="/"><button>Back to Generator</button></a>
        </div>
    </div>
</body>
</html>
""")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt for generation")
    temperature: float = Field(0.8, ge=0.1, le=1.5, description="Controls randomness (higher = more random)")
    max_length: int = Field(200, ge=10, le=1000, description="Maximum length of generated text")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Nucleus sampling parameter")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to generate")

class GenerationResponse(BaseModel):
    request_id: str
    prompt: str
    generated_text: str
    timestamp: str
    execution_time: float

class StatusResponse(BaseModel):
    status: str
    model_name: str
    device: str
    uptime: str
    request_count: int
    memory_usage: float
    cpu_usage: float
    platform_info: Dict[str, Any]

class LogEntry(BaseModel):
    request_id: str
    timestamp: str
    prompt: str
    status: str
    error: Optional[str] = None

def format_uptime(seconds):
    """Format uptime in human-readable format"""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"
    elif hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def log_request(request_id, prompt, status, error=None):
    """Log request to the in-memory log"""
    app.state.request_count += 1
    
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "status": status,
        "error": error
    }
    
    app.state.request_log.append(log_entry)
    if len(app.state.request_log) > 100:
        app.state.request_log.pop(0)
    
    return log_entry

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the web UI homepage"""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/generate-web", response_class=HTMLResponse)
async def generate_web(request: Request, prompt: str = Form(...), temperature: float = Form(0.8), max_length: int = Form(200)):
    """Generate weapon for web UI"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        if MODEL_PATH != "distilgpt2":
            formatted_prompt = f"<|prompt|>{prompt}<|completion|>"
        else:
            formatted_prompt = prompt
        
        input_ids = request.app.state.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        
        output = request.app.state.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.2,  
            num_return_sequences=1,
            pad_token_id=request.app.state.tokenizer.eos_token_id
        )
        
        generated_text = request.app.state.tokenizer.decode(output[0], skip_special_tokens=False)
        
        if MODEL_PATH != "distilgpt2" and "<|completion|>" in generated_text:
            if "<|endoftext|>" in generated_text:
                generated_text = generated_text.split("<|completion|>")[1].split("<|endoftext|>")[0]
            else:
                generated_text = generated_text.split("<|completion|>")[1]
        elif MODEL_PATH == "distilgpt2":
            generated_text = generated_text[len(prompt):].strip()
        
        lines = generated_text.split("\n")
        processed_lines = []
        seen_content = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_content:
                processed_lines.append(line)
                seen_content.add(line)
        
        processed_text = "\n".join(processed_lines)
        
        formatted_result = '<div class="weapon-card">'
        
        weapon_name = ""
        weapon_type = ""
        weapon_damage = ""
        weapon_properties = []
        weapon_description = ""
        
        for line in processed_lines:
            if "Name:" in line:
                weapon_name = line.split("Name:")[1].strip()
                formatted_result += f'<div class="weapon-title">{weapon_name}</div>'
            elif "Type:" in line or "Damage Type:" in line:
                weapon_type = line.split(":")[1].strip()
                formatted_result += f'<div class="weapon-property"><span class="property-label">Type:</span> {weapon_type}</div>'
            elif "Damage:" in line:
                weapon_damage = line.split("Damage:")[1].strip()
                formatted_result += f'<div class="weapon-property"><span class="property-label">Damage:</span> {weapon_damage}</div>'
            elif "Properties:" in line:
                props = line.split("Properties:")[1].strip().split(", ")
                weapon_properties = props
                formatted_result += '<div class="weapon-properties">'
                for prop in props:
                    formatted_result += f'<span class="property-tag">{prop}</span>'
                formatted_result += '</div>'
            elif "Description:" in line:
                weapon_description = line.split("Description:")[1].strip()
                formatted_result += f'<div class="weapon-description">{weapon_description}</div>'
            elif line and not any(key in line for key in ["Name:", "Type:", "Damage Type:", "Damage:", "Properties:", "Description:"]):
               
                weapon_description += " " + line
                formatted_result += f'<p class="weapon-lore">{line}</p>'
        
        
        if not weapon_name:
            formatted_result = '<div class="weapon-card"><pre class="weapon-raw">' + processed_text + '</pre></div>'
        
        formatted_result += '</div>'
        
        formatted_result = '''
        <style>
        .weapon-card {
            background: linear-gradient(to bottom, #2c3e50, #1a2530);
            color: #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .weapon-title {
            color: #e74c3c;
            font-size: 2rem;
            font-weight: bold;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 15px;
            font-family: 'Cinzel', serif;
        }
        .weapon-property {
            margin: 8px 0;
            font-size: 1.1rem;
        }
        .property-label {
            color: #f39c12;
            font-weight: bold;
        }
        .weapon-properties {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 15px 0;
        }
        .property-tag {
            background-color: rgba(52, 152, 219, 0.3);
            border: 1px solid #3498db;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
        }
        .weapon-description {
            font-style: italic;
            color: #bdc3c7;
            margin: 15px 0;
            line-height: 1.5;
            font-size: 1.1rem;
        }
        .weapon-lore {
            color: #ecf0f1;
            line-height: 1.6;
            margin-top: 15px;
            font-size: 1rem;
            text-align: justify;
        }
        .weapon-raw {
            white-space: pre-wrap;
            color: #ecf0f1;
            line-height: 1.6;
        }
        </style>
        ''' + formatted_result
        
        # Log successful request
        log_request(request_id, prompt, "success")
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "result": formatted_result
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        log_request(request_id, prompt, "error", str(e))
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "result": f"<p>Error: {str(e)}</p>"
            }
        )

@app.get("/generate", response_model=GenerationResponse)
async def generate_get(prompt: str, temperature: float = 0.8, max_length: int = 200):
    """Generate weapon based on GET request parameters"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        request_data = GenerationRequest(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            top_p=0.9,
            num_return_sequences=1
        )
        
        return await generate(request_data, request_id, start_time)
        
    except Exception as e:
        logger.error(f"Error in GET generation: {str(e)}")
        log_request(request_id, prompt, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerationResponse)
async def generate(request_data: GenerationRequest, request_id: str = None, start_time: float = None):
    """Generate weapon based on POST request with full parameter control"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    if start_time is None:
        start_time = time.time()
    
    try:
        if MODEL_PATH != "distilgpt2":
            formatted_prompt = f"<|prompt|>{request_data.prompt}<|completion|>"
        else:
            formatted_prompt = request_data.prompt
        
        input_ids = app.state.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        
        output = app.state.model.generate(
            input_ids,
            max_length=request_data.max_length,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
            num_return_sequences=request_data.num_return_sequences,
            pad_token_id=app.state.tokenizer.eos_token_id
        )
        
        generated_text = app.state.tokenizer.decode(output[0], skip_special_tokens=False)
        
        if MODEL_PATH != "distilgpt2" and "<|completion|>" in generated_text:
            if "<|endoftext|>" in generated_text:
                generated_text = generated_text.split("<|completion|>")[1].split("<|endoftext|>")[0]
            else:
                generated_text = generated_text.split("<|completion|>")[1]
        elif MODEL_PATH == "distilgpt2":
            generated_text = generated_text[len(request_data.prompt):].strip()
        
        execution_time = time.time() - start_time
        
        log_request(request_id, request_data.prompt, "success")
        
        return {
            "request_id": request_id,
            "prompt": request_data.prompt,
            "generated_text": generated_text.strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": round(execution_time, 4)
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        log_request(request_id, request_data.prompt, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def status():
    """Return API status and system information"""
    uptime_seconds = time.time() - app.state.start_time
    uptime_formatted = format_uptime(uptime_seconds)
    
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }
    
    model_name = MODEL_PATH.split("/")[-1]
    
    return {
        "status": "running",
        "model_name": model_name,
        "device": str(device),
        "uptime": uptime_formatted,
        "request_count": app.state.request_count,
        "memory_usage": memory.percent,
        "cpu_usage": cpu_usage,
        "platform_info": platform_info
    }

@app.get("/logs", response_class=HTMLResponse)
async def view_logs(request: Request):
    """View API request logs (admin page)"""
    uptime_seconds = time.time() - app.state.start_time
    uptime_formatted = format_uptime(uptime_seconds)
    
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    model_name = MODEL_PATH.split("/")[-1]
    
    return templates.TemplateResponse(
        "logs.html", 
        {
            "request": request,
            "logs": app.state.request_log,
            "request_count": app.state.request_count,
            "uptime": uptime_formatted,
            "model_name": model_name,
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "platform": platform.system()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)