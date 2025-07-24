# DeepWukong Backend API

A FastAPI-based backend service for the DeepWukong vulnerability detection system. This API provides automated vulnerability detection for C/C++ source code using Deep Graph Neural Networks.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- DeepWukong trained model checkpoint
- Joern static analysis tool (optional, falls back to mock mode)

### Installation

1. **Clone and setup project structure:**
```bash
# Run the setup script to create directory structure
bash setup_project.sh

# Or create manually:
mkdir -p deepwukong-backend
cd deepwukong-backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Initialize database:**
```bash
python scripts/init_db.py
```

5. **Setup DeepWukong model:**
```bash
python scripts/setup_model.py
```

6. **Copy DeepWukong source code:**
```bash
# Copy your DeepWukong source to deepwukong/src/
cp -r /path/to/your/DeepWukong/src/* deepwukong/src/

# Copy trained model checkpoint
cp /path/to/your/model.ckpt storage/models/deepwukong_current.ckpt

# Copy Joern (if available)
cp /path/to/joern-parse deepwukong/joern/
```

7. **Start the server:**
```bash
python run.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000

# File limits
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=".c,.cpp,.h,.hpp,.cc,.cxx"

# AI Model
DEEPWUKONG_MODEL_PATH="./storage/models/deepwukong_current.ckpt"
DEFAULT_CONFIDENCE_THRESHOLD=0.7
AI_TIMEOUT_SECONDS=300

# Database
DATABASE_URL="sqlite:///./storage/deepwukong.db"
```

## 🔗 API Endpoints

### Health & Status
- `GET /api/health` - Health check
- `GET /api/ai/status` - AI model status
- `GET /api/ai/info` - Detailed model information

### Analysis
- `POST /api/analyze` - Analyze file for vulnerabilities
- `GET /api/analyses` - List previous analyses
- `GET /api/analyses/{id}` - Get specific analysis
- `DELETE /api/analyses/{id}` - Delete analysis

### Example Usage

**Analyze a file:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.c" \
  -F "confidence_threshold=0.7"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "results": {
      "file_analyzed": "example.c",
      "vulnerabilities": [
        {
          "line_number": 15,
          "type": "buffer_overflow",
          "confidence": 0.85,
          "severity": "high",
          "description": "Possible buffer overflow vulnerability with high confidence",
          "recommendation": "Add bounds checking before array access"
        }
      ],
      "summary": {
        "total_vulnerabilities": 1,
        "high_confidence": 1,
        "processing_time": "2.34s"
      }
    }
  }
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  DeepWukong      │────│   SQLite DB     │
│                 │    │  Service         │    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • File Upload   │    │ • Model Loading  │    │ • Analysis      │
│ • Validation    │    │ • Inference      │    │   History       │
│ • API Routes    │    │ • Result Format  │    │ • Results       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **FastAPI Application** (`app/main.py`)
   - HTTP API endpoints
   - File upload handling
   - CORS middleware
   - Exception handling

2. **DeepWukong Service** (`app/services/deepwukong_service.py`)
   - AI model wrapper
   - Analysis pipeline
   - Mock mode for development

3. **Analysis Service** (`app/services/analysis_service.py`)
   - Database operations
   - Analysis history management

4. **Database Models** (`app/models/`)
   - SQLAlchemy models
   - Analysis storage

## 🧪 Development

### Mock Mode

The service automatically falls back to mock mode if:
- Model checkpoint is not found
- DeepWukong source code is missing
- Model loading fails

Mock mode generates realistic test results for development.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Development Server

```bash
# Run with auto-reload
python run.py
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
deepwukong-backend/
├── app/                          # FastAPI application
│   ├── api/                      # API routes
│   ├── core/                     # Core utilities
│   ├── models/                   # Database models
│   ├── schemas/                  # Pydantic schemas
│   └── services/                 # Business logic
├── deepwukong/                   # DeepWukong source
│   ├── src/                      # Original source code
│   └── configs/                  # Model configurations
├── storage/                      # File storage
│   ├── uploads/                  # Temporary files
│   ├── models/                   # Model checkpoints
│   ├── results/                  # Analysis results
│   └── logs/                     # Application logs
├── scripts/                      # Utility scripts
└── tests/                        # Test files
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model not loading:**
   - Check model file exists at `DEEPWUKONG_MODEL_PATH`
   - Verify DeepWukong source code is in `deepwukong/src/`
   - Service will fall back to mock mode

2. **File upload errors:**
   - Check file size limit (`MAX_FILE_SIZE_MB`)
   - Verify file extension is allowed
   - Ensure upload directory is writable

3. **Analysis timeout:**
   - Increase `AI_TIMEOUT_SECONDS`
   - Check system resources
   - Large files may need more time

### Logs

Check application logs for detailed error information:
```bash
# Application logs
tail -f storage/logs/app.log

# Server logs in console when running with DEBUG=True
```

## 📄 License

This project is licensed under the MIT License. See the original DeepWukong paper and repository for model-specific licensing.

## 📚 References

- [DeepWukong Paper (TOSEM'21)](https://doi.org/10.1145/3436877)
- [Original DeepWukong Repository](https://github.com/username/DeepWukong)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review API documentation at `/docs`
