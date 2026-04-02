# OpenEnv-Hackathon

An OpenEnv environment implementation for the hackathon, showcasing a client-server architecture for interactive environments.

## Project Structure

```
OpenEnv-Hackathon/
├── my_first_env/              # Main package directory
│   ├── client.py              # Client implementation
│   ├── models.py              # Data models
│   ├── openenv.yaml           # Environment configuration
│   ├── pyproject.toml         # Python project metadata
│   └── server/                # Server implementation
│       ├── app.py             # FastAPI application
│       ├── my_first_env_environment.py  # Environment setup
│       ├── requirements.txt   # Server dependencies
│       └── Dockerfile         # Docker containerization
├── README.md                  # Project documentation
└── requirements.txt           # Main dependencies
```

## Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/OpenEnv-Hackathon.git
cd OpenEnv-Hackathon
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or with uv:
```bash
uv sync
```

### Running the Server

Start the FastAPI server:
```bash
python -m my_first_env.server.app
```

The server will be available at `http://localhost:8000`

### Running the Client

Use the client to interact with the environment:
```bash
python -c "from my_first_env.client import Client; client = Client()"
```

## Development

### Running Tests

```bash
pytest my_first_env/
pytest --cov=my_first_env  # With coverage
```

### Docker Support

Build and run the environment in Docker:
```bash
docker build -t openenv-hackathon .
docker run -p 8000:8000 openenv-hackathon
```

## Configuration

The environment is configured via `my_first_env/openenv.yaml`. Modify this file to customize:
- Environment parameters
- Server settings
- Client behavior

## Related Links

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD-style license - see the LICENSE file for details.
