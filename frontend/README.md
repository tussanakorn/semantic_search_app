# Semantic Search Frontend

This directory contains a Streamlit-based frontend for the Semantic Search API. It provides a user-friendly web interface for searching documents using semantic search.

## Features

- Interactive search interface
- Real-time results display
- Configurable number of search results
- API health monitoring
- Example queries for quick testing

## Running the Frontend

### Option 1: Using Docker Compose

The easiest way to run the entire system (API + Frontend) is using Docker Compose:

```bash
# From the project root directory
docker-compose up -d
```

This will start both the API and the frontend services. The frontend will be available at http://localhost:8501.

### Option 2: Running Locally

To run the frontend locally:

1. Make sure the API is running (see main README.md)
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the frontend using the provided script:
   ```bash
   cd frontend
   python run_frontend.py --api-url http://localhost:8000
   ```
   
   Or directly with Streamlit:
   ```bash
   cd frontend
   streamlit run app.py
   ```

## Configuration

The frontend can be configured using the following environment variables:

- `API_URL`: URL of the Semantic Search API (default: http://localhost:8000)

## Customization

You can customize the Streamlit theme by modifying the `config.toml` file in this directory.

## Troubleshooting

If you encounter issues with the frontend:

1. Check that the API is running and accessible
2. Verify that the API URL is correctly configured
3. Check the browser console for any JavaScript errors
4. Look at the Streamlit logs for Python errors