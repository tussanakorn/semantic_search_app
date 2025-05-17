#!/usr/bin/env python3
"""
Standalone script to run the Streamlit frontend locally.
This script sets up the environment and launches the Streamlit app.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the Semantic Search Streamlit frontend')
    parser.add_argument('--port', type=int, default=8501, help='Port to run Streamlit on')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', 
                        help='URL of the API service')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['API_URL'] = args.api_url
    
    print(f"Starting Streamlit frontend on port {args.port}")
    print(f"API URL: {args.api_url}")
    
    # Run Streamlit
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0"
    ]
    
    try:
        # Change to the frontend directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit frontend...")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()