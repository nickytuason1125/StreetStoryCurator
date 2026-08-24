"""Thin launcher — the FastAPI app lives in server_impl.py.

The monolith is being split into routers/ (Milestone 4). This file keeps
the historical `python server.py` entry point working unchanged.
"""
from server_impl import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
