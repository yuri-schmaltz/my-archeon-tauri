import pytest
import httpx
import json
import os
from datetime import datetime

BASE_URL = "http://127.0.0.1:8081"

@pytest.mark.asyncio
async def test_vram_monitor():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/system/monitor")
        assert response.status_code == 200
        data = response.json()
        assert "gpu" in data
        assert "vram" in data["gpu"]
        assert "allocated" in data["gpu"]["vram"]
        assert "total" in data["gpu"]["vram"]

@pytest.mark.asyncio
async def test_i18n_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/i18n")
        assert response.status_code == 200
        data = response.json()
        assert "en" in data
        assert "pt" in data
        assert "zh" in data
        assert data["en"]["app_title"] == "Archeon 3D"

@pytest.mark.asyncio
async def test_history_management():
    entry = {
        "request_id": "test_id_123",
        "prompt": "Test Model",
        "path": "test_output.glb",
        "timestamp": datetime.now().isoformat()
    }
    
    async with httpx.AsyncClient() as client:
        # Add to history
        add_resp = await client.post(f"{BASE_URL}/v1/history/add", json=entry)
        assert add_resp.status_code == 200
        
        # Get history
        get_resp = await client.get(f"{BASE_URL}/v1/history")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert "entries" in data
        # Ensure our test entry is at the top
        assert data["entries"][0]["prompt"] == "Test Model"

@pytest.mark.asyncio
async def test_downloads_status():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/system/downloads")
        assert response.status_code == 200
        data = response.json()
        assert "active" in data
        assert "progress" in data
        assert isinstance(data["models"], list)
