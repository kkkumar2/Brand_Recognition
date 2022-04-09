from app import app 
import pytest
from starlette.testclient import TestClient
from src.utils.all_utills import read_yaml
import os
import numpy as np
from PIL import Image
import base64
import io

@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client


@pytest.fixture(scope="module")
def test_read_yaml():
    data = read_yaml(os.path.join("config","config.yaml"))
    return data

@pytest.fixture
def test_create_random_base64image():
    data = np.random.randint(0, 255, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    buffered = io.BytesIO()
    img.save(buffered,format="JPEG")
    base_64_img = base64.b64encode(buffered.getvalue())

    return base_64_img.decode("utf-8")



