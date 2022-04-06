
import json





# def test_get(test_app):
#     response = test_app.get("/") 
#     assert response.status_code == 200
#     assert "text/html" in response.headers['content-type']

def test_post(test_app,test_create_random_base64image):
    payload = json.dumps({"image":f"{test_create_random_base64image}"})
    respones = test_app.post("/predict",data=payload)
    assert respones.headers['content-type'] == 'application/json'
    assert respones.status_code == 200
    assert respones.is_redirect is False


def test_create_note_invalid_post(test_app):
    false_payload = json.dumps({"image":"sandeepjean"})
    respones = test_app.post("/predict",data=false_payload)
    assert respones.status_code == 418


