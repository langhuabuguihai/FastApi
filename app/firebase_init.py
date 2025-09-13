import os, json, base64
import firebase_admin
from firebase_admin import credentials

def init_firebase():
    # If already initialized, return existing app
    if firebase_admin._apps:
        return firebase_admin.get_app()

    # Prefer base64 var
    b64 = os.getenv("FIREBASE_CREDENTIALS_B64")
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if b64:
        cred_dict = json.loads(base64.b64decode(b64).decode("utf-8"))
        cred = credentials.Certificate(cred_dict)
        return firebase_admin.initialize_app(cred)
    if raw:
        cred = credentials.Certificate(json.loads(raw))
        return firebase_admin.initialize_app(cred)

    # Fallback to ADC only if you’ve set up GOOGLE_APPLICATION_CREDENTIALS etc.
    proj = os.getenv("GOOGLE_CLOUD_PROJECT")
    if proj:
        cred = credentials.ApplicationDefault()
        return firebase_admin.initialize_app(cred, {"projectId": proj})

    raise RuntimeError(
        "Firebase credentials not provided. Set FIREBASE_CREDENTIALS_B64 or FIREBASE_SERVICE_ACCOUNT_JSON."
    )
