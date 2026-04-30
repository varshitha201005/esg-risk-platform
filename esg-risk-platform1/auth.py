import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# ─── User Credentials ──────────────────────────────────────────────────────────
# To add more users, just add them to the credentials dict below
credentials = {
    "usernames": {
        "admin": {
            "email": "admin@esg.com",
            "name": "Admin User",
            "password": "$2b$12$9tHSUMEDWJGHEpCpZQMfuuMJ3s5wKkHbLFyJjFMNPZLiWHrKqDgHK"
        },
        "analyst": {
            "email": "analyst@esg.com",
            "name": "ESG Analyst",
            "password": "$2b$12$9tHSUMEDWJGHEpCpZQMfuuMJ3s5wKkHbLFyJjFMNPZLiWHrKqDgHK"
        }
    }
}

cookie_config = {
    "expiry_days": 1,
    "key": "esg_platform_secret_key",
    "name": "esg_auth_cookie"
}

def get_authenticator():
    authenticator = stauth.Authenticate(
        credentials,
        cookie_config["name"],
        cookie_config["key"],
        cookie_config["expiry_days"]
    )
    return authenticator