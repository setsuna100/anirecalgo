import requests
import secrets
import urllib.parse

CLIENT_ID = 'dffc7ca58c5c7162228ee20f12a26957'
CLIENT_SECRET = '3511ee635f1862e03dfa934a404ece974f3da863d6ed2d8fb1f74a7ead2b6995'

# 1. Generate the PKCE Code Verifier (must be 43-128 characters)
# MAL uses plain transformation, so challenge = verifier
code_verifier = secrets.token_urlsafe(100)[:128] 
code_challenge = code_verifier

# 2. Construct the Authorization URL
auth_url = (
    f"https://myanimelist.net/v1/oauth2/authorize"
    f"?response_type=code"
    f"&client_id={CLIENT_ID}"
    f"&code_challenge={code_challenge}"
    f"&code_challenge_method=plain"
)

print("Please navigate to the following URL to authorize the app:")
print(auth_url)

# 3. User logs in, clicks "Allow", and gets redirected to localhost/?code=...
# You paste that full localhost URL back into this script
redirect_response = input("\nPaste the full localhost redirect URL here: ")
parsed_url = urllib.parse.urlparse(redirect_response)
authorization_code = urllib.parse.parse_qs(parsed_url.query)['code'][0]

# 4. Exchange the Authorization Code for an Access Token
token_url = "https://myanimelist.net/v1/oauth2/token"
data = {
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
    'code': authorization_code,
    'code_verifier': code_verifier,
    'grant_type': 'authorization_code'
}

response = requests.post(token_url, data=data)
token_info = response.json()

print("\nAuthentication Successful!")
print(f"Access Token: {token_info.get('access_token')}")
print(f"Refresh Token: {token_info.get('refresh_token')}")