import requests
import os
from dotenv import load_dotenv

load_dotenv()

r = requests.get(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY")}'}
)

models = r.json()['data']
free = [m['id'] for m in models if m.get('pricing', {}).get('prompt') == '0']

print(f"Total free models: {len(free)}\n")
for m in free:
    print(m)
