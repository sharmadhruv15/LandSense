import requests
import json
import base64
import os

url = 'http://localhost:5000/analyze'

# Test with all 4 screenshots
screenshots = [
    'Screenshot 2026-04-28 224712.png',
    'Screenshot 2026-04-28 225047.png',
    'Screenshot 2026-04-28 232244.png',
    'Screenshot 2026-04-28 232326.png',
]

for ss in screenshots:
    path = os.path.join('.', ss)
    if not os.path.exists(path):
        print(f"SKIP: {ss} not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing: {ss}")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        files = {'image': f}
        resp = requests.post(url, files=files)
    
    data = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Risk Level: {data.get('risk_level')}")
    print(f"  Risk Score: {data.get('risk_score')}")
    print(f"  Factors: {data.get('factors')}")
    print(f"  Has risk_map: {'risk_map' in data}")
    print(f"  risk_map length: {len(data.get('risk_map', ''))} chars")
    
    # Save the overlay image to verify visually
    if 'risk_map' in data and data['risk_map']:
        out_name = f"overlay_{ss}"
        img_data = base64.b64decode(data['risk_map'])
        with open(os.path.join('uploads', out_name), 'wb') as f:
            f.write(img_data)
        print(f"  Saved overlay to: uploads/{out_name}")
    else:
        print(f"  ERROR: No risk_map in response!")

print("\nDONE - check uploads/ folder for overlay images")
