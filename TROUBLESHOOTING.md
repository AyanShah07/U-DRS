# Quick Network Test

## Test 1: Check if Backend is Accessible

Open your web browser and try these URLs:

### From Your Computer:
```
http://127.0.0.1:8080/api/health
http://localhost:8080/api/health
```

### From Your Network:
```
http://10.33.56.62:8080/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "models_loaded": true
}
```

---

## Test 2: Check Firewall

Your Windows Firewall might be blocking external connections to port 8080.

**Quick Fix:**
```bash
# Run in PowerShell as Administrator
New-NetFirewallRule -DisplayName "U-DRS API" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

---

## Test 3: Reload Mobile App

In your Expo terminal, press **`r`** to reload the app.

Or on your phone:
- Shake the device
- Tap "Reload"

---

## Test 4: Check Mobile App Console

Look at the Expo terminal for any error messages when the app tries to connect.

---

## Alternative: Use Localhost for Web Testing

If you want to test quickly in the browser instead of on a phone:

1. Press **`w`** in the Expo terminal
2. The web app will use your network IP automatically
3. You can test the full workflow in the browser

---

## Current Configuration:

**Mobile App** (`config/api.js`):
```javascript
const API_BASE_URL = 'http://10.33.56.62:8080/api';
```

**Backend** (`api/main.py`):
```python
host: str = "0.0.0.0"  # Listens on all interfaces
port: int = 8080
```

**Your Network IP**: `10.33.56.62`
