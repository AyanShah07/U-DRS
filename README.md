# U-DRS Mobile App

React Native mobile application for the U-DRS (Universal Damage Reconstruction System).

## Features

✨ **Camera Integration**
- Take photos directly from the app
- Select images from gallery
- Real-time image preview

📊 **Damage Analysis**
- Upload images to U-DRS backend
- Real-time analysis status
- Comprehensive results display

📈 **Results Visualization**
- Severity classification with color coding
- 2D & 3D measurements
- Cost estimation & urgency levels
- Detailed damage metrics

🔗 **Backend Connectivity**
- API health monitoring
- Automatic connection status
- Error handling & retry logic

---

## Prerequisites

- Node.js 16+ and npm
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (Mac) or Android Emulator
- U-DRS backend running on `http://127.0.0.1:8080`

---

## Installation

```bash
cd U-DRS-Mobile

# Install dependencies
npm install

# Start the app
npm start
```

---

## Running the App

### On iOS Simulator (Mac only)
```bash
npm run ios
```

### On Android Emulator
```bash
npm run android
```

### On Web Browser
```bash
npm run web
```

### Using Expo Go App
1. Install **Expo Go** on your phone (iOS/Android)
2. Run `npm start`
3. Scan the QR code with:
   - iPhone: Camera app
   - Android: Expo Go app

---

## Configuration

### API Endpoint
Edit `config/api.js` to change the backend URL:

```javascript
const API_BASE_URL = 'http://127.0.0.1:8080/api';
```

**For physical device testing:**
- Replace `127.0.0.1` with your computer's IP address
- Example: `http://192.168.1.100:8080/api`

### Finding Your IP Address
**Windows:**
```bash
ipconfig
# Look for "IPv4 Address"
```

**Mac/Linux:**
```bash
ifconfig
# Look for "inet" address
```

---

## Usage

### 1. Ensure Backend is Running
```bash
cd ../U-DRS
python api/main.py
# Server should be running on http://127.0.0.1:8080
```

### 2. Launch Mobile App
```bash
cd ../U-DRS-Mobile
npm start
```

### 3. Take or Select Photo
- Tap **"📷 Take Photo"** to use camera
- Tap **"🖼️ Choose Photo"** to select from gallery

### 4. Analyze Damage
- Tap **"🔍 Analyze Damage"** button
- Wait 30-60 seconds for processing

### 5. View Results
- Severity assessment (Minor/Moderate/Severe/Critical)
- Detailed measurements
  - 2D: Area, crack length/width
  - 3D: Depth, volume (if enabled)
- Cost estimation with range
- Urgency level & timeline

---

## Screenshots

### Home Screen
Shows API connection status and action buttons.

### Image Preview
Displays selected image with analyze button.

### Analysis Results
- Color-coded severity cards
- Detailed measurements
- Cost & urgency predictions
- Scrollable results view

---

## API Integration

The app communicates with the U-DRS backend via REST API:

### Endpoints Used
- `GET /api/health` - Check backend status
- `POST /api/analyze` - Upload & analyze image
- `GET /api/result/{id}` - Retrieve results

### Request Format
```javascript
FormData {
  file: Image blob
  pixel_to_mm_ratio: float
  depth_scale: float  
  generate_3d: boolean
}
```

### Response Format
```javascript
{
  status: "damage_detected",
  measurements: { /* 2D & 3D metrics */ },
  severity: { class, score, confidence },
  cost_urgency: { cost_prediction, urgency, timeline }
}
```

---

## Troubleshooting

### "API Connection Error"
**Problem**: Cannot connect to backend

**Solutions**:
1. Ensure backend is running: `python api/main.py`
2. Check URL in `config/api.js`
3. For physical devices, use computer's IP address
4. Check firewall settings

### "Analysis Failed"
**Problem**: Image upload or processing failed

**Solutions**:
1. Check image format (JPG/PNG supported)
2. Ensure image is not too large (< 50MB)
3. Verify backend has models loaded
4. Check backend logs for errors

### Camera Not Working
**Problem**: Camera permission denied

**Solutions**:
1. Grant camera permission in phone settings
2. Restart the app
3. Try selecting from gallery instead

---

## File Structure

```
U-DRS-Mobile/
├── App.js                      # Main app component
├── config/
│   └── api.js                  # API configuration
├── services/
│   └── api.js                  # API service layer
├── package.json                # Dependencies
└── README.md                   # This file
```

---

## Dependencies

### Core
- `expo` - Framework
- `react-native` - Mobile framework
- `react` - UI library

### Features
- `expo-camera` - Camera access
- `expo-image-picker` - Gallery access
- `expo-file-system` - File handling
- `axios` - HTTP client

---

## Development

### Add New Features
1. Create new components in `components/`
2. Add new API calls in `services/api.js`
3. Update routes in `App.js`

### Testing
```bash
# Run on multiple devices simultaneously
npm start
# Then scan QR code on each device
```

### Build for Production

**iOS:**
```bash
expo build:ios
```

**Android:**
```bash
expo build:android
```

---

## Performance Tips

1. **Image Size**: Compress images before upload
2. **Network**: Use WiFi for faster uploads
3. **3D Models**: Disable 3D generation for faster results
4. **Background**: Close other apps during analysis

---

## Future Enhancements

- [ ] Offline mode with local caching
- [ ] History of past analyses
- [ ] 3D model viewer in-app
- [ ] Batch image processing
- [ ] Export reports as PDF
- [ ] Dark mode support
- [ ] Multi-language support

---

## Support

For issues or questions:
- Check backend logs: `U-DRS/udrs.log`
- Verify API is accessible: Visit `http://127.0.0.1:8080/docs`
- Review app console for errors

---

## License

Same as U-DRS main project (MIT)
