# Migration Guide: ngrok → Modal

The SAM3_3D API has been migrated from ngrok to Modal for improved stability and performance.

---

## What Changed

| | Old (ngrok) | New (Modal) |
|---|---|---|
| **URL** | `https://diploidic-describably-anabelle.ngrok-free.dev` | `https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run` |
| **Stability** | Tunnel drops frequently, needs manual restart | Permanent URL, auto-scaling |
| **Cold start** | Always warm (server running 24/7) | ~90s cold start if idle >5min |
| **Headers** | Requires `ngrok-skip-browser-warning: true` | No special headers needed |
| **Cost** | Free (but unreliable) | ~$25/mo for 100 req/day |

---

## Client Changes Required

### 1. Update API URL

**Before:**
```
https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json
```

**After:**
```
https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json
```

### 2. Remove ngrok Header

**Before:**
```
headers["ngrok-skip-browser-warning"] = "true"
```

**After:**
Remove this header. Not needed for Modal.

### 3. Handle Cold Start (Important)

Modal containers scale to zero after 5 minutes of inactivity. The first request after idle triggers a cold start (~90 seconds to load model).

**Client should:**
- Set HTTP timeout to **120+ seconds** (not 30s)
- Show a loading indicator on first request
- Subsequent requests will be fast (~3s per category)

---

## Code Changes

### Python
```python
# Before
url = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
headers = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json"
}

# After
url = "https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json"
headers = {
    "Content-Type": "application/json"
}
```

### Swift (iPhone)
```swift
// Before
var apiUrl = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
request.timeoutInterval = 60

// After
var apiUrl = "https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json"
// Remove ngrok header
request.timeoutInterval = 120  // increase for cold start
```

### Unity C# (Quest)
```csharp
// Before
string apiUrl = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json";
request.SetRequestHeader("ngrok-skip-browser-warning", "true");
request.timeout = 60;

// After
string apiUrl = "https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json";
// Remove ngrok header
request.timeout = 120;  // increase for cold start
```

### curl
```bash
# Before
curl -X POST https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json \
  -H "ngrok-skip-browser-warning: true" \
  -H "Content-Type: application/json" \
  -d '...'

# After
curl -X POST https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json \
  -H "Content-Type: application/json" \
  -d '...'
```

---

## What Stays the Same

Everything else is identical:

- **Request format**: Same JSON body, same fields
- **Response format**: Same `{"boxes": [...], "mode": "..."}`
- **Endpoints**: `/health`, `/detect3d`, `/detect3d_json`
- **Features**: depth input, multi-frame, multi-view reference, alignment_post_process
- **source field**: Still `"iphone"` or `"quest"`
- **cy convention**: Same as before

---

## Quick Verification

```bash
# Test health
curl https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/health

# Expected (may take ~90s on first call):
# {"status":"ok","model_loaded":true,"depth_input_supported":true}
```

---

## Cold Start Tips

- **Warm-up request**: Send a `/health` request before the user needs detection. This triggers model loading in the background.
- **Keep warm**: If you need instant responses, send a `/health` request every 4 minutes to prevent the container from scaling down.
- **Retry logic**: If you get a timeout, retry once. The container should be warm by then.

```swift
// Example: warm-up on app launch
func warmUpAPI() {
    let url = URL(string: "https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/health")!
    URLSession.shared.dataTask(with: url) { _, _, _ in
        print("API warmed up")
    }.resume()
}
```
