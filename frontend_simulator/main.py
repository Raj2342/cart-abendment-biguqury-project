from fastapi import FastAPI, Request
from datetime import datetime
import pandas as pd
import xgboost as xgb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. THE BRAIN: Load Multiclass XGBoost Model
try:
    bst = xgb.Booster()
    bst.load_model("model.bst") 
    print("✅ Multiclass XGBoost Engine Online")
except Exception as e:
    print(f"🛑 ERROR LOADING MODEL: {e}")
    bst = None

active_sessions = {}

def calculate_new_7_features(session_id):
    events = active_sessions.get(session_id, [])
    
    # 1. EVENT TIMES SEPARATE KARO
    views = [e["time"] for e in events if e["type"] == "view"]
    carts = [e["time"] for e in events if e["type"] == "cart"]
    
    # JARVIS FIX: Extract Cart Finalized Time
    finalized = [e["time"] for e in events if e["type"] == "cart_finalized"]
    
    first_view = min(views) if views else None
    first_cart = min(carts) if carts else None
    
    # 2. BOUNDARY DECIDE KARO (Cart Finalized button daba ya nahi?)
    if finalized:
        last_cart_boundary = max(finalized) # Agar final daba diya, toh wahi boundary hai
    elif carts:
        last_cart_boundary = max(carts) # Warna aakhri cart item boundary hai
    else:
        last_cart_boundary = None

    if not first_cart:
        return None

    # 3. VIEWS LOGIC (Based on Event Time and Boundary)
    # views_before_first_cart = len([e for e in events if e["type"] == "view" and e["time"] < first_cart])
    total_views_up_to_last_cart = len([e for e in events if e["type"] == "view" and e["time"] <= last_cart_boundary])
    
    # JARVIS FIX: Views strictly after 'Cart Finalized' or Last Cart
    views_after_last_cart = len([e for e in events if e["type"] == "view" and e["time"] > last_cart_boundary])
    
    unique_categories = len(set([e["category"] for e in events if e["type"] == "view" and e["time"] <= last_cart_boundary]))
    unique_products = len(set([e["product"] for e in events if e["type"] == "view" and e["time"] <= last_cart_boundary]))
    
    # 4. TIME LOGIC (Proper Seconds Calculation)
    time_to_cart_sec = 0
    if first_view and last_cart_boundary:
        time_to_cart_sec = max(0, (last_cart_boundary - first_view).total_seconds())
        
    overthinker_ratio = (total_views_up_to_last_cart / unique_products) if unique_products > 0 else 0.0

    features = {
        # "views_before_first_cart": views_before_first_cart,
        "total_views_up_to_last_cart": total_views_up_to_last_cart,
        "views_after_last_cart": views_after_last_cart,
        "unique_categories_viewed": unique_categories,
        "unique_products_viewed_total": unique_products,
        "time_to_cart_sec": int(time_to_cart_sec), # Send exact seconds
        "overthinker_ratio": round(overthinker_ratio, 2)
    }
    return features

@app.post("/track")
async def track_endpoint(request: Request):
    payload = await request.json()
    session_id = payload["session_id"]
    
    # JARVIS FIX: Frontend ka event_time use kar rahe hain
    # Replace 'Z' handles timezone format from JS
    event_time_str = payload["event_time"].replace('Z', '+00:00')
    event_time = datetime.fromisoformat(event_time_str)
    
    if session_id not in active_sessions:
        active_sessions[session_id] = []
        
    active_sessions[session_id].append({
        "time": event_time, # Using simulated time
        "type": payload["event_type"],
        "category": payload.get("category", ""),
        "product": payload.get("product", "")
    })
    return {"status": "tracked"}

@app.post("/predict")
async def predict_endpoint(request: Request):
    payload = await request.json()
    session_id = payload["session_id"]
    
    features_dict = calculate_new_7_features(session_id)
    print(f"\n🧠 Jarvis Features Calculated:")
    print(features_dict)
    
    if not features_dict:
        return {"action": "👀 Ignore (Class 1: Window Shopper - No Cart Yet)"}

    df = pd.DataFrame([features_dict])
    
    if bst:
        dmatrix = xgb.DMatrix(df)
        raw_pred = bst.predict(dmatrix)
        
        probs = raw_pred[0] 
        class_0_prob = probs[0] # Safe
        class_1_prob = probs[1] # Window
        class_2_prob = probs[2] # Hesitator
        
        print(f"🎯 Probabilities -> Safe: {class_0_prob:.2f} | Window: {class_1_prob:.2f} | Hesitator: {class_2_prob:.2f}")
        
        if class_2_prob > 0.60: 
            action = "🔥 TRIGGER 10% DISCOUNT POP-UP (Hesitator Detected) 🔥"
        elif class_0_prob > class_1_prob:
            action = "✅ Suppress Discount (Safe Buyer)"
        else:
            action = "👀 Ignore (Window Shopper)"
            
    else:
        action = "Model offline."

    return {
        "action": action,
        "features": features_dict
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)