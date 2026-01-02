# ===== Poké Pack Pal - FastAPI Backend =====
# Complete backend with authentication, predictions, and subscription management

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import secrets
import jwt
import os
from dotenv import load_dotenv

# ============ LOAD ENVIRONMENT VARIABLES ============
load_dotenv()  # Load .env file

DATABASE_URL = os.getenv('DATABASE_URL')
JWT_SECRET = os.getenv('JWT_SECRET', 'dev-secret-key')
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PRO_PRICE_ID = os.getenv('STRIPE_PRO_PRICE_ID')
STRIPE_ULTIMATE_PRICE_ID = os.getenv('STRIPE_ULTIMATE_PRICE_ID')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

if not DATABASE_URL:
    raise Exception("DATABASE_URL not found in environment variables!")

# ============ FASTAPI APP SETUP ============
app = FastAPI(title="Poké Pack Pal API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# ============ DATABASE CONNECTION ============
def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# ============ PYDANTIC MODELS ============
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class PredictionRequest(BaseModel):
    set_name: str
    weight: float
    user_id: Optional[int] = None

class PackLogRequest(BaseModel):
    set_name: str
    weight: float
    hit: bool
    hit_type: Optional[str] = None
    card_name: Optional[str] = None
    rarity: Optional[str] = None

class SubscriptionRequest(BaseModel):
    tier: str  # "pro" or "ultimate"
    payment_method_id: str

# ============ UTILITY FUNCTIONS ============
def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(32)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return pwd_hash.hex(), salt

def create_jwt_token(user_id: int, username: str, tier: str) -> str:
    """Create JWT token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'tier': tier,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user data"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============ TIER ENFORCEMENT ============
def check_prediction_limit(user_id: int, tier: str):
    """Check if user can make a prediction based on their tier"""
    if tier == "ultimate":
        return {"allowed": True, "remaining": "unlimited"}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    month_year = datetime.now().strftime("%Y-%m")
    
    try:
        if tier == "free":
            # Check free tier usage
            cursor.execute("""
                SELECT predictions_used FROM free_tier_usage
                WHERE user_id = %s AND month_year = %s
            """, (user_id, month_year))
            
            result = cursor.fetchone()
            used = result['predictions_used'] if result else 0
            
            if used >= 5:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "message": "Free tier limit reached (5/month). Upgrade to Pro for 100/month!",
                    "upgrade_prompt": True
                }
            
            return {"allowed": True, "remaining": 5 - used}
        
        elif tier == "pro":
            # Check pro tier usage
            cursor.execute("""
                SELECT predictions_used FROM pro_tier_usage
                WHERE user_id = %s AND month_year = %s
            """, (user_id, month_year))
            
            result = cursor.fetchone()
            used = result['predictions_used'] if result else 0
            
            if used >= 100:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "message": "Pro tier limit reached (100/month). Upgrade to Ultimate for unlimited!",
                    "upgrade_prompt": True
                }
            
            return {"allowed": True, "remaining": 100 - used}
        
    finally:
        cursor.close()
        conn.close()

def increment_prediction_count(user_id: int, tier: str):
    """Increment prediction count for user"""
    if tier == "ultimate":
        return  # Unlimited, no tracking needed
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    month_year = datetime.now().strftime("%Y-%m")
    
    try:
        if tier == "free":
            cursor.execute("""
                INSERT INTO free_tier_usage (user_id, month_year, predictions_used)
                VALUES (%s, %s, 1)
                ON CONFLICT (user_id, month_year)
                DO UPDATE SET predictions_used = free_tier_usage.predictions_used + 1
            """, (user_id, month_year))
        
        elif tier == "pro":
            cursor.execute("""
                INSERT INTO pro_tier_usage (user_id, month_year, predictions_used)
                VALUES (%s, %s, 1)
                ON CONFLICT (user_id, month_year)
                DO UPDATE SET predictions_used = pro_tier_usage.predictions_used + 1
            """, (user_id, month_year))
        
        conn.commit()
    finally:
        cursor.close()
        conn.close()

# ============ API ENDPOINTS ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# ============ AUTHENTICATION ENDPOINTS ============

@app.post("/api/auth/register")
async def register(user: UserRegister):
    """Register new user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if username or email exists
        cursor.execute("""
            SELECT user_id FROM users 
            WHERE username = %s OR email = %s
        """, (user.username, user.email))
        
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Hash password
        pwd_hash, salt = hash_password(user.password)
        
        # Create user (free tier by default)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, salt, subscription_tier)
            VALUES (%s, %s, %s, %s, 'free')
            RETURNING user_id, username, email, subscription_tier
        """, (user.username, user.email, pwd_hash, salt))
        
        new_user = cursor.fetchone()
        conn.commit()
        
        # Create JWT token
        token = create_jwt_token(new_user['user_id'], new_user['username'], 'free')
        
        return {
            "message": "User registered successfully",
            "token": token,
            "user": {
                "user_id": new_user['user_id'],
                "username": new_user['username'],
                "email": new_user['email'],
                "tier": 'free'
            }
        }
    
    finally:
        cursor.close()
        conn.close()

@app.post("/api/auth/login")
async def login(user: UserLogin):
    """Login user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user
        cursor.execute("""
            SELECT user_id, username, email, password_hash, salt, subscription_tier
            FROM users WHERE username = %s
        """, (user.username,))
        
        db_user = cursor.fetchone()
        
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Verify password
        pwd_hash, _ = hash_password(user.password, db_user['salt'])
        
        if pwd_hash != db_user['password_hash']:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Create JWT token
        token = create_jwt_token(
            db_user['user_id'], 
            db_user['username'], 
            db_user['subscription_tier']
        )
        
        return {
            "message": "Login successful",
            "token": token,
            "user": {
                "user_id": db_user['user_id'],
                "username": db_user['username'],
                "email": db_user['email'],
                "tier": db_user['subscription_tier']
            }
        }
    
    finally:
        cursor.close()
        conn.close()

# ============ PREDICTION ENDPOINTS ============

@app.post("/api/predict")
async def predict(request: PredictionRequest, user: dict = Depends(verify_jwt_token)):
    """Get pack weight prediction"""
    
    # Check tier limits
    limit_check = check_prediction_limit(user['user_id'], user['tier'])
    
    if not limit_check['allowed']:
        raise HTTPException(
            status_code=403, 
            detail={
                "message": limit_check['message'],
                "upgrade_prompt": True,
                "tier": user['tier']
            }
        )
    
    # Get prediction from database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Advanced prediction with confidence intervals
        cursor.execute("""
            WITH weight_range AS (
                SELECT 
                    hit,
                    weight,
                    hit_type
                FROM global_packs
                WHERE set_name = %s 
                AND weight BETWEEN %s AND %s
            )
            SELECT 
                COUNT(*) as total_samples,
                SUM(CASE WHEN hit = true THEN 1 ELSE 0 END) as hit_count,
                AVG(weight) as avg_weight
            FROM weight_range
        """, (request.set_name, request.weight - 0.05, request.weight + 0.05))
        
        result = cursor.fetchone()
        
        if not result or result['total_samples'] == 0:
            return {
                "set_name": request.set_name,
                "weight": request.weight,
                "prediction": None,
                "message": "Not enough data for prediction",
                "remaining_predictions": limit_check.get('remaining', 'unlimited')
            }
        
        total = result['total_samples']
        hits = result['hit_count'] or 0
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        # Calculate confidence interval (Wilson score)
        import math
        z = 1.96  # 95% confidence
        p = hit_rate / 100
        n = total
        
        if n > 0:
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = (z / denominator) * math.sqrt(p * (1-p) / n + z**2 / (4*n**2))
            
            confidence_lower = max(0, (center - margin) * 100)
            confidence_upper = min(100, (center + margin) * 100)
        else:
            confidence_lower = 0
            confidence_upper = 100
        
        # Calculate reliability score
        reliability = min(100, (total / 50) * 100)
        
        # Increment prediction count
        increment_prediction_count(user['user_id'], user['tier'])
        
        return {
            "set_name": request.set_name,
            "weight": request.weight,
            "prediction": {
                "hit_rate": round(hit_rate, 1),
                "confidence_lower": round(confidence_lower, 1),
                "confidence_upper": round(confidence_upper, 1),
                "sample_size": total,
                "reliability_score": round(reliability, 0)
            },
            "remaining_predictions": limit_check.get('remaining', 'unlimited')
        }
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/stats/global")
async def get_global_stats(set_name: Optional[str] = None):
    """Get global statistics (free for all users)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if set_name:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_packs,
                    AVG(weight) as avg_weight,
                    SUM(CASE WHEN hit = true THEN 1 ELSE 0 END) as total_hits
                FROM global_packs
                WHERE set_name = %s
            """, (set_name,))
        else:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_packs,
                    AVG(weight) as avg_weight,
                    SUM(CASE WHEN hit = true THEN 1 ELSE 0 END) as total_hits,
                    COUNT(DISTINCT submitted_by_user_id) as total_contributors,
                    COUNT(DISTINCT set_name) as total_sets
                FROM global_packs
            """)
        
        result = cursor.fetchone()
        
        total_packs = result['total_packs'] or 0
        total_hits = result['total_hits'] or 0
        hit_rate = (total_hits / total_packs * 100) if total_packs > 0 else 0
        
        return {
            "total_packs": total_packs,
            "avg_weight": round(result['avg_weight'], 3) if result['avg_weight'] else 0,
            "total_hits": total_hits,
            "hit_rate": round(hit_rate, 1),
            "total_contributors": result.get('total_contributors', 0),
            "total_sets": result.get('total_sets', 0)
        }
    
    finally:
        cursor.close()
        conn.close()

# ============ PACK LOGGING ENDPOINTS ============

@app.post("/api/packs/add")
async def add_pack(pack: PackLogRequest, user: dict = Depends(verify_jwt_token)):
    """Add pack to global database (Pro and Ultimate only)"""
    
    if user['tier'] == 'free':
        raise HTTPException(
            status_code=403,
            detail="Pack logging requires Pro or Ultimate tier. Upgrade to unlock!"
        )
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO global_packs (
                set_name, weight, hit, hit_type, card_name, rarity,
                submitted_by_user_id, submitted_by_username, submitted_by_email
            )
            SELECT %s, %s, %s, %s, %s, %s, %s, u.username, u.email
            FROM users u
            WHERE u.user_id = %s
            RETURNING pack_id
        """, (
            pack.set_name, pack.weight, pack.hit, pack.hit_type,
            pack.card_name, pack.rarity, user['user_id'], user['user_id']
        ))
        
        pack_id = cursor.fetchone()['pack_id']
        
        # Update user's contribution count
        cursor.execute("""
            UPDATE users
            SET total_contributions = total_contributions + 1
            WHERE user_id = %s
        """, (user['user_id'],))
        
        conn.commit()
        
        return {
            "message": "Pack logged successfully",
            "pack_id": pack_id
        }
    
    finally:
        cursor.close()
        conn.close()

# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Poké Pack Pal Backend...")
    print(f"📡 Database: {DATABASE_URL[:50]}...")
    print(f"🔑 JWT Secret: {JWT_SECRET[:20]}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)