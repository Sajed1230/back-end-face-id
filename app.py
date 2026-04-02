import cv2
import numpy as np
import os
import base64
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, request, jsonify
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
from deepface import DeepFace
from pymongo import MongoClient
from scipy.spatial.distance import cosine
app = Flask(__name__)
CORS(app)

# إعداد "المخ" (Face Landmarker)
# تأكد من وجود ملف face_landmarker.task في نفس المجلد
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#--------------------------------------------------------
load_dotenv()
#--------------------------------------------------------

cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure = True
)

# إعداد MongoDB باستخدام متغيرات البيئة
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["face_id_db"]
users_collection = db["users"]
#-------------------------------------------------------------------------

@app.route('/test_opencv', methods=['POST'])
def test_opencv():
    try:
        data = request.json
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({"status": "waiting", "message": "AWAITING_STREAM"}), 200

        # 1. تحويل الصورة من Base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 2. خطوة MediaPipe: الحصول على مواقع النقاط
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # --- خطوة OpenCV الذكية: تحليل الإضاءة بناءً على موقع الوجه ---
                # استخراج أقصى وأقل إحداثيات للوجه لعمل "قص" (Crop)
                xs = [l.x for l in landmarks]
                ys = [l.y for l in landmarks]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                
                # التأكد من أن الإحداثيات داخل حدود الصورة
                face_roi = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_roi.size > 0:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray_face)
                    
                    # فحص الإضاءة أولاً (حد الإضاءة 95)
                    if brightness < 50:
                        return jsonify({
                            "status": "waiting",
                            "message": f"🔦 LOW_LIGHT: NEED_MORE_BRIGHTNESS ({int(brightness)})",
                            "landmarks": [{"x": l.x, "y": l.y} for l in landmarks]
                        }), 200

                # 3. فحص المسافة (المسافة بين العينين)
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                eye_dist = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)

                if eye_dist < 0.18:
                    return jsonify({
                        "status": "waiting", "message": "⚠️ MOVE_CLOSER",
                        "landmarks": [{"x": l.x, "y": l.y} for l in landmarks]
                    }), 200
                
                # 4. إذا مر من كل الفحوصات
                return jsonify({
                    "status": "ok",
                    "message": "✅ READY_FOR_IDENTITY_SCAN",
                    "landmarks": [{"x": l.x, "y": l.y} for l in landmarks]
                }), 200
            
            return jsonify({"status": "waiting", "message": "🔍 SEARCHING_FOR_FACE..."}), 200

        return jsonify({"status": "error", "message": "FRAME_ERROR"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

#----------------------------------------------------------------------------------



def check_if_user_exists(new_embedding):
    """تقارن البصمة الحالية بكل البصمات في قاعدة البيانات"""
    all_users = list(users_collection.find({})) # جلب كل المستخدمين
    
    if not all_users:
        return None

    # نغير الـ threshold ليكون أكثر صرامة (Strict)
    # 0.30 يعني يجب أن يكون الوجه متطابقاً جداً
    threshold = 0.60 
    
    found_user = None
    min_dist = 1.0# نبدأ بأكبر مسافة ممكنة

    for user in all_users:
        stored_vector = user['face_vector']
        # حساب المسافة
        dist = cosine(new_embedding, stored_vector)
        
        print(f"📊 [COMPARING]: Distance with {user['name']} is: {dist:.4f}")

        if dist < threshold:
            # إذا وجدنا وجهين قريبين، نختار الأقرب على الإطلاق
            if dist < min_dist:
                min_dist = dist
                found_user = user['name']
            
    return found_user


@app.route('/enroll_user', methods=['POST'])
def enroll_user():
    temp_filename = "temp_enrollment.jpg"
    try:
        data = request.json
        # تنظيف الاسم من المسافات الزائدة
        user_name = data.get('userName', 'Guest').strip()
        image_b64 = data.get('image')

        # 1. تحويل الـ Base64 إلى صورة OpenCV
        header, encoded = image_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- الخطوة الأولى: التحقق من الاسم أولاً ---
        print(f"🔍 [CHECK 1]: Searching for name '{user_name}'...")
        name_exists = users_collection.find_one({"name": {"$regex": f"^{user_name}$", "$options": "i"}})
        
        if name_exists:
            print(f"❌ [REJECTED]: Name '{user_name}' already taken.")
            return jsonify({
                "status": "error",
                "message": f"The name '{user_name}' is already registered. Please choose another name or login.",
                "error_type": "NAME_EXISTS"
            }), 400

        # --- الخطوة الثانية: استخراج البصمة والتحقق من الوجه ---
        results = DeepFace.represent(
            img_path = img, 
            model_name = "VGG-Face", 
            enforce_detection = True,
            detector_backend = "opencv"
        )
        face_embedding = results[0]["embedding"]

        print(f"🔍 [CHECK 2]: Comparing face with database...")
        existing_user_by_face = check_if_user_exists(face_embedding)
        
        if existing_user_by_face:
            print(f"❌ [REJECTED]: Face already belongs to '{existing_user_by_face}'")
            return jsonify({
                "status": "error",
                "message": f"This face is already linked to the account: '{existing_user_by_face}'.",
                "error_type": "FACE_EXISTS"
            }), 400

        # --- الخطوة الثالثة: إذا كان الاسم والوجه جديدين، نبدأ الرفع ---
        print(f"☁️ [STEP 3]: Everything looks new. Uploading to Cloudinary...")
        clean_name_slug = user_name.replace(" ", "_")
        cv2.imwrite(temp_filename, img)
        
        upload_result = cloudinary.uploader.upload(
            temp_filename, 
            folder = "face_id_users",
            public_id = f"user_{clean_name_slug}_{np.random.randint(1000)}"
        )
        image_url = upload_result.get('secure_url')

        # 5. الحفظ النهائي في MongoDB
        user_document = {
            "name": user_name,
            "image_url": image_url,
            "face_vector": face_embedding,
            "created_at": str(np.datetime64('now'))
        }
        
        result = users_collection.insert_one(user_document)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        print(f"✅ [SUCCESS]: {user_name} registered successfully!")

        return jsonify({
            "status": "success",
            "message": f"Welcome {user_name}! Your face and name are now secured in our system.",
            "db_id": str(result.inserted_id)
        })

    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        print(f"❌ Error: {str(e)}")
        return jsonify({"status": "error", "message": "System couldn't process your request."}), 500
#==========================================LOGIN===============================================================
@app.route('/login_user', methods=['POST'])
def login_user():
    try:
        data = request.json
        user_name_input = data.get('userName', '').strip()
        image_b64 = data.get('image')

        # 1. البحث عن المستخدم بالاسم المدخل
        user_in_db = users_collection.find_one({"name": {"$regex": f"^{user_name_input}$", "$options": "i"}})
        
        if not user_in_db:
            return jsonify({"status": "error", "message": "USER_NOT_FOUND"}), 404

        # 2. تحويل المعالجة والمقارنة (كما فعلنا سابقاً)
        header, encoded = image_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        current_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = DeepFace.represent(img_path=current_img, model_name="VGG-Face", enforce_detection=True)
        current_embedding = results[0]["embedding"]
        
        dist = cosine(current_embedding, user_in_db['face_vector'])

        # 3. التحقق من النتيجة وإرسال الاسم المسجل
        if dist < 0.45:
            # نرسل الاسم الحقيقي من قاعدة البيانات (user_in_db['name'])
            return jsonify({
                "status": "success", 
                "message": "VERIFICATION_SUCCESSFUL",
                "registeredName": user_in_db['name'] 
            })
        else:
            return jsonify({"status": "error", "message": "FACE_NOT_MATCHED"}), 401

    except Exception as e:
        return jsonify({"status": "error", "message": "SYSTEM_BUSY"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)