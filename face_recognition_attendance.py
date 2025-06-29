import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import csv
import json
from datetime import datetime, timedelta
from PIL import Image

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize Haar cascades for emotion detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Attendance configuration
ATTENDANCE_COOLDOWN = 30  # Minimum seconds between attendance marks for same person
MIN_CONFIDENCE = 0.4      # Minimum confidence for face recognition
ATTENDANCE_FILE = 'attendance.csv'
SESSION_FILE = 'session_data.json'

# Session tracking
session_data = {}

def load_session_data():
    """Load session data from file"""
    global session_data
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r') as f:
                session_data = json.load(f)
        else:
            session_data = {}
    except Exception as e:
        print(f"Error loading session data: {e}")
        session_data = {}

def save_session_data():
    """Save session data to file"""
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        print(f"Error saving session data: {e}")

def initialize_attendance_file():
    """Initialize attendance CSV file with headers if it doesn't exist"""
    if not os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Status', 'Emotion', 'Confidence', 'Recognition_Score'])
        except Exception as e:
            print(f"Error initializing attendance file: {e}")

def get_today_attendance():
    """Get today's attendance records"""
    today = datetime.now().strftime('%Y-%m-%d')
    today_attendance = []
    
    try:
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Date'] == today:
                        today_attendance.append(row)
    except Exception as e:
        print(f"Error reading attendance file: {e}")
    
    return today_attendance

def is_already_marked_today(name):
    """Check if person is already marked present today"""
    today_attendance = get_today_attendance()
    for record in today_attendance:
        if record['Name'].upper() == name.upper() and record['Status'] == 'PRESENT':
            return True
    return False

def can_mark_attendance(name):
    """Check if attendance can be marked based on cooldown period"""
    current_time = datetime.now()
    
    # Check session cooldown
    if name in session_data:
        last_marked = datetime.fromisoformat(session_data[name]['last_marked'])
        time_diff = (current_time - last_marked).total_seconds()
        if time_diff < ATTENDANCE_COOLDOWN:
            return False, f"Cooldown: {int(ATTENDANCE_COOLDOWN - time_diff)}s remaining"
    
    # Check if already marked today
    if is_already_marked_today(name):
        return False, "Already marked present today"
    
    return True, "Can mark attendance"

def mark_attendance(name, emotion="Unknown", confidence=0.0, recognition_score=0.0):
    """Enhanced attendance marking with better logic"""
    can_mark, reason = can_mark_attendance(name)
    
    if not can_mark:
        print(f"Attendance not marked for {name}: {reason}")
        return False
    
    try:
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        # Mark attendance
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                name.upper(),
                date_string,
                time_string,
                'PRESENT',
                emotion.title(),
                f"{confidence:.2f}",
                f"{recognition_score:.3f}"
            ])
        
        # Update session data
        session_data[name] = {
            'last_marked': now.isoformat(),
            'total_detections': session_data.get(name, {}).get('total_detections', 0) + 1,
            'last_emotion': emotion,
            'last_confidence': confidence
        }
        save_session_data()
        
        print(f"✅ Attendance marked for {name} - {emotion} ({confidence:.2f})")
        return True
        
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False

def get_attendance_stats():
    """Get attendance statistics"""
    today_attendance = get_today_attendance()
    present_count = len([r for r in today_attendance if r['Status'] == 'PRESENT'])
    
    # Emotion statistics
    emotions = [r['Emotion'] for r in today_attendance]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        'present_today': present_count,
        'emotion_stats': emotion_counts,
        'total_records': len(today_attendance)
    }

def load_images(path):
    images = []
    class_names = []
    ilist = os.listdir(path)
    for cl in ilist:
        if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
            cur_image = cv2.imread(f'{path}/{cl}')
            if cur_image is not None:
                images.append(cur_image)
                class_names.append(os.path.splitext(cl)[0])
    return images, class_names

def find_encodings(images):
    encoded = []
    for img in images:
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        try:
            # Detect and crop face
            img_cropped = mtcnn(pil_img)
            
            if img_cropped is not None and len(img_cropped) > 0:
                # Use the first detected face
                face_tensor = img_cropped[0]
                # Generate embedding
                with torch.no_grad():
                    embedding = resnet(face_tensor.unsqueeze(0).to(device))
                    encoded.append(embedding.cpu().numpy().flatten())
            else:
                print(f"No face detected in one of the images")
                # Add a zero embedding as placeholder
                encoded.append(np.zeros(512))
        except Exception as e:
            print(f"Error processing image: {e}")
            encoded.append(np.zeros(512))
    
    return encoded

def calculate_distance(embedding1, embedding2):
    """Calculate cosine distance between two embeddings"""
    return np.linalg.norm(embedding1 - embedding2)

def calculate_confidence(distance, threshold=0.6):
    """Calculate confidence score based on distance"""
    if distance >= threshold:
        return 0.0
    return max(0.0, (threshold - distance) / threshold)

def detect_emotion(face_img):
    """Detect emotion using Haar cascade smile detection"""
    try:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(
            gray_face, 
            scaleFactor=1.7, 
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        if len(smiles) > 0:
            return "Happy", 0.8
        else:
            return "Neutral", 0.6
            
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Unknown", 0.0

def get_emotion_color(emotion):
    """Get color for emotion display"""
    emotion_colors = {
        'Happy': (0, 255, 0),      # Green
        'Neutral': (128, 128, 128), # Gray
        'Unknown': (64, 64, 64)    # Dark Gray
    }
    return emotion_colors.get(emotion, (255, 255, 255))

def display_attendance_info(img, stats):
    """Display attendance information on the frame"""
    y_pos = 60
    cv2.putText(img, f"Present Today: {stats['present_today']}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_pos += 25
    for emotion, count in stats['emotion_stats'].items():
        cv2.putText(img, f"{emotion}: {count}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20

# Initialize system
load_session_data()
initialize_attendance_file()

# Load and encode known faces
path = 'image'
images, class_names = load_images(path)
encoded_list = find_encodings(images)
print('Encoded list completed')

# Get initial stats
stats = get_attendance_stats()
print(f"Today's attendance: {stats['present_today']} people present")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("🎯 Enhanced Attendance System")
print("Press 'q' to quit, 's' to show stats")
print(f"Cooldown period: {ATTENDANCE_COOLDOWN} seconds")
print(f"Min confidence: {MIN_CONFIDENCE}")

show_stats = True
last_stats_update = datetime.now()

while True:
    success, img = cap.read()
    if not success:
        continue
    
    # Update stats every 5 seconds
    if (datetime.now() - last_stats_update).seconds >= 5:
        stats = get_attendance_stats()
        last_stats_update = datetime.now()
    
    # Convert frame to PIL Image for MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    try:
        # Detect faces and get bounding boxes and cropped faces
        boxes, _ = mtcnn.detect(pil_img)
        
        if boxes is not None:
            # Get cropped faces for embedding
            try:
                cropped_faces = mtcnn(pil_img)
            except:
                cropped_faces = None
            
            for i, box in enumerate(boxes):
                # Extract face region coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                # Crop face for emotion detection
                face_img = img_rgb[y1:y2, x1:x2]
                if face_img.size > 0:
                    # Detect emotion using Haar cascade
                    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    emotion, emotion_confidence = detect_emotion(face_bgr)
                    
                    # Try to get face embedding if cropped faces are available
                    if cropped_faces is not None and i < len(cropped_faces):
                        try:
                            face_tensor = cropped_faces[i]
                            # Generate embedding for current face
                            with torch.no_grad():
                                current_embedding = resnet(face_tensor.unsqueeze(0).to(device))
                                current_embedding = current_embedding.cpu().numpy().flatten()
                            
                            # Compare with known faces
                            distances = [calculate_distance(current_embedding, known_encoding) 
                                       for known_encoding in encoded_list]
                            
                            min_distance = min(distances)
                            match_index = distances.index(min_distance)
                            
                            # Calculate confidence
                            recognition_confidence = calculate_confidence(min_distance)
                            # print(f"Recognition confidence: {recognition_confidence}")
                            # print(f"Min distance: {min_distance}")
                            # Threshold for face recognition
                            threshold = 0.6
                            
                            if min_distance < threshold:
                                name = class_names[match_index].upper()
                                
                                # Check attendance status
                                can_mark, status_msg = can_mark_attendance(name)
                                
                                # Get emotion color
                                emotion_color = get_emotion_color(emotion)
                                
                                # Draw rectangle and name
                                cv2.rectangle(img, (x1, y1), (x2, y2), emotion_color, 2)
                                cv2.rectangle(img, (x1, y2 - 90), (x2, y2), emotion_color, cv2.FILLED)
                                
                                # Display name, emotion, and status
                                cv2.putText(img, name, (x1 + 6, y2 - 65), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(img, f"{emotion} ({emotion_confidence:.2f})", (x1 + 6, y2 - 45), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.putText(img, f"Conf: {recognition_confidence:.2f}", (x1 + 6, y2 - 25), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Show attendance status
                                status_color = (0, 255, 0) if can_mark else (0, 165, 255)
                                status_text = "✓" if can_mark else "●"
                                cv2.putText(img, status_text, (x1 + 6, y2 - 5), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, status_color, 2)
                                
                                # Try to mark attendance
                                if can_mark:
                                    mark_attendance(name, emotion, emotion_confidence, recognition_confidence)
                                
                            else:
                                # Unknown face or low confidence
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.rectangle(img, (x1, y2 - 70), (x2, y2), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 45), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                                cv2.putText(img, f"{emotion} ({emotion_confidence:.2f})", (x1 + 6, y2 - 25), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.putText(img, f"Conf: {recognition_confidence:.2f}", (x1 + 6, y2 - 5), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        except Exception as e:
                            print(f"Error processing face {i}: {e}")
                            # Still show emotion detection even if face recognition fails
                            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 2)
                            cv2.rectangle(img, (x1, y2 - 50), (x2, y2), (128, 128, 128), cv2.FILLED)
                            cv2.putText(img, "PROCESSING", (x1 + 6, y2 - 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img, f"{emotion}", (x1 + 6, y2 - 10), 
                                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Show only emotion detection without face recognition
                        emotion_color = get_emotion_color(emotion)
                        cv2.rectangle(img, (x1, y1), (x2, y2), emotion_color, 2)
                        cv2.rectangle(img, (x1, y2 - 30), (x2, y2), emotion_color, cv2.FILLED)
                        cv2.putText(img, f"{emotion}", (x1 + 6, y2 - 10), 
                                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    
    except Exception as e:
        print(f"Error in main loop: {e}")
    
    # Display system info
    cv2.putText(img, "Enhanced Attendance System", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display attendance stats
    if show_stats:
        display_attendance_info(img, stats)
    
    cv2.imshow('Enhanced Face Recognition Attendance', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        show_stats = not show_stats
        print(f"Stats display: {'ON' if show_stats else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

# Display final statistics
final_stats = get_attendance_stats()
print("\n📊 Final Session Statistics:")
print(f"Total present today: {final_stats['present_today']}")
print("Emotion distribution:", final_stats['emotion_stats'])
print(f"Session data saved to: {SESSION_FILE}")
print(f"Attendance records saved to: {ATTENDANCE_FILE}")
