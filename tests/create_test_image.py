#!/usr/bin/env python3
"""
Create a test image for main.py testing
"""
import cv2
import numpy as np

def create_test_image():
    """Create a realistic test image with a person"""
    width, height = 640, 480
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Create a more realistic human figure
    person_x = width // 2 - 60
    person_y = height // 2 - 120
    person_w = 120
    person_h = 240
    
    # Body
    cv2.rectangle(img, (person_x, person_y + 40), (person_x + person_w, person_y + person_h), (120, 150, 180), -1)
    
    # Head
    cv2.circle(img, (person_x + person_w//2, person_y + 25), 20, (200, 180, 150), -1)
    
    # Arms
    cv2.rectangle(img, (person_x - 15, person_y + 60), (person_x + 15, person_y + 140), (100, 130, 160), -1)
    cv2.rectangle(img, (person_x + person_w - 15, person_y + 60), (person_x + person_w + 15, person_y + 140), (100, 130, 160), -1)
    
    # Legs
    cv2.rectangle(img, (person_x + 20, person_y + 180), (person_x + 50, person_y + person_h + 30), (90, 120, 150), -1)
    cv2.rectangle(img, (person_x + 70, person_y + 180), (person_x + 100, person_y + person_h + 30), (90, 120, 150), -1)
    
    return img

if __name__ == "__main__":
    img = create_test_image()
    cv2.imwrite("test_person.jpg", img)
    print("✅ Created test_person.jpg")
