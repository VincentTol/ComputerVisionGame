import cv2
import mediapipe as mp
import numpy as np
import pygame
import ctypes
import time

# Initialize Pygame
pygame.init()

# Get screen resolution using Ctypes
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

print(f"Screen Resolution: {screen_width}x{screen_height}")

# Create a fullscreen window using the full resolution
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("MediaPipe with Pygame")

# Mediapipe solutions for pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

# Set camera resolution (match screen resolution if camera supports it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def track_angle(pose, resized_frame, screen_width, screen_height, nodeOne, nodeTwo, nodeThree):
    """
    Process pose landmarks and calculate the angle between nodeOne, nodeTwo, and nodeThree.

    Parameters:
    - pose: Mediapipe pose object
    - resized_frame: The resized frame to process
    - screen_width: Screen width for landmark scaling
    - screen_height: Screen height for landmark scaling
    - nodeOne, nodeTwo, nodeThree: The pose landmarks to track (as strings, e.g., 'LEFT_SHOULDER')

    Returns:
    - frame with the angle text drawn on it
    """
    angle = 0
    image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Draw landmarks and calculate the angle
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Use getattr to dynamically access the correct PoseLandmark attributes
        shoulder = [landmarks[getattr(mp_pose.PoseLandmark, nodeOne).value].x * screen_width,
                    landmarks[getattr(mp_pose.PoseLandmark, nodeOne).value].y * screen_height]
        elbow = [landmarks[getattr(mp_pose.PoseLandmark, nodeTwo).value].x * screen_width,
                 landmarks[getattr(mp_pose.PoseLandmark, nodeTwo).value].y * screen_height]
        wrist = [landmarks[getattr(mp_pose.PoseLandmark, nodeThree).value].x * screen_width,
                 landmarks[getattr(mp_pose.PoseLandmark, nodeThree).value].y * screen_height]

        # Calculate the angle
        angle = calculate_angle(shoulder, elbow, wrist)

    return angle





# Setup MediaPipe pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    clock = pygame.time.Clock()
    frame_count = 0
    start_time = time.time()
    fps_interval = 2  # Increase interval to reduce FPS calculation overhead
    running = True

    # Create surface once, reuse for each frame
    frame_surface = pygame.Surface((screen_width, screen_height))

    # Tracking state
    tracking_active = False  # Angle tracking initially inactive

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Press space to toggle angle tracking
                    tracking_active = not tracking_active

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB once and resize to the screen resolution if needed
        resized_frame = cv2.resize(frame, (screen_width, screen_height))

        returned_angle = 0
        # Process pose and calculate angle if tracking is active
        if tracking_active:
            returned_angle = track_angle(pose, resized_frame, screen_width, screen_height, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")

        cv2.putText(resized_frame, str(int(returned_angle)),
                    (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rotate and update the frame in Pygame
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Update the Pygame surface
        pygame.surfarray.blit_array(frame_surface, rotated_frame)

        # Draw FPS
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= fps_interval:
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time
            fps_text = pygame.font.SysFont(None, 36).render(f"FPS: {int(fps)}", True, (255, 255, 255))
            frame_surface.blit(fps_text, (10, 10))

        # Indicate whether tracking is active or not
        tracking_text = "Tracking: ON" if tracking_active else "Tracking: OFF"
        tracking_status = pygame.font.SysFont(None, 36).render(tracking_text, True,
                                                               (0, 255, 0) if tracking_active else (255, 0, 0))
        frame_surface.blit(tracking_status, (screen_width - 200, 10))

        # Display the frame on the Pygame screen
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Limit FPS to 30
        clock.tick(30)

    cap.release()
    pygame.quit()
