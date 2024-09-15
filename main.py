import cv2
import mediapipe as mp
import numpy as np
import pygame
import ctypes
import random
import time

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


# List of landmarks to track
t_pose = [
    # Store the name of the pose
    "TPose",
    # Check if arm is out straight
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",

    # Check if legs are not bent
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",

    # Check if 90-degree angle from hip to shoulder to hand
    "LEFT_WRIST", "LEFT_SHOULDER", "LEFT_HIP",
    "RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP"

]

flex_pose = [
    "FlexPose",

    # Check 90-degree bent arm
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",

]

all_poses = [flex_pose, t_pose]


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def track_angle(pose, resized_frame, screen_width, screen_height, landmks_list):
    """
    Process pose landmarks and calculate the angle between nodeOne, nodeTwo, and nodeThree.

    Parameters:
    - pose: Mediapipe pose object
    - resized_frame: The resized frame to process
    - screen_width: Screen width for landmark scaling
    - screen_height: Screen height for landmark scaling
    - landmks_list: List of landmarks

    Returns:
    - angles between the three points
    """
    angles = []
    image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Draw landmarks and calculate the angle
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        for i in range(1, len(landmks_list), 3):
            # Use getattr to dynamically access the correct PoseLandmark attributes
            shoulder = [landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i]).value].x * screen_width,
                        landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i]).value].y * screen_height]
            elbow = [landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i + 1]).value].x * screen_width,
                     landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i + 1]).value].y * screen_height]
            wrist = [landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i + 2]).value].x * screen_width,
                     landmarks[getattr(mp_pose.PoseLandmark, landmks_list[i + 2]).value].y * screen_height]

            # Calculate the angle
            angle1 = calculate_angle(shoulder, elbow, wrist)

            # Append calculated angle to the list
            angles.append(angle1)


    return angles


# Button class
class Button:
    def __init__(self, text, x, y, width, height, color, text_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text_color = text_color
        self.original_color = color  # Store the original color
        self.clicked = False

    def draw(self, screen):
        button_color = RED if self.clicked else self.original_color
        pygame.draw.rect(screen, button_color, self.rect)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

def instructions_menu():
    running_instruction = True
    while running_instruction:
        screen.fill(WHITE)

        instructions_text = small_font.render("Press ESC to go back", True, BLACK)
        screen.blit(instructions_text, (150, 180))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_instruction = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running_instruction = False  # Return to main menu when ESC is pressed

        pygame.display.update()

def main_menu():
    running_menu = True
    while running_menu:
        screen.fill(WHITE)

        # Draw buttons
        start_button.draw(screen)
        instructions_button.draw(screen)
        quit_button.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_menu = False

            # Check if buttons are clicked
            if start_button.is_clicked(event):
                game_loop()  # Switch to game loop
            if instructions_button.is_clicked(event):
                instructions_menu()  # Switch to instructions screen
            if quit_button.is_clicked(event):
                running_menu = False

        pygame.display.update()

def pose_detection(angles, pose):
    if pose is not None:
        if pose[0] == "TPose":
            if 170 < angles[0] <= 185 and 170 < angles[1] <= 185 and 170 < angles[
                2] <= 185 and 170 < angles[3] <= 185 and 80 < angles[4] <= 110 and 80 < \
                    angles[5] <= 110:
                return "You are T Posing!"
        elif pose[0] == "FlexPose":
            if 60 < angles[0] <= 90 and 60 < angles[1] <= 90:
                return "You are Flexing!"
    return None



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



# Define fonts
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 36)


# Create buttons for Start Menu
start_button = Button("Start", 200, 100, 200, 50, GREEN, BLACK)
instructions_button = Button("Instructions", 200, 180, 200, 50, GREEN, BLACK)
quit_button = Button("Quit", 200, 260, 200, 50, RED, BLACK)

# Create buttons for Game
t_pose_button = Button("T-pose", 300, 100, 200, 50, GREEN, BLACK)
flex_pose_button = Button("Flex-pose", 300, 180, 200, 50, GREEN, BLACK)



def game_loop():
    # Set the countdown time in seconds
    countdown_time = 8  # 5 seconds countdown 3 second pose time
    selected_pose = None
    pose_active = False
    random_pose = None

    # Setup MediaPipe pose instance
    with (mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose):
        clock = pygame.time.Clock()
        running = True

        # Tracking state
        tracking_active = False  # Angle tracking initially inactive

        # Start the timer
        start_ticks = pygame.time.get_ticks()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_0:
                    running = False
                    main_menu()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Press space to toggle angle tracking
                        tracking_active = not tracking_active
                    if not tracking_active:
                        # Reset selected pose if tracking is deactivated
                        selected_pose = None
                        t_pose_button.clicked = False
                        flex_pose_button.clicked = False
                if t_pose_button.is_clicked(event):
                    if tracking_active:
                        t_pose_button.clicked = not t_pose_button.clicked
                        if t_pose_button.clicked:
                            selected_pose = t_pose
                        else:
                            selected_pose = None
                if flex_pose_button.is_clicked(event):
                    if tracking_active:
                        flex_pose_button.clicked = not flex_pose_button.clicked
                        if flex_pose_button.clicked:
                            selected_pose = flex_pose
                        else:
                            selected_pose = None

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB once and resize to the screen resolution if needed
            resized_frame = cv2.resize(frame, (screen_width, screen_height))

            # Store the array that contains the angles measured
            returned_angle_list = []

            # Process pose and calculate angle if tracking is active
            if tracking_active and selected_pose is not None:
                returned_angle_list = track_angle(pose, resized_frame, screen_width, screen_height, selected_pose)

            # Flip the frame horizontally to mirror it
            mirrored_frame = cv2.flip(resized_frame, 1)

            # Then, create a Pygame surface from the mirrored frame
            frame_surface = pygame.surfarray.make_surface(mirrored_frame.swapaxes(0, 1))

            # ---- Countdown Logic ----
            # Calculate the remaining time
            seconds = countdown_time - (pygame.time.get_ticks() - start_ticks) // 1000

            text = font.render("", True, BLACK)

            # Render the countdown text
            if seconds > 3:
                countdown_text = font.render(str(seconds - 3), True, BLACK)
            elif  0 < seconds <= 3:
                countdown_text = font.render("Pose!", True, BLACK)
                if random_pose is not None:
                    text = font.render(random_pose[0], True, BLACK)
                if not pose_active:
                    random_pose = random.choice(all_poses)
                pose_active = True
                selected_pose = random_pose
            else:
                countdown_time += 8
                pose_active = False

            # Display the countdown text at the center of the screen
            countdown_text_rect = countdown_text.get_rect(center=(screen_width // 2, 100))
            screen.blit(frame_surface, (0, 0))  # First draw the frame
            screen.blit(countdown_text, countdown_text_rect)  # Then overlay the countdown

            # Display random pose text will be empty if not defined
            text_rect = text.get_rect(center=(screen_width // 2, 140))
            screen.blit(text, text_rect)

            # ---- End of Countdown Logic ----

            # Only track and calculate angles when tracking is active and a pose is selected
            if tracking_active and selected_pose:
                angles = track_angle(pose, resized_frame, screen_width, screen_height, selected_pose)
                pose_text = pose_detection(angles, selected_pose)

                # If there was a pose detected print on screen
                if pose_text:
                    pose_text_display = font.render(pose_text, True, BLACK)

                    # Display text at the center of the screen
                    t_pose_text_rect = pose_text_display.get_rect(center=(screen_width // 2, screen_height // 2))
                    screen.blit(pose_text_display, t_pose_text_rect)

                # Draw angles on the screen using Pygame (this ensures they're not rotated)
                for idx, angle in enumerate(returned_angle_list):
                    angle_text = font.render(f"Angle {idx+1}: {int(angle)}", True, BLACK)
                    screen.blit(angle_text, (50, 100 + idx * 50))

            # Display buttons
            t_pose_button.draw(screen)
            flex_pose_button.draw(screen)

            # Indicate whether tracking is active or not
            tracking_text = "Tracking: ON" if tracking_active else "Tracking: OFF"
            tracking_status = small_font.render(tracking_text, True, GREEN if tracking_active else RED)
            screen.blit(tracking_status, (screen_width - 200, 10))

            # Display everything on the Pygame screen
            pygame.display.flip()

            # Limit FPS to 30
            clock.tick(30)

        cap.release()
        pygame.quit()



main_menu()
