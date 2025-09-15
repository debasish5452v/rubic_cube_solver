import cv2
import numpy as np
try:
    from imutils import contours
    from webcolors import rgb_to_name
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'imutils', 'webcolors'])
    from imutils import contours
    from webcolors import rgb_to_name
from kociema_module import *
color = []
cubecolor = (0,0,0)
cubelineSize = 2
def getcolor(r, g, b):
    """Robust color detection: use mean, median, HSV, LAB, and fallback to central region for difficult sides."""
    print(f"Raw RGB: ({r}, {g}, {b})")
    min_brightness = 40
    if r < min_brightness and g < min_brightness and b < min_brightness:
        print("Too dark - adjust lighting")
        return None

    # --- RGB-based detection (mean and median) ---
    if r > 200 and g > 200 and b > 200 and max(abs(r-g), abs(r-b), abs(g-b)) < 30:
        return 'w'
    if r > 225 and g > 220 and b < 50 and abs(r-g) < 10:
        return 'y'
    if r > 210 and 140 < g < 200 and b < 70 and 15 < (r-g) < 45:
        return 'o'
    if r > 200 and g < 110 and b < 90 and (r-g) > 60:
        return 'r'
    if g > 150 and r < 120 and b < 120:
        return 'g'
    if b > 150 and r < 100 and g < 150:
        return 'b'

    # --- HSV-based fallback ---
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    print(f"HSV: ({h}, {s}, {v})")
    if s < 40 and v > 180:
        return 'w'
    if 24 <= h <= 34 and s > 80 and v > 120:
        return 'y'
    if 13 <= h <= 21 and s > 70 and v > 100:
        return 'o'
    if (h <= 9 or h >= 160) and s > 60 and v > 80:
        return 'r'
    if 40 <= h <= 85 and s > 60 and v > 80:
        return 'g'
    if 90 <= h <= 140 and s > 60 and v > 80:
        return 'b'

    # --- LAB-based fallback ---
    lab = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0][0]
    l, a, b_lab = lab
    print(f"LAB: ({l}, {a}, {b_lab})")
    if l > 200 and abs(a-128) < 10 and abs(b_lab-128) < 10:
        return 'w'
    if l > 180 and b_lab > 160 and a < 140:
        return 'y'
    if 140 < l < 200 and 150 < b_lab < 180 and 140 < a < 160:
        return 'o'
    if l > 120 and a > 160 and b_lab < 140:
        return 'r'
    if l > 120 and a < 110 and b_lab < 140:
        return 'g'
    if l > 120 and a < 120 and b_lab > 160:
        return 'b'

    # --- Fallback: try central region of the cell for color detection ---
    # This fallback should be used in your cell scanning loop if getcolor returns None.
    # Example usage in your scanning loop:
    # if res is None:
    #     h, w = cell.shape[:2]
    #     central = cell[h//4:3*h//4, w//4:3*w//4]
    #     b_c, g_c, r_c = [int(x) for x in cv2.mean(central)[:3]]
    #     res = getcolor(r_c, g_c, b_c)

    print(f"Unrecognized color: R={r}, G={g}, B={b}, HSV=({h},{s},{v}), LAB=({l},{a},{b_lab})")
    return None

def draw_detected_colors(img, face_colors, start_point=(100,60), cell_size=60):
    """Draw detected colors on the image for visual feedback"""
    colors = {
        'r': (0, 0, 255),     # Red in BGR
        'g': (0, 255, 0),     # Green in BGR
        'b': (255, 0, 0),     # Blue in BGR
        'y': (0, 255, 255),   # Yellow in BGR
        'w': (255, 255, 255), # White in BGR
        'o': (0, 165, 255)    # Orange in BGR
    }
    
    for i, color in enumerate(face_colors):
        if color in colors:
            row = i // 3
            col = i % 3
            x = start_point[0] + col * cell_size
            y = start_point[1] + row * cell_size
            cv2.rectangle(img, 
                        (x+2, y+2), 
                        (x+cell_size-2, y+cell_size-2), 
                        colors[color], 
                        -1)  # -1 fills the rectangle
            # Add color letter for clarity
            cv2.putText(img, color.upper(), 
                       (x+20, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0,0,0), 2)
            
def drawCube(img, cubesize, cubeshape, start_point): # start_poing (100, 60)
    cubecell = int(cubesize / cubeshape)
    # draw horizontal lines first
    for i in range(cubeshape + 1):
        start_line = (start_point[0], start_point[1] + i * cubecell)
        end_line = (start_point[0] + cubesize, start_point[1] + i * cubecell)
        cv2.line(img, start_line, end_line, cubecolor, 2)
    
    for i in range(cubeshape + 1):
        start_line = (start_point[0] + i * cubecell, start_point[1])
        end_line = (start_point[0] + i * cubecell, start_point[1] + cubesize)
        cv2.line(img, start_line, end_line, cubecolor, cubelineSize)

    return img

def convert_to_kociemba_notation(colors):
    """
    Convert detected colors to Kociemba notation.
    Standard mapping (assuming centers):
    U: y (yellow)
    R: o (orange)
    F: g (green)
    D: w (white)
    L: r (red)
    B: b (blue)
    """
    notation_map = {
        'y': 'U',  # Up (yellow)
        'o': 'R',  # Right (orange)
        'g': 'F',  # Front (green)
        'w': 'D',  # Down (white)
        'r': 'L',  # Left (red)
        'b': 'B',  # Back (blue)
    }
    return ''.join(notation_map.get(c, 'X') for c in colors)  # 'X' for unknowns

def reorder_faces_for_kociemba(colors):
    """
    Reorder scanned faces from [U, D, F, L, R, B] to [U, R, F, D, L, B] for Kociemba.
    Each face has 9 stickers.
    Your scan order: ['y', 'w', 'g', 'r', 'o', 'b']
    Kociemba expects: [U, R, F, D, L, B] = [y, o, g, w, r, b]
    """
    # Indices for each face in your scanned order
    # U: 0-8 (y), D: 9-17 (w), F: 18-26 (g), L: 27-35 (r), R: 36-44 (o), B: 45-53 (b)
    U = colors[0:9]    # y
    D = colors[9:18]   # w
    F = colors[18:27]  # g
    L = colors[27:36]  # r
    R = colors[36:45]  # o
    B = colors[45:54]  # b
    # Kociemba expects: U, R, F, D, L, B
    return U + R + F + D + L + B

def validate_cube_state(colors):
    """Validate the scanned cube state"""
    # Count each color - should have exactly 9 of each
    color_counts = {'y':0, 'b':0, 'r':0, 'g':0, 'o':0, 'w':0}
    for c in colors:
        if c in color_counts:
            color_counts[c] += 1
    
    # Check counts
    for color, count in color_counts.items():
        if count != 9:
            return False, f"Found {count} {color} squares, expected 9"
            
    # Check centers
        centers = [colors[4], colors[13], colors[22], colors[31], colors[40], colors[49]]
        expected = ['y', 'w', 'g', 'r', 'o', 'b']
        if centers != expected:
            return False, f"Center colors don't match expected order. Found: {centers}, Expected: {expected}"
        
    return True, "Valid cube state"

def showlable(img,index): 
    scan_instructions = [
        "SCAN ORDER:",
        "1. Top (Up) - Yellow center",
        "2. Bottom (Down) - White center",
        "3. Front - Green center",
        "4. Left - Red center",
        "5. Right - Orange center",
        "6. Back - Blue center",
        "",
        "Press 'c' to scan each face",
        "Press 'r' to rescan last face",
        "Follow the order above!"
    ]
    # Show which faces have been scanned
    face_names = {'y': 'Top (Yellow)', 'w': 'Bottom (White)', 'g': 'Front (Green)', 'r': 'Left (Red)', 'o': 'Right (Orange)', 'b': 'Back (Blue)'}
    scanned = [face_names[c] for c in scanned_faces.keys()]
    not_scanned = [face_names[c] for c in ['y','w','g','r','o','b'] if c not in scanned_faces]
    status_text = f"Scanned: {', '.join(scanned) if scanned else 'None'}"
    status_text2 = f"To scan: {', '.join(not_scanned) if not_scanned else 'None'}"
    step_instructions = {
        1: "Step 1: Hold cube with YELLOW center on top (UP)\n(Arrow: Up)",
        2: "Step 2: Flip cube to show WHITE center (DOWN)\n(Arrow: Down)",
        3: "Step 3: Show GREEN center (FRONT)\n(Arrow: Front)",
        4: "Step 4: Show RED center (LEFT)\n(Arrow: Left)",
        5: "Step 5: Show ORANGE center (RIGHT)\n(Arrow: Right)",
        6: "Step 6: Show BLUE center (BACK)\n(Arrow: Back)"
    }
    # Move scan instructions to the bottom
    img_height = img.shape[0]
    for i, text in enumerate(scan_instructions):
        cv2.putText(img, text, (17, img_height - 180 + i*22), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,255) if i==0 else (0,0,0), 1)
    # Show scan status below instructions
    cv2.putText(img, status_text, (17, img_height - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,128,0), 2)
    cv2.putText(img, status_text2, (17, img_height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    # Show current step instruction just above the scan instructions
    cv2.putText(img, step_instructions.get(index, ""), (17, img_height - 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)

    # Draw a visual arrow for the current scan step
    arrow_color = (0, 0, 255)
    arrow_thickness = 4
    arrow_start = (320, 120)
    arrow_length = 80
    if index == 1:  # Up
        arrow_end = (320, 120 - arrow_length)
    elif index == 2:  # Down
        arrow_end = (320, 120 + arrow_length)
    elif index == 3:  # Front
        arrow_end = (320 + arrow_length, 120)
    elif index == 4:  # Left
        arrow_end = (320 - arrow_length, 120)
    elif index == 5:  # Right
        arrow_end = (320 + arrow_length, 120)
    elif index == 6:  # Back
        arrow_end = (320, 120)
    else:
        arrow_end = (320, 120)
    cv2.arrowedLine(img, arrow_start, arrow_end, arrow_color, arrow_thickness, tipLength=0.3)
    cv2.imshow("cube",img)

def adjust_brightness(img):
    """Improve color detection with better image preprocessing"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# Try all camera indices automatically until one works
for cam_index in [0, 1, 2]:
    cap = cv2.VideoCapture(cam_index)
    res, test_img = cap.read()
    if res and test_img is not None:
        print(f"Using camera index {cam_index}")
        break
    cap.release()
else:
    print("Camera error: Unable to read frame from any index.")
    print("Troubleshooting tips:")
    print("- Make sure your webcam is connected and not used by another application.")
    print("- Try restarting your computer.")
    exit()

index = 1
color = []  # Reset color list at start
scanned_faces = {}  # Track scanned faces by center color

while True:
    res, cubeImg = cap.read()
    if not res or cubeImg is None:
        print("Camera error: Unable to read frame.")
        break

    # Draw cube grid and show labels
    drawCube(cubeImg,180,3,(100,60))
    showlable(cubeImg, index)
    cv2.imshow("cube",cubeImg)
    
    # Single waitKey for all key events
    key = cv2.waitKey(1) & 0xFF

    # Rescan option: remove last scanned face and allow rescan
    if key == ord('r'):
        if len(scanned_faces) > 0:
            last_face = list(scanned_faces.keys())[-1]
            del scanned_faces[last_face]
            print(f"Rescanning face {last_face.upper()}. Please scan again.")
            index = len(scanned_faces) + 1
        else:
            print("No face to rescan yet.")

    if key == ord('c'):
        print("Start processing\nScanning face...")
        img = cubeImg.copy()
        face_colors = []
        uncertain_cells = []

        # Define the grid cells
        cells = [
            (60,120,100,160), (60,120,160,220), (60,120,220,280),
            (120,180,100,160), (120,180,160,220), (120,180,220,280),
            (180,240,100,160), (180,240,160,220), (180,240,220,280)
        ]

        # Process the center square first to identify the face
        center_y1, center_y2 = 120, 180
        center_x1, center_x2 = 160, 220
        center_cell = img[center_y1:center_y2, center_x1:center_x2]
        b_avg, g_avg, r_avg = [int(x) for x in cv2.mean(center_cell)[:3]]
        center_color = getcolor(r_avg, g_avg, b_avg)

        if center_color not in ['y', 'b', 'r', 'g', 'o', 'w']:
            print(f"Warning: Unrecognized center color: {center_color}")
            print("Please show a valid cube face and ensure good lighting.")
            continue

        # Don't allow duplicate scans for the same center
        if center_color in scanned_faces:
            print(f"Face with center {center_color} already scanned.")
            continue

        # Process each cell with robust fallback
        for idx, (y1,y2,x1,x2) in enumerate(cells):
            cell = img[y1:y2, x1:x2]
            h, w = cell.shape[:2]
            center = cell[h//3:2*h//3, w//3:2*w//3]
            b, g, r = cv2.split(center)
            r_med = int(np.median(r))
            g_med = int(np.median(g))
            b_med = int(np.median(b))
            r_avg = int(np.mean(r))
            g_avg = int(np.mean(g))
            b_avg = int(np.mean(b))
            res = getcolor(r_med, g_med, b_med)
            print(f"Cell {idx+1}: Median RGB=({r_med},{g_med},{b_med}), Mean RGB=({r_avg},{g_avg},{b_avg}), Detected={res}")
            if res is None:
                res = getcolor(r_avg, g_avg, b_avg)
                print(f"Cell {idx+1}: Fallback to mean, Detected={res}")
            if res is None:
                # Try full cell mean as last resort
                b_full, g_full, r_full = [int(x) for x in cv2.mean(cell)[:3]]
                res = getcolor(r_full, g_full, b_full)
                print(f"Cell {idx+1}: Fallback to full cell mean RGB=({r_full},{g_full},{b_full}), Detected={res}")
            if res is None:
                print(f"Warning: Could not detect color at position {idx+1}. Marking as '?'")
                face_colors.append('?')
                uncertain_cells.append(idx+1)
            else:
                face_colors.append(res)

        draw_detected_colors(cubeImg, face_colors)
        cv2.imshow("cube", cubeImg)
        cv2.waitKey(1000)
        if face_colors.count('?') == 0 and len(face_colors) == 9:
            scanned_faces[center_color] = face_colors
            print(f"Face {center_color.upper()} scanned successfully")
            print(f"Colors detected: {face_colors}")
            print(f"Total faces scanned: {len(scanned_faces)}/6")
            index = len(scanned_faces) + 1
        else:
            print(f"Face scan completed with {face_colors.count('?')} uncertain cells: {uncertain_cells}")
            print(f"Colors detected: {face_colors}")
            print("Try adjusting lighting or cube position and scan again if needed.")

    elif key == ord('s'): # start kociema module                       
        if len(scanned_faces) == 6:
            try:
                # Order: U, R, F, D, L, B (user's scan order, swapped left/right)
                face_order = ['y', 'w', 'g', 'r', 'o', 'b']
                color = []
                for fc in face_order:
                    if fc not in scanned_faces:
                        raise ValueError(f"Missing face with center {fc}")
                    color.extend(scanned_faces[fc])

                # For validation, expected centers: ['y', 'w', 'g', 'o', 'r', 'b']
                is_valid, message = validate_cube_state(color)
                if not is_valid:
                    raise ValueError(message)

                ordered_colors = reorder_faces_for_kociemba(color)
                cube_str = convert_to_kociemba_notation(ordered_colors)
                print(f"Cube string: {cube_str}")

                kociema(cube_str)
                scanned_faces.clear()
            except Exception as e:
                print(f"Error: {e}")
                print("\nScan all 6 faces (any order).")
                print("Current faces:", list(scanned_faces.keys()))
                scanned_faces.clear()
        else:
            print(f"Not enough faces detected ({len(scanned_faces)}/6). Please scan all faces.")

    elif key == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()