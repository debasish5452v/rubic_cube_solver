# Rubik's Cube Solver

This project provides a computer vision-based Rubik's Cube scanner and solver using Python and OpenCV.  
It detects the colors of a physical Rubik's Cube, converts the state to Kociemba notation, and displays step-by-step solution instructions.

## Features

- Webcam-based scanning of all 6 cube faces
- Robust color detection using RGB, HSV, and LAB color spaces
- Visual feedback for scanned colors and scan order
- Validates cube state before solving
- Uses the Kociemba algorithm to compute the optimal solution
- Step-by-step move instructions with visual arrows
- Detects if the cube is turned correctly for each move

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- imutils
- webcolors
- kociemba

Install dependencies:
```bash
pip install opencv-python numpy imutils webcolors kociemba
```

## Usage

1. Run the main solver script:
    ```bash
    python rubik_cube_solver.py
    ```
2. Follow the on-screen instructions to scan each face in the correct order.
3. After scanning all faces, press `s` to start solving.
4. Follow the visual arrows and instructions to solve your cube.

## File Structure

- `rubik_cube_solver.py` — Main scanning and solving logic
- `kociema_module.py` — Kociemba solver and move instruction display
- `drawLine.py` — Drawing functions for cube grid and arrows
- `README.md` — Project documentation

## Troubleshooting

- If the webcam is not detected, the program will try camera indices 0, 1, and 2.
- If color detection fails, ensure good lighting and adjust cube position.
- If you get "Unrecognized center color", check your cube stickers and calibrate thresholds in `getcolor()`.
