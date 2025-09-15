import kociemba
import cv2
import numpy as np
import time
from drawLine import *
import winsound
def kociema(cube_str):
    """
    Solve a Rubik's cube given its cube string in Kociemba notation
    Args:
        cube_str: String of 54 characters using only URFDLB
    """
    # Validate cube string
    if len(cube_str) != 54:
        raise ValueError(f"Invalid cube string length: {len(cube_str)}. Expected 54.")
    
    if not all(c in 'URFDLB' for c in cube_str):
        raise ValueError("Invalid characters in cube string. Use only URFDLB.")
    
    # Get solution using kociemba
    try:
        solution = kociemba.solve(cube_str)
        result_string = solution.split()
    except Exception as e:
        print(f"Error solving cube: {e}")
        print("Cube string:", cube_str)
        return

    # Process the solution moves
    processed_moves = []
    for move in result_string:
        if len(move) > 1:
            if move[1] == '2':
                processed_moves.extend([move[0], move[0]])
            elif move == "B'":
                processed_moves.extend(["up", "U'", "down"])
        elif move == "B":
            processed_moves.extend(["up", "U", "down"])
        else:
            processed_moves.append(move)

    print(f"Solution steps: {processed_moves}")

    cap = cv2.VideoCapture(0)
    frame_idx = 0
    frame_reset_cnt = 120
    result_index = 0
    cubeshape = 180
    cubesize = 3
    startpoint = (100, 100)

    last_cube_state = None
    correct_turn = True

    while True:
        cubeImg = np.zeros((480,640))
        res, cubeImg = cap.read()
        if not res or cubeImg is None:
            print("Camera error: Unable to read frame.")
            break

        drawCube(cubeImg, cubeshape, cubesize, startpoint)

        # Show arrow/instruction for current move before "Right! Next"
        if result_index < len(processed_moves):
            result = processed_moves[result_index]
            # Draw arrows/instructions BEFORE incrementing result_index
            if result == "D":
                arrowlines(cubeImg,(2,0),(2,2), cubeshape, cubesize)
            elif result == "D'":
                arrowlines(cubeImg,(2,2),(2,0), cubeshape, cubesize)
            elif result == 'F':
                rotation(cubeImg)
            elif result == "F'":
                antirotation(cubeImg)
            elif result == 'R':
                arrowlines(cubeImg,(2,2),(0,2), cubeshape, cubesize)
            elif result == "R'":
                arrowlines(cubeImg,(0,2),(2,2), cubeshape, cubesize)
            elif result == 'U':
                arrowlines(cubeImg,(0,2),(0,0), cubeshape, cubesize)
            elif result == "U'":
                arrowlines(cubeImg,(0,0),(0,2), cubeshape, cubesize)
            elif result == 'L':
                arrowlines(cubeImg,(0,0),(2,0), cubeshape, cubesize)
            elif result == "L'":
                arrowlines(cubeImg,(2,0),(0,0), cubeshape, cubesize)
            elif result == "down":
                arrowlines(cubeImg,(0,0),(2,0), cubeshape, cubesize)
                arrowlines(cubeImg,(0,1),(2,1), cubeshape, cubesize)
                arrowlines(cubeImg,(0,2),(2,2), cubeshape, cubesize)
            elif result == "up":
                arrowlines(cubeImg,(2,0),(0,0), cubeshape, cubesize)
                arrowlines(cubeImg,(2,1),(0,1), cubeshape, cubesize)
                arrowlines(cubeImg,(2,2),(0,2), cubeshape, cubesize)

        if result_index >= len(processed_moves):
            cv2.putText(cubeImg, "DONE!!! Solved the Cube", (int(cubeImg.shape[1]/4), 30), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
            cv2.imshow("cube", cubeImg)
            if cv2.waitKey(10) == ord('q'):
                break
            continue

        frame_idx += 1
        if frame_idx > frame_reset_cnt:
            winsound.Beep(500,300)
            # Check if cube state matches expected move
            # --- Feature: Detect if cube is turned correctly ---
            # This is a placeholder: you must implement a function to scan the cube and compare with expected state.
            # For demonstration, we simulate correct_turn = True (replace with your own detection logic)
            correct_turn = True  # Replace with actual detection

            if correct_turn:
                cv2.putText(cubeImg, "Right! Next", (int(cubeImg.shape[1]/4), 60), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 3)
                cv2.imshow("cube", cubeImg)
                cv2.waitKey(1000)  # Show "Right! Next" for 1 second
                result_index += 1
            else:
                cv2.putText(cubeImg, "Wrong Turn! Repeat", (int(cubeImg.shape[1]/4), 60), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 3)
                cv2.imshow("cube", cubeImg)
                cv2.waitKey(1000)  # Show "Wrong Turn! Repeat" for 1 second
            frame_idx = 0
            continue  # Skip rest of loop to avoid flicker

        cv2.imshow("cube", cubeImg)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example scrambled cube string (not solved)
    scrambled_cube_str = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    # Replace above with an actual scrambled string for real solving
    kociema(scrambled_cube_str)
