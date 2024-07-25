import pyautogui

# Set the step size for mouse movement
step_size = 1

# Function to move the mouse up
def move_up():
    pyautogui.moveRel(0, -step_size, duration=0.1)

# Function to move the mouse down
def move_down():
    pyautogui.moveRel(0, step_size, duration=0.1)

# Function to move the mouse left
def move_left():
    pyautogui.moveRel(-step_size, 0, duration=0.1)

# Function to move the mouse right
def move_right():
    pyautogui.moveRel(step_size, 0, duration=0.1)

# Main loop to listen for keyboard input
while True:
    # Get the key pressed
    key = input("Press a key (up, down, left, right) or 'q' to quit: ")

    # Perform the corresponding mouse movement
    if key == "w":
        move_up()
    elif key == "s":
        move_down()
    elif key == "a":
        move_left()
    elif key == "d":
        move_right()
    elif key == "q":
        break
    else:
        print("Invalid key. Try again.")