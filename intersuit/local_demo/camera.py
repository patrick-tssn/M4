import cv2

def get_available_cameras():
    available_cameras = []
    # Check for 5 cameras in local
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

cap = cv2.VideoCapture('http://10.1.101.4:1234')
cnt = 0
while True:
    ret, frame = cap.read()
    cnt += 1
    if not ret:
        print("no camera detected")
        break
    if cnt == 10:
        cv2.imwrite("ouput.jpg", frame)
    # Process the frame
    # cv2.imshow('Frame', frame)
    print("frame: ", ret)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
