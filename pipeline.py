import cv2
import numpy as np

def extract_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)

    threshold = 0.3 * mask.shape[1] * 255
    rows = np.where(row_sums > threshold)[0]
    cols = np.where(col_sums > threshold)[0]

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]

    result = img[y1:y2, x1:x2]
    return result



cap = cv2.VideoCapture(0)  # 0 = default camera
i = cv2.imread("20260512_234129.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("stream", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):       # press S to capture
        print("Captured")
        cv2.imshow("board", extract_board(img=i))
        cv2.waitKey(0)

    elif key == ord('q'):     # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()