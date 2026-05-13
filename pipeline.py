import os
import uuid
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

def run_inference(model, images):
    device = torch.device("mps")
    model.to(device)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # HWC numpy/PIL → CHW float tensor, scales [0,255] → [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    # ImageNet mean/std, use these if your backbone was pretrained on ImageNet
                    std=[0.229, 0.224, 0.225])
    ])
    batch = torch.stack([
        transform(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))) for p in images
    ])

    with torch.no_grad():
        output = model(batch.to(device))  # (N, num_classes)
        probs = torch.softmax(output, dim=1)
        preds = probs.argmax(dim=1).tolist()  # list of N predicted class indices
        confs = probs.max(dim=1).values.tolist()

    idx_to_class = {
        0: "Bishop (B)",
        1: "Bishop (W)",
        2: "Empty (B)",
        3: "Empty (W)",
        4: "King (B)",
        5: "King (W)",
        6: "Knight (B)",
        7: "Knight (W)",
        8: "Pawn (B)",
        9: "Pawn (W)",
        10: "Queen (B)",
        11: "Queen (W)",
        12: "Rook (B)",
        13: "Rook (W)",
    }

    pred_labels = [idx_to_class[i] for i in preds]

    abbrev = {
        "Bishop (B)": "bB", "Bishop (W)": "wB",
        "Empty (B)": "_.", "Empty (W)": " .",
        "King (B)": "bK", "King (W)": "wK",
        "Knight (B)": "bN", "Knight (W)": "wN",
        "Pawn (B)": "bP", "Pawn (W)": "wP",
        "Queen (B)": "bQ", "Queen (W)": "wQ",
        "Rook (B)": "bR", "Rook (W)": "wR",
    }

    board = np.array([abbrev[l] for l in pred_labels]).reshape(8, 8)

    return board

def extract_chess_squares(img):
    if img is None:
        raise ValueError(f"Could not load image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape


    sq_height = height // 8
    sq_width = width // 8

    squares = []

    for y in range(8):
        for x in range(8):

            start_y = y * sq_height
            end_y = start_y + sq_height

            start_x = x * sq_width
            end_x = start_x + sq_width

            square_crop = img[start_y:end_y, start_x:end_x]
            squares.append(square_crop)

    return squares

def save_images(board, squares):
    reverse_abbrev = {
        "bB": "Bishop (B)", "wB": "Bishop (W)",
        " .": "Empty (W)", "_.": "Empty (B)",
        "bK": "King (B)", "wK": "King (W)",
        "bN": "Knight (B)", "wN": "Knight (W)",
        "bP": "Pawn (B)", "wP": "Pawn (W)",
        "bQ": "Queen (B)", "wQ": "Queen (W)",
        "bR": "Rook (B)", "wR": "Rook (W)",
    }

    for label, square in zip(board.flatten(), squares):
        class_name = reverse_abbrev[label]
        out_dir = os.path.join("data", "NewDataset", class_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(path, cv2.cvtColor(square, cv2.COLOR_RGB2BGR))

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

    height, width, _ = result.shape

    vert_crop = int((1 / 15) * height)
    horiz_crop = int((1 / 15) * width)

    result = result[vert_crop:-vert_crop, horiz_crop:-horiz_crop]

    return result



cap = cv2.VideoCapture(1)  # 0 = default camera
i = cv2.imread("img2.jpg")
model = torch.load("models/model.pth", map_location="mps", weights_only=False)

while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    cv2.imshow("stream", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):       # press S to capture
        print("Captured")
        board_img = extract_board(i)
        squares = extract_chess_squares(board_img)
        board = run_inference(model, squares)
        save_images(board, squares)
        for row in board:
            print(" ".join(row))
        cv2.waitKey(0)

    elif key == ord('q'):     # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()