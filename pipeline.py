import os
import uuid
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from PIL import ImageDraw

import chess.engine
import chess.svg
import chess

import cairosvg


def run_inference(model, images):
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
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
        2: "Empty",
        3: "King (B)",
        4: "King (W)",
        5: "Knight (B)",
        6: "Knight (W)",
        7: "Pawn (B)",
        8: "Pawn (W)",
        9: "Queen (B)",
        10: "Queen (W)",
        11: "Rook (B)",
        12: "Rook (W)",
    }

    pred_labels = [idx_to_class[i] for i in preds]

    abbrev = {
        "Bishop (B)": "b", "Bishop (W)": "B",
        "Empty": ".",
        "King (B)": "k", "King (W)": "K",
        "Knight (B)": "n", "Knight (W)": "N",
        "Pawn (B)": "p", "Pawn (W)": "P",
        "Queen (B)": "q", "Queen (W)": "Q",
        "Rook (B)": "r", "Rook (W)": "R",
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
        "b": "Bishop (B)", "B": "Bishop (W)",
        ".": "Empty",
        "k": "King (B)", "K": "King (W)",
        "n": "Knight (B)", "N": "Knight (W)",
        "p": "Pawn (B)", "P": "Pawn (W)",
        "q": "Queen (B)", "Q": "Queen (W)",
        "r": "Rook (B)", "R": "Rook (W)",
    }

    for label, square in zip(board.flatten(), squares):
        class_name = reverse_abbrev[label]
        out_dir = os.path.join("data", "NewDataset4", class_name)
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

def board_to_numpy(board):
    arr = np.full((8, 8), '.', dtype='<U2')
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            arr[7 - (sq // 8)][sq % 8] = piece.symbol()  # rank flip for visual orientation
    return arr

def board_to_img(board, size=480, lastmove=None):
    svg = chess.svg.board(board, lastmove=lastmove)
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    arr = np.frombuffer(png, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# Set up chess engine
engine = chess.engine.SimpleEngine.popen_uci("stockfish")
gboard = chess.Board()
gboard_img = board_to_img(gboard)

# Convert board from FEN to a Numpy Array
npboard = board_to_numpy(gboard)
for row in npboard:
    print(" ".join(row))


cap = cv2.VideoCapture("http://10.167.210.238:8080/video")  # IP Webcam URL

# ret, frame = cap.read()
# roi = cv2.selectROI("select board", frame, showCrosshair=True)
# x, y, w, h = roi
# cv2.destroyAllWindows()

# ======== Undo the camera lens distortion start
corners = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append([x, y])
        print(f"corner {len(corners)}: ({x}, {y})")

ret, frame = cap.read()
cv2.imshow("pick corners", frame)
cv2.setMouseCallback("pick corners", click)

while len(corners) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()
pts_src = np.float32(corners)

size = 1792  # output board size in pixels, 1792 = 224px per square
pts_dst = np.float32([
    [0,    0   ],
    [size, 0   ],
    [size, size],
    [0,    size],
])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# ======== Undo the camera lens distortion end

# change mps to cpu or cuda for non Apple Silicon devices
device = torch.device("cpu")
model = torch.load("models/model4.pth", map_location="cpu", weights_only=False)
model.to(device)

# main loop
last_move_text = None
while True:
    ret, frame = cap.read()
    # frame = frame[y:y + h, x:x + w] #cropping
    # fixing camera warp based on the warp transform we got earlier
    frame = cv2.warpPerspective(frame, M, (size, size))
    if not ret:
        break

    display_frame = cv2.resize(frame, (480, 480)) # smaller frame to show on the window (bigger one is used internally)
    header = np.zeros((40, 960, 3), dtype=np.uint8)
    pil_header = Image.fromarray(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_header)
    draw.text((10, 10), f"Engine played: {last_move_text}" if last_move_text else "Waiting...", fill=(255, 255, 255))
    header = cv2.cvtColor(np.array(pil_header), cv2.COLOR_RGB2BGR)

    row = np.hstack([display_frame, gboard_img])
    s = np.vstack([header, row])

    cv2.imshow("SIChess", s)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('c'):
        print("Captured")
        # board_img = extract_board(frame)
        board_img = frame
        squares = extract_chess_squares(board_img)
        board = run_inference(model, squares)
        for row in board:
            print(" ".join(row))

        # check to see if the current board matches one of the legal boards possible
        legal_flag = False
        for move in gboard.legal_moves:
            gboard.push(move)
            if np.array_equal(board_to_numpy(gboard), board):
                legal_flag = True
                break

            gboard.pop()  # restore


        if not legal_flag:
            print("Illegal board state detected.")
            break

        # Get chess engine to play the next move right after
        move = engine.play(gboard, chess.engine.Limit(time=0.1))
        gboard.push(move.move)
        last_move_text = str(move.move)

        save_images(board, squares)
        print(f"Engine played: {move.move}")
        gboard_img = board_to_img(gboard, lastmove=move.move)

engine.close()
cap.release()
cv2.destroyAllWindows()