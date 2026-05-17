import os
import uuid

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import chess.engine
import chess.svg
import chess
import cairosvg

MODEL_PATH = "models/model.pth"
ENGINE_PATH = "./stockfish-macos-m1-apple-silicon"
IP_WEBCAM_URL = "http://192.168.2.21:8080/video"

IDX_TO_CLASS: dict[int, str] = {
    0: "b", 1: "B",
    2: ".",
    3: "k", 4: "K",
    5: "n", 6: "N",
    7: "p", 8: "P",
    9: "q", 10: "Q",
    11: "r", 12: "R",
}

REVERSED_ABBREV: dict[str, str] = {
    "b": "Bishop (B)",
    "B": "Bishop (W)",
    ".": "Empty",
    "k": "King (B)",
    "K": "King (W)",
    "n": "Knight (B)",
    "N": "Knight (W)",
    "p": "Pawn (B)",
    "P": "Pawn (W)",
    "q": "Queen (B)",
    "Q": "Queen (W)",
    "r": "Rook (B)",
    "R": "Rook (W)",
}



class PiecesClassifier:
    """Runs piece classification on board square patches.

    Wraps a torchvision model with the preprocessing pipeline to go from
    raw image patches to an (8, 8) FEN symbol array.

    Attributes:
        device (str): The torch device the model runs on.

    Args:
        model_path: Path to a serialised torchvision model checkpoint.
        device: Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
            Defaults to ``'cpu'``.
    """
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.to(self.device)
        self.__transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def predict(self, squares: list[np.ndarray]) -> np.ndarray:
        """Classify 64 square patches and return an (8, 8) FEN symbol array.

        Args:
            squares: A flat list of 64 BGR numpy arrays in row-major order,
                one per board square starting from the top-left.

        Returns:
            An (8, 8) numpy array of dtype ``<U2`` containing FEN piece
            symbols (uppercase = White, lowercase = Black, ``'.'`` = empty).
        """
        batch = torch.stack([
            self.__transform(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))) for p in squares
        ])

        with torch.no_grad():
            output = self.model(batch.to(self.device))  # (N, num_classes)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1).tolist()  # list of N predicted class indices
            confs = probs.max(dim=1).values.tolist()

        board = np.array([IDX_TO_CLASS[i] for i in preds], dtype="U2").reshape(8, 8)

        return board

class BoardDetector:
    """Detects and crops the chess board from raw camera frames.

        Args:
            vert_crop (float): Fraction of frame height to crop from top and bottom
                before board extraction. Defaults to 0.1.
            horiz_crop (float): Fraction of frame width to crop from left and right
                before board extraction. Defaults to 0.25.
    """
    def __init__(self, vert_crop: float = 0.1, horiz_crop: float = 0.25):
        self.__vert_crop: float = vert_crop
        self.__horiz_crop: float = horiz_crop
        self.__rotation: int | None = None


    def extract_board(self, img: np.ndarray) -> np.ndarray:
        """Crop out a chess board from a full camera frame.

        Applies the configured fractional edge crop, then locates the board
        grid via thresholding and trims to it.

        Args:
            img (np.ndarray): A BGR numpy array of the full camera frame.

        Returns:
            A BGR numpy array cropped to the edges of the chess grid.
        """
        h, w = img.shape[:2]
        vc = int(self.__vert_crop * h)
        hc = int(self.__horiz_crop * w)
        board = img[vc:-vc, hc:-hc]

        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
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

        result = board[y1:y2, x1:x2]

        rh, rw = result.shape[:2]
        vert_crop = int((1 / 15) * rh)
        horiz_crop = int((1 / 15) * rw)

        return result[vert_crop:-vert_crop, horiz_crop:-horiz_crop]

    def process(self, frame: np.ndarray, extractor: "PieceExtractor", classifier: "PiecesClassifier") -> np.ndarray:
        """Extract, auto-orient, and return the board from a raw camera frame.

        On the first call the board orientation is inferred from rook positions
        and cached for all subsequent frames.

        Args:
            frame: A BGR numpy array of the full camera frame.
            extractor: A :class:`PieceExtractor` instance.
            classifier: A :class:`PiecesClassifier` instance.

        Returns:
            A BGR numpy array of the board, correctly oriented (a1 bottom-left).
        """
        extracted = self.extract_board(frame)

        if self.__rotation is None:
            pieces = extractor.extract_pieces(extracted)
            pieces_array = classifier.predict(pieces)
            if pieces_array[0, 0] == "r" and pieces_array[0, 7] == "r":
                self.__rotation = 0
            elif pieces_array[7, 0] == "r" and pieces_array[7, 7] == "r":
                self.__rotation = 180
            elif pieces_array[0, 7] == "r" and pieces_array[7, 7] == "r":
                self.__rotation = -90
            else:
                self.__rotation = 90
            print(f"Board rotation set to {self.__rotation}")

        if self.__rotation == 90:
            extracted = cv2.rotate(extracted, cv2.ROTATE_90_CLOCKWISE)
        elif self.__rotation == 180:
            extracted = cv2.rotate(extracted, cv2.ROTATE_180)
        elif self.__rotation == -90:
            extracted = cv2.rotate(extracted, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return extracted

class PieceExtractor:
    """Extracts individual chess board squares from a cropped chess board image."""
    def extract_pieces(self, img: np.ndarray) -> list[np.ndarray]:
        """Extracts individual chess board squares from a cropped chess board image.

        Args:
            img (np.ndarray): A BGR numpy array of a cropped chess board.

        Returns:
            A flat list of 64 BGR numpy arrays, one per square, in row-major order.

        """
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

class Chess:
    """Manages game state, engine interaction, and display composition.

    Args:
        engine_path: Path to the Stockfish binary.
    """
    def __init__(self, engine_path: str = ENGINE_PATH):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.board = chess.Board()
        self._last_move: chess.Move | None = None
        self._last_eval: int | str = 0
        self._board_img: np.ndarray = self.board_to_img(self.board)

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.close()

    @staticmethod
    def board_to_numpy(board: chess.Board) -> np.ndarray:
        """Convert a :class:`chess.Board` to an (8, 8) FEN symbol array.

        Row 0 is rank 8 (Black's back rank) so the array matches the visual
        top-to-bottom orientation of a standard board image.

        Args:
            board: The board to convert.

        Returns:
            An (8, 8) numpy array of dtype ``<U2`` with FEN piece symbols.
        """
        arr = np.full((8, 8), ".", dtype="<U2")
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                arr[7 - (sq // 8)][sq % 8] = piece.symbol()
        return arr

    @staticmethod
    def board_to_img(board: chess.Board, size: int = 480, lastmove: chess.Move | None = None) -> np.ndarray:
        """Render a :class:`chess.Board` to a BGR numpy array via SVG.

        Args:
            board: The board to render.
            size: Output image side length in pixels.
            lastmove: Move to highlight on the board, or ``None``.

        Returns:
            A ``(size, size, 3)`` BGR numpy array.
        """
        svg = chess.svg.board(board, lastmove=lastmove)
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
        arr = np.frombuffer(png, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def eval_bar(eval_val: int | str, height: int = 480, width: int = 20) -> np.ndarray:
        """Render a vertical evaluation bar as a BGR numpy array.

        Args:
            eval_val: Material score (int) or ``'+M'``/``'-M'`` for checkmate.
            height: Bar height in pixels.
            width: Bar width in pixels.

        Returns:
            A ``(height, width, 3)`` BGR numpy array.
        """
        if eval_val == "+M":
            w_frac = 1.0
        elif eval_val == "-M":
            w_frac = 0.0
        else:
            w_frac = (max(-9, min(9, eval_val)) + 9) / 18
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        split = int((1 - w_frac) * height)
        bar[:split] = (50, 50, 50)
        bar[split:] = (220, 220, 220)
        return bar

    @staticmethod
    def material_eval(board: chess.Board) -> int:
        """Compute material balance from White's perspective.

        Args:
            board: The board to evaluate.

        Returns:
            An integer score (positive = White advantage).
        """
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        score = 0
        for piece_type, value in values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score

    @staticmethod
    def save_images(board_array: np.ndarray, squares: list[np.ndarray]) -> None:
        """Save square patches to the training dataset directory.

        Args:
            board_array: An (8, 8) FEN symbol array labelling each square.
            squares: A flat list of 64 BGR numpy arrays in row-major order.
        """
        for label, square in zip(board_array.flatten(), squares):
            class_name = REVERSED_ABBREV[label]
            out_dir = os.path.join("data", "NewDataset4", class_name)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(path, cv2.cvtColor(square, cv2.COLOR_RGB2BGR))

    def capture(self, board_array: np.ndarray, squares: list[np.ndarray]) -> bool:
        """Validate a captured board, advance game state, and play an engine move.

        Checks whether ``board_array`` matches a board reachable via one legal
        move.  If valid, pushes that move, has the engine reply, updates the
        display image and eval, and saves the square patches.

        Args:
            board_array: An (8, 8) FEN symbol array from the classifier.
            squares: The 64 square patches that produced ``board_array``.

        Returns:
            ``True`` if the position was legal and the engine replied,
            ``False`` if no legal move matched.
        """
        legal_flag = False
        for move in self.board.legal_moves:
            self.board.push(move)
            if np.array_equal(self.board_to_numpy(self.board), board_array):
                legal_flag = True
                break
            self.board.pop()

        if not legal_flag:
            print("Illegal board state detected.")
            return False

        engine_move = self.engine.play(self.board, chess.engine.Limit(time=0.001))
        self.board.push(engine_move.move)
        self._last_move = engine_move.move
        self._last_eval = self.material_eval(self.board)
        self.save_images(board_array, squares)
        print(f"Engine played: {engine_move.move}")
        self._board_img = self.board_to_img(self.board, lastmove=engine_move.move)
        return True

    def get_display(self, frame: np.ndarray) -> np.ndarray:
        """Compose the full UI frame: camera feed | board render | eval bar.

        Args:
            frame: The current processed board frame from the camera.

        Returns:
            A ``(520, 980, 3)`` BGR numpy array ready for :func:`cv2.imshow`.
        """
        display_frame = cv2.resize(frame, (480, 480))
        header = np.zeros((40, 980, 3), dtype=np.uint8)
        text = (
            f"Engine: {self._last_move} | Eval: {self._last_eval}"
            if self._last_move
            else "Waiting..."
        )
        cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        row = np.hstack([display_frame, self._board_img, self.eval_bar(self._last_eval)])
        return np.vstack([header, row])


def main():
    """Run the SIChess live capture loop.

    Opens the IP webcam stream, processes each frame, and handles two key
    bindings:

    * ``q`` — quit
    * ``c`` — capture the current board position, validate it, and have the
      engine play the next move
    """
    cap = cv2.VideoCapture(IP_WEBCAM_URL)

    # Change 'mps' to 'cpu' or 'cuda' for non-Apple Silicon devices.
    classifier = PiecesClassifier(device="mps")
    detector = BoardDetector()
    extractor = PieceExtractor()
    game = Chess()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        board_frame = detector.process(frame, extractor, classifier)
        display = game.get_display(board_frame)
        cv2.imshow("SIChess", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("c"):
            print("Captured")
            squares = extractor.extract_pieces(board_frame)
            board_array = classifier.predict(squares)
            for row in board_array:
                print(" ".join(row))
            game.capture(board_array, squares)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()