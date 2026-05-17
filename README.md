# SIChess

Digital Image Processing semester project that turns a live camera feed of a physical chess board into a fully playable game against Stockfish (or any engine of your choice).

https://hamdannawaz582.github.io/SIChess/

## How It Works

The project is split into 5 stages:
1. Board detection and cropping (implemented in the BoardDetector class)
2. Piece extraction (implemented in the PieceExtractor class)
3. Piece classification (implemented in the PiecesClassifier class)
4. Chess logic (implemented in the Chess class)
5. Display

### Pipeline Stages

**BoardDetector**: Applies a configurable edge crop, then uses thresholding and morphological closing to locate the board grid. On the first frame it infers orientation by checking rook positions and caches the rotation for all subsequent frames.

**PieceExtractor**: Divides the cropped board into a flat list of 64 BGR square patches in row-major order.

**PiecesClassifier**: Runs a serialised torchvision model on all 64 patches in a single batched forward pass. Outputs an 8x8 array of FEN symbols (`R`, `n`, `.`, …).

**Chess**: Wraps `python-chess` and Stockfish. On each capture it checks whether the classifier's board matches any position reachable via one legal move. If valid it pushes the player's move, asks Stockfish to reply, and re-renders the board SVG. The display composites the live camera frame, the rendered board, and a material-balance evaluation bar side by side.

## Setup

**Requirements:** Python >= 3.12, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

PyTorch is pulled from the CPU index on macOS/Windows and from the CUDA 12.6 index on Linux automatically.

You also need a Stockfish binary. The program defaults to `./stockfish-macos-m1-apple-silicon` but you can change it in `sichess.py` via `ENGINE_PATH` or by passing the path as an argument to the `Chess` class.

### IP Webcam

The default stream URL is `http://192.168.2.21:8080/video` (set via `IP_WEBCAM_URL` in `sichess.py`). Use the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) Android app or any MJPEG stream.

## Running

```bash
uv run python sichess.py
```

| Key | Action                                                                                  |
|-----|-----------------------------------------------------------------------------------------|
| `c` | Capture the current frame, classify pieces, validate the move, and have Stockfish reply |
| `q` | Quit                                                                                    |

## Device Selection

Edit the `device` argument in `main()`:

```python
classifier = PiecesClassifier(device="mps")   # Apple Silicon (default)
classifier = PiecesClassifier(device="cuda")  # NVIDIA GPU
classifier = PiecesClassifier(device="cpu")   # CPU fallback
```

## Project Structure

```
sichess.py               # Full pipeline: (detection, classification, game logic)
models/
    model.pth            # Trained piece classifier checkpoint (Custom classification head on MobileNetV2)
preprocessing/
    kmeans.ipynb         # Colour-based board segmentation experiments
    model.ipynb          # Model training notebook
    extraction.ipynb         # Board Extraction experiments
    slicing.ipynb        # Board slicing experiments
docs/                    # MkDocs documentation source
```

## Dependencies

| Package                            | Purpose                              |
|------------------------------------|--------------------------------------|
| `torch` / `torchvision`            | Piece classification model           |
| `opencv-python`                    | Frame capture and image processing   |
| `python-chess`                     | Board state and move validation      |
| `cairosvg`                         | Rendering board SVGs to numpy arrays |
| `scikit-learn` / `scikit-image`    | Preprocessing utilities              |
| `mkdocs-material` + `mkdocstrings` | Documentation                        |

## Documentation

```bash
uv run mkdocs serve
```
