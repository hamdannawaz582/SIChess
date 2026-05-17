# SIChess

Digital Image Processing semester project that turns a live camera feed of a physical chess board into a fully playable game against Stockfish.

## Pipeline

```
Camera Feed -> BoardDetector -> PieceExtractor -> PiecesClassifier -> Stockfish -> Display
```

Each frame from an IP webcam is processed end-to-end:

1. **BoardDetector**: Thresholding + morphological closing finds the board grid and fractional edge crops remove the surrounding border. On the first frame, rook positions are used to auto detect board orientation and the rotation is cached.
2. **PieceExtractor**: Divides the cropped board into 64 square images in row-major order.
3. **PiecesClassifier**: Runs a torchvision CNN on all 64 images in one batched forward pass and returns an 8x8 array of FEN symbols.
4. **Chess**: Validates the recognised position against legal moves in `python-chess`, pushes the player's move, queries Stockfish for a reply, and saves the square patches to the training dataset.
5. **Display**: Composites the live camera feed, a rendered board SVG, and a material-balance evaluation bar into a single OpenCV window.

## Quick Start

```bash
uv sync
uv run python sichess.py
```

Point the IP webcam stream at `IP_WEBCAM_URL` in `sichess.py`, then:

| Key | Action                                        |
|-----|-----------------------------------------------|
| `c` | Capture current frame and let Stockfish reply |
| `q` | Quit                                          |

## API Reference

See the sidebar for full class and method documentation:

- [`BoardDetector`](api.md#boarddetector): Board extraction and orientation
- [`PieceExtractor`](api.md#pieceextractor): Square patch slicing
- [`PiecesClassifier`](api.md#piecesclassifier): CNN inference
- [`Chess`](api.md#chess): Game state, engine, and display
