# simulator

Achtung: Python 3.14 ist nicht Kompatibel mit OpenCV

Setup:
```python
pip install opencv-python numpy scipy matplotlib
```

Pull out puzzle piece contours from an image and save individual masks.
```python
python .\pull_pieces.py --image ..\sample_images\simple_1_rotated.png --outdir ..\output
```
Annotate puzzle pieces
```python
python .\annotate_piece_masks.py --indir ..\output\ --outdir ..\output\annotated
```

Sample detect edges
```python
python detect_edges.py --glob "..\output\piece_*.png" --out ..\output\
```