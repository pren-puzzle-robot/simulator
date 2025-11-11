# Simulator

> Achtung: Python 3.14 ist nicht Kompatibel mit OpenCV

**Step 0** - _Setup_
```powershell
pip install opencv-python numpy scipy matplotlib
```

**Step 1** - _Measure Contours_\
Pull out puzzle piece contours from an image and save individual masks.
```powershell
python .\pull_pieces.py --image ..\sample_images\simple_1_rotated.png --outdir ..\output
```

**Step 2** - _Detect Edges_\
Sample the detected edges and compile their surface as a plot
```powershell
python detect_edges.py --input "..\output\" --out ..\output\
```

**Step 3** - _Match Edges_\
Match the compiled data of the edges with each other
```powershell
python match_edges.py
```

**Step X** - _Reset_\
Delete previously computed values to start a new computation cycle
```powershell
Remove-Item "..\output\*" -Recurse
```

## Attempt #2

**Step 1** - _Measure Contours_\
Pull out puzzle piece contours from an image and save individual masks.
```powershell
python .\pull_pieces.py --image ..\sample_images\simple_1_rotated.png --outdir ..\output
```

**Step 2** - _Annotate corners_\
Annotate puzzle pieces
```powershell
python .\corners.py
```
Saves all corners in `corners.json` (Pixel Coordinates) and also annotated images in the output folder.

**Step 3** - _Compute Longest Edges_\
Compute the longest edges of each puzzle piece
```powershell
python .\long_edges.py
```

**Step 4** - _Match Edges_\
Arrange Pieces
```powershell
python .\arrange_pieces.py
```


