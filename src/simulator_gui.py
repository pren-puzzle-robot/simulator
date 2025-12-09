import os
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from pull_pieces import pull_pieces
from corners import detect_corners
from match import solve
from component import PuzzlePiece, Point
from utilities.draw_puzzle_piece import print_whole_puzzle_image, render_puzzle_piece


SCALE: float = 0.5
MARGIN: int = 50


def ensure_out_dir(outdir: str, log=print) -> None:
    os.makedirs(outdir, exist_ok=True)
    for filename in os.listdir(outdir):
        file_path = os.path.join(outdir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            log(f"Failed to delete {file_path}. Reason: {e}")

def fit_pil_to_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    if max_w <= 0 or max_h <= 0:
        return img
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    if new_size == img.size:
        return img
    return img.resize(new_size, Image.Resampling.LANCZOS)


@dataclass
class PieceView:
    idx: int
    filename: str
    piece_path: str
    piece: PuzzlePiece
    annotated_pil: Optional[Image.Image] = None


def run_pipeline(image_path: str, outdir: str, log=print) -> Tuple[Image.Image, List[PieceView]]:
    log(f"Input image: {image_path}")
    log(f"Output dir : {outdir}")
    ensure_out_dir(outdir, log=log)

    img = cv.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    log("Step 1/4: pull_pieces(...)")
    piece_paths: List[str] = pull_pieces(img, outdir)
    log(f"  -> extracted {len(piece_paths)} piece images")

    log("Step 2/4: detect_corners(...)")
    corners = detect_corners(piece_paths, outdir)
    log(f"  -> detected corners for {len(corners)} pieces")

    log("Step 3/4: create PuzzlePiece objects")
    puzzle_pieces: Dict[int, PuzzlePiece] = {}
    piece_views: List[PieceView] = []

    for i, (filename, corner_list) in enumerate(corners):
        points = [Point(x=float(x), y=float(y)) for x, y in corner_list]
        piece = PuzzlePiece(points)
        puzzle_pieces[i] = piece

        candidate_path = filename
        if not os.path.isfile(candidate_path):
            candidate_path = os.path.join(outdir, os.path.basename(filename))
        if not os.path.isfile(candidate_path) and i < len(piece_paths) and os.path.isfile(piece_paths[i]):
            candidate_path = piece_paths[i]

        piece_views.append(
            PieceView(
                idx=i,
                filename=os.path.basename(filename),
                piece_path=candidate_path,
                piece=piece,
                annotated_pil=None,  # filled after solve()
            )
        )

        log(f"  -> piece {i}: {os.path.basename(filename)}")

    log("Step 4/4: solve(...)")
    solve(puzzle_pieces)

    log("Rendering annotated pieces (PIL, post-solve)...")
    for pv in piece_views:
        pv.annotated_pil = render_puzzle_piece(pv.piece, scale=SCALE, margin=MARGIN)

    log("Creating final image (PIL)...")
    final_img: Image.Image = print_whole_puzzle_image(puzzle_pieces)
    log("Done.")
    return final_img, piece_views




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Puzzle Simulator GUI")
        self.geometry("1100x650")
        self.minsize(950, 560)

        self.image_var = tk.StringVar(value="")
        self.outdir_var = tk.StringVar(value=os.path.abspath(os.path.join("..", "output")))

        self.running = False
        self.piece_views: List[PieceView] = []
        self.final_pil: Optional[Image.Image] = None

        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._final_photo: Optional[ImageTk.PhotoImage] = None

        self.view_mode = tk.StringVar(value="piece")  # "piece" or "annotated"
        self._build_ui()

    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Input image:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.image_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="Browse...", command=self.pick_image).grid(row=0, column=2)

        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.outdir_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(top, text="Browse...", command=self.pick_outdir).grid(row=1, column=2, pady=(8, 0))

        controls = ttk.Frame(top)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        self.run_btn = ttk.Button(controls, text="Run simulation", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        ttk.Button(controls, text="Clear log", command=self.clear_log).pack(side=tk.LEFT, padx=(10, 0))

        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        top.columnconfigure(1, weight=1)

        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=pad, pady=(0, pad))

        self.tab_pieces = ttk.Frame(nb, padding=pad)
        self.tab_final = ttk.Frame(nb, padding=pad)
        self.tab_log = ttk.Frame(nb, padding=pad)

        nb.add(self.tab_pieces, text="Pieces")
        nb.add(self.tab_final, text="Final image")
        nb.add(self.tab_log, text="Log")

        self._build_pieces_tab()
        self._build_final_tab()
        self._build_log_tab()

    def _build_pieces_tab(self):
        left = ttk.Frame(self.tab_pieces)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Pieces:").pack(anchor="w")
        self.listbox = tk.Listbox(left, height=25, width=35)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        sb = ttk.Scrollbar(left, command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.update_piece_preview())

        right = ttk.Frame(self.tab_pieces)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        mode_row = ttk.Frame(right)
        mode_row.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mode_row, text="Preview mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Extracted piece image", variable=self.view_mode, value="piece",
                        command=self.update_piece_preview).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(mode_row, text="Annotated render", variable=self.view_mode, value="annotated",
                        command=self.update_piece_preview).pack(side=tk.LEFT, padx=(10, 0))

        self.preview_label = ttk.Label(right, text="Run the simulation to see previews.", anchor="center")
        self.preview_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 0))

        # refresh previews on resize
        right.bind("<Configure>", lambda _e: self.update_piece_preview())

    def _build_final_tab(self):
        self.final_label = ttk.Label(self.tab_final, text="Run the simulation to see the final image.", anchor="center")
        self.final_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tab_final.bind("<Configure>", lambda _e: self.update_final_preview())

    def _build_log_tab(self):
        ttk.Label(self.tab_log, text="Log:").pack(anchor="w")
        self.log_text = tk.Text(self.tab_log, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(self.tab_log, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set)

        try:
            self.log_text.configure(font=("Consolas", 10))
        except Exception:
            pass

    def pick_image(self):
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.outdir_var.set(path)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def log(self, msg: str):
        def _append():
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.after(0, _append)

    def set_running(self, is_running: bool):
        def _set():
            self.running = is_running
            self.run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
            if is_running:
                self.progress.start(10)
            else:
                self.progress.stop()
        self.after(0, _set)

    def on_run(self):
        if self.running:
            return

        image_path = self.image_var.get().strip()
        outdir = self.outdir_var.get().strip()

        if not image_path:
            messagebox.showerror("Missing input", "Please choose an input image.")
            return
        if not os.path.isfile(image_path):
            messagebox.showerror("Invalid input", f"File does not exist:\n{image_path}")
            return
        if not outdir:
            messagebox.showerror("Missing output", "Please choose an output folder.")
            return

        self.piece_views = []
        self.final_pil = None
        self.listbox.delete(0, tk.END)
        self._preview_photo = None
        self._final_photo = None
        self.preview_label.configure(image="", text="Running...")
        self.final_label.configure(image="", text="Running...")

        self.set_running(True)
        self.log(f"=== Starting simulation (scale={SCALE}, margin={MARGIN}) ===")

        def worker():
            try:
                final_img, piece_views = run_pipeline(image_path=image_path, outdir=outdir, log=self.log)

                def apply_results():
                    self.piece_views = piece_views
                    self.final_pil = final_img

                    self.listbox.delete(0, tk.END)
                    for pv in self.piece_views:
                        self.listbox.insert(tk.END, f"{pv.idx:03d}  {pv.filename}")

                    if self.piece_views:
                        self.listbox.selection_set(0)
                        self.listbox.activate(0)
                        self.update_piece_preview()

                    self.update_final_preview()

                self.after(0, apply_results)
                self.log("=== Finished successfully ===")
                self.after(0, lambda: messagebox.showinfo("Done", "Simulation finished."))

            except Exception as e:
                self.log("=== ERROR ===")
                self.log(str(e))
                self.log(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror("Error", f"{e}"))
            finally:
                self.set_running(False)

        threading.Thread(target=worker, daemon=True).start()

    def _get_selected_piece(self) -> Optional[PieceView]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        i = sel[0]
        if 0 <= i < len(self.piece_views):
            return self.piece_views[i]
        return None

    def update_piece_preview(self):
        pv = self._get_selected_piece()
        if pv is None:
            if self.piece_views:
                self.preview_label.configure(text="Select a piece.", image="")
            return

        mode = self.view_mode.get()

        try:
            if mode == "piece":
                if not pv.piece_path or not os.path.isfile(pv.piece_path):
                    self.preview_label.configure(
                        text=f"Could not find piece image on disk.\nExpected: {pv.piece_path}",
                        image=""
                    )
                    self._preview_photo = None
                    return
                pil = Image.open(pv.piece_path).convert("RGB")
            else:
                if pv.annotated_pil is None:
                    self.preview_label.configure(text="No annotated render available.", image="")
                    self._preview_photo = None
                    return
                pil = pv.annotated_pil

            self.preview_label.update_idletasks()
            w = max(1, self.preview_label.winfo_width() - 12)
            h = max(1, self.preview_label.winfo_height() - 12)
            pil_fit = fit_pil_to_box(pil, w, h)

            self._preview_photo = ImageTk.PhotoImage(pil_fit)
            self.preview_label.configure(image=self._preview_photo, text="")

        except Exception as e:
            self.preview_label.configure(text=f"Preview error:\n{e}", image="")
            self._preview_photo = None

    def update_final_preview(self):
        if self.final_pil is None:
            self.final_label.configure(text="Run the simulation to see the final image.", image="")
            self._final_photo = None
            return

        try:
            self.final_label.update_idletasks()
            w = max(1, self.final_label.winfo_width() - 12)
            h = max(1, self.final_label.winfo_height() - 12)
            pil_fit = fit_pil_to_box(self.final_pil, w, h)

            self._final_photo = ImageTk.PhotoImage(pil_fit)
            self.final_label.configure(image=self._final_photo, text="")

        except Exception as e:
            self.final_label.configure(text=f"Final image preview error:\n{e}", image="")
            self._final_photo = None


if __name__ == "__main__":
    App().mainloop()
