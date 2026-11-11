"""Render the OBJ on Photo B with the calibrated OpenGL camera."""
import argparse
import json
from pathlib import Path

import numpy as np
import pygame
from OpenGL import GL as gl
from PIL import Image

from ObjLoader import draw_faces, load_obj

ROOT = Path(__file__).resolve().parent


def projection(K, w, h, near=0.1, far=100):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0],
                     [0, 2 * fy / h, 2 * cy / h - 1, 0],
                     [0, 0, -(far + near) / (far - near),
                      -2 * far * near / (far - near)],
                     [0, 0, -1, 0]])


def modelview(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return np.diag([1, -1, -1, 1]) @ T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    stats = json.loads((ROOT / "results/p4.json").read_text())
    K, R, t = (np.asarray(stats[key]) for key in ["K", "R", "t"])
    background = Image.open(ROOT / "inputs/IMG_4398.jpeg").convert("RGB")
    w, h = background.size
    pygame.display.init()
    if not args.snapshot:
        desktop = pygame.display.get_desktop_sizes()[0]
        scale = min(1, (desktop[0] - 60) / w, (desktop[1] - 100) / h)
        ww, hh = round(w * scale), round(h * scale)
    else:
        ww, hh = w, h
    background = background.resize((ww, hh), Image.Resampling.LANCZOS)
    pixels = background.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()
    vertices, faces = load_obj(ROOT / "inputs/cube1_color.obj")
    # Inches in cover coordinates. The base lies on Z = 0, top lies above it.
    world = (vertices - [1.5, 1, 0]) * [0.80, 0.80, -0.95]
    gl_projection, view = projection(K, w, h), modelview(R, t)
    # Independently compare OpenGL window coordinates with the CV pinhole result.
    homogeneous = np.c_[world, np.ones(len(world))]
    clip = homogeneous @ (gl_projection @ view).T
    ndc = clip[:, :2] / clip[:, 3, None]
    screen = (ndc * [1, -1] + 1) * [w / 2, h / 2]
    image = homogeneous @ np.asarray(stats["P2"]).T
    discrepancy = np.max(np.abs(screen - image[:, :2] / image[:, 2, None]))
    assert discrepancy < 1e-9
    print(f"Max CV/OpenGL vertex discrepancy: {discrepancy:.3e} px")

    pygame.display.set_mode((ww, hh), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("HW3 - notebook augmented reality")
    gl.glViewport(0, 0, ww, hh)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glWindowPos2i(0, 0)
    gl.glDrawPixels(ww, hh, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadMatrixd(gl_projection.T.copy())
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadMatrixd(view.T.copy())
    draw_faces(world, faces)

    if args.snapshot:
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels(0, 0, ww, hh, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        snapshot = Image.frombytes("RGB", (ww, hh), data)
        snapshot.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ROOT / "results/p4_ar.png")
    pygame.display.flip()
    if not args.snapshot:
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                               event.key == pygame.K_ESCAPE):
                break
    pygame.quit()


if __name__ == "__main__":
    main()
