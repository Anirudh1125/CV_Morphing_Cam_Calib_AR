"""Capture the first non-interactive frame from the supplied ObjViz viewer."""
import sys
from pathlib import Path

import pygame
from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glReadPixels
from PIL import Image

import ObjViz

ROOT = Path(__file__).resolve().parent


def capture():
    w, h = pygame.display.get_surface().get_size()
    data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    frame = Image.frombytes("RGB", (w, h), data)
    frame.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(ROOT / "results/p4_objviz.png")
    pygame.quit()
    raise SystemExit(0)


if __name__ == "__main__":
    sys.argv[1:] = [str(ROOT / "inputs/cube1_color.obj")]
    pygame.display.flip = capture
    ObjViz.main()
