# Basic Wavefront OBJ file viewer.
# ObjLoader: http://www.pygame.org/wiki/OBJFileLoader
# Left Mouse Button + move: Rotate
# Right Mouse Button + move: Pan
# Scroll Wheel: Zoom In/Out
# Adapted by Ming-Ching Chang, Feb 21, 2013.
# See http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html for more example OBJ data files.

import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from ObjLoader import *

def main():
  print ('ObjViz: Wavefront OBJ File Viewer')
  print ('Usage:')
  print ('  Left Mouse Button + move: Rotate')
  print ('  Right Mouse Button + move: Pan')
  print ('  Scroll Wheel: Zoom In/Out')

  #INPUT_OBJ_FILE = 'triangle1.obj'
  #INPUT_OBJ_FILE = 'triangle2_color.obj'
  #INPUT_OBJ_FILE = 'triangle3_color.obj'
  INPUT_OBJ_FILE = 'inputs/cube1_color.obj'
  #INPUT_OBJ_FILE = 'cube1_color.obj'
  #INPUT_OBJ_FILE = 'airboat.obj'
  #INPUT_OBJ_FILE = 'toyplane.obj'
  #INPUT_OBJ_FILE = 'foo'
  if len(sys.argv)>1:
    INPUT_OBJ_FILE = sys.argv[1]

  pygame.init()
  #viewport = (800,600)
  viewport = (400,300)
  hx = viewport[0]/2
  hy = viewport[1]/2
  srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

  glEnable(GL_LIGHTING)
  glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
  glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
  glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
  glEnable(GL_LIGHT0)
  glEnable(GL_COLOR_MATERIAL)
  glEnable(GL_DEPTH_TEST)

  # For shading see http://en.wikipedia.org/wiki/Shading
  # Set OpenGL to use Smooth Shading
  glShadeModel(GL_SMOOTH)
  # Set OpenGL to use Flat Shading
  #glShadeModel(GL_FLAT)

  # LOAD OBJECT AFTER PYGAME INIT
  obj = OBJ(INPUT_OBJ_FILE, swapyz=True)
  clock = pygame.time.Clock()

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  width, height = viewport
  gluPerspective(90.0, width/float(height), 1, 100.0)
  glEnable(GL_DEPTH_TEST)
  glMatrixMode(GL_MODELVIEW)

  rx, ry = (28, 23)
  tx, ty = (0, 0)
  zpos = 3.5
  rotate = move = False
  while 1:
    for e in pygame.event.get():
      if e.type == QUIT:
        sys.exit()
      elif e.type == KEYDOWN and e.key == K_ESCAPE:
        sys.exit()
      elif e.type == MOUSEBUTTONDOWN:
        if e.button == 4: zpos = max(1, zpos-1)
        elif e.button == 5: zpos += 1
        elif e.button == 1: rotate = True
        elif e.button == 3: move = True
      elif e.type == MOUSEBUTTONUP:
        if e.button == 1: rotate = False
        elif e.button == 3: move = False
      elif e.type == MOUSEMOTION:
        i, j = e.rel
        if rotate:
          rx += i
          ry += j
        if move:
          tx += i
          ty -= j

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # RENDER OBJECT
    glTranslate(tx/20., ty/20., - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    if hasattr(obj,'gl_list'):
      glTranslate(*(-obj.center))
      glCallList(obj.gl_list)
    else:
      print ('Nothing to display. Program exits!')
      sys.exit()

    pygame.display.flip()
    clock.tick(50)


if __name__ == '__main__':
  main()
