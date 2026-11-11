"""Wavefront vertex, face, and diffuse material loader for the supplied cube."""
from pathlib import Path

import numpy as np
from OpenGL.GL import (GL_COMPILE, GL_POLYGON, glBegin, glColor3f, glEnd,
                       glEndList, glGenLists, glNewList, glNormal3f, glVertex3f)


def load_obj(filename):
    filename = Path(filename)
    vertices, faces = [], []
    materials = {}
    color = (0.8, 0.8, 0.8)
    for line in filename.read_text().splitlines():
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        key, *args = fields
        if key == "mtllib":
            for line in filename.with_name(args[0]).read_text().splitlines():
                tokens = line.partition("#")[0].split()
                if tokens and tokens[0] == "newmtl":
                    name = tokens[1]
                elif tokens and tokens[0] == "Kd":
                    materials[name] = tuple(map(float, tokens[1:4]))
        elif key == "v":
            vertices.append(tuple(map(float, args[:3])))
        elif key == "usemtl":
            color = materials[args[0]]
        elif key == "f":
            indices = [int(token.split("/")[0]) for token in args]
            faces.append(([i - 1 if i > 0 else len(vertices) + i for i in indices], color))
    return np.asarray(vertices), faces


def draw_faces(vertices, faces):
    center = vertices.mean(axis=0)
    for ids, color in faces:
        face = vertices[ids]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        if np.dot(normal, face.mean(axis=0) - center) < 0:
            normal = -normal
        normal /= np.linalg.norm(normal)
        glColor3f(*color)
        glNormal3f(*normal)
        glBegin(GL_POLYGON)
        for vertex in face:
            glVertex3f(*vertex)
        glEnd()


class OBJ:
    """Display-list adapter for the supplied ObjViz.py."""
    def __init__(self, filename, swapyz=False):
        vertices, faces = load_obj(filename)
        if swapyz:
            vertices = vertices[:, [0, 2, 1]]
        self.center = vertices.mean(axis=0)
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        draw_faces(vertices, faces)
        glEndList()
