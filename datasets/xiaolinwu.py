"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""Script demonstrating drawing of anti-aliased lines using Xiaolin Wu's line
algorithm
 
usage: python xiaolinwu.py [output-file]
 
"""
from __future__ import division
import sys
 
from PIL import Image
 
 
def _fpart(x):
    return x - int(x)
 
def _rfpart(x):
    return 1 - _fpart(x)
 
def putpixel(img, xy, color, alpha=1):
    """Paints color over the background at the point xy in img.
 
    Use alpha for blending. alpha=1 means a completely opaque foreground.
 
    """
    c = tuple(map(lambda bg, fg: int(round(alpha * fg + (1-alpha) * bg)),
                  img.getpixel(xy), color))
    img.putpixel(xy, c)
 
def draw_line(img, p1, p2, color):
    """Draws an anti-aliased line in img from p1 to p2 with the given color."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px,py), (py,px))[steep]
 
    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1
 
    grad = dy/dx
    intery = y1 + _rfpart(x1) * grad
    def draw_endpoint(pt):
        x, y = pt
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        putpixel(img, p(px, py), color, _rfpart(yend) * xgap)
        putpixel(img, p(px, py+1), color, _fpart(yend) * xgap)
        return px
 
    xstart = draw_endpoint(p(*p1)) + 1
    xend = draw_endpoint(p(*p2))
 
    for x in range(xstart, xend):
        y = int(intery)
        putpixel(img, p(x, y), color, _rfpart(intery))
        putpixel(img, p(x, y+1), color, _fpart(intery))
        intery += grad