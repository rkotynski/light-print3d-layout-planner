# Print3D Layout Planner

**A desktop tool for arranging STL models, checking collisions and exporting printable scenes**

Print3D Layout Planner is a Python / PyQt-based application for planning
the layout of multiple STL objects on a 3D printer bed.
It provides fast 2D and 3D previews, collision detection, automatic placement
and export-ready STL generation with optional lattice support.

---

## Features

- Load multiple STL files
- Scale, rotate and move objects
- Automatic object placement
- Collision detection before export
- Fast 2D preview (mask-based)
- Accurate 2D preview matching export
- 3D preview with simplified lattice
- Export combined scene to STL
- Optional lattice generation:
  - constant or variable height
  - adjustable strength
  - optimized for easy print removal
- Supports high-DPI / 4K displays
- Polish and English user interface

---

## Requirements

- Python 3.9+
- PyQt5
- numpy
- matplotlib
- numpy-stl

Install dependencies:

```bash
pip install -r requirements.txt

## License
MIT License
