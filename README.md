# Light Print3D Layout Planner

[https://github.com/rkotynski/light-print3d-layout-planner](https://github.com/rkotynski/light-print3d-layout-planner)

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
- Polish and English user interface

---

## Instructions

- The “Load object (stl)” button loads an STL file containing a triangulated 3D object
- Objects can be duplicated and deleted using the “Duplicate” and “Remove” buttons
- “Auto place” finds a suitable position for a selected object, while “Arrange” attempts to arrange all objects
- Some operations may be slow; the title bar displays information when calculations are in progress
- Objects can be dragged with the mouse in the preview area and rotated during dragging using the mouse wheel
- Sliders allow scaling, rotation around all three axes, and translation in X and Y
- Main settings, such as the stage (window) size, are automatically saved and restored when the program starts
- The layout can be saved and loaded using the “Save layout” and “Load layout” buttons
- A simple 3D scene preview is available via the “3D preview” button; only a simplified lattice is shown in this preview
- A combined STL file can be generated using the “Export scene (stl)” button
- The exported scene may optionally include a lattice at the bottom:
    - the lattice can be flat
    - or it can extend upward to support the lowest parts of objects
- Press F11 to toggle full-screen mode

---

## Disclaimer
- The program is provided in the hope that it may be useful, but without any warranty
- AI techniques were used extensively during development
- Automatic object arrangement is currently experimental and may be slow or imperfect

---

## Requirements

- Python 3.9+
- PyQt5
- numpy
- matplotlib
- numpy-stl


## License
MIT License

## Screenshots
![Main window](https://github.com/rkotynski/light-print3d-layout-planner/blob/main/screenshot.png "Main window")
![3D Preview](https://github.com/rkotynski/light-print3d-layout-planner/blob/main/view%203d.png "3D Preview")

