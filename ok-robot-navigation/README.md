# Navigation Module
This directory contains all the code related to navigation part of the project. It mainly contains following stages
* **Loading Voxel Map:** Initially, given a r3d file of 3D scan it computes the semantic representation of the scene and stores in `cfg.cache_path`directory.
* **Generating Pointcloud:** To properly visualize the path, pointcloud is generated from r3d file and stored in `cfg.pointcloud_path`
* **Object Localisation:** Using the semantic map created with voxel map we extract the most similar point in the scence with the clip embedding of the natural language query.
* **Path Computation:** Once we have the localized point we use A* algorithm with custom weights we generate a path to apporach the point.

When `path_planning.py` is run all the above stages happen sequentially.
