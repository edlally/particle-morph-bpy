# particle_morph_bpy [WIP]
###### Generate 3d linear interpolation animations between any two meshes. Morph one object into another (looks best with particles). Objects can be moved in viewport and interpolation updates in real time.

### Features

- **Particle Mode**: *Boolean* - Cover mesh faces in pseudo-randomly placed particles. They are spread relatively evenly to avoid collisions.
- **Collision Avoidance**: Make particles avoid each other. Samples all points on paths and checks if they will collide. If too close, paths are offset with separation vector.
- **Trajectory Visualisation**: Paths are displayed using curves.
- **Simple Timeline Control**: Set start and end frames easily.
- **Live viewport interpolation updates**: Objects can be moved and interpolation is calculated on-the-fly. Useful for precise placements.

### Parameters:

- **Distortion (Turbulence)**: *FloatPropery* - Make particles deviate linear path. Effect is most pronounced (furthest deviation) when particle is equal distance from both objects.
- **Particle Count**: *IntProperty*: Determines number of particles. 
- **Allow Crossing Paths (collision)**: *Boolean* - If unchecked collisions are prevented.
- **Collision Avoidance**: *FloatProperty* - Determines  amount of avoidance (higher particle counts require higher avoidance setting).
- **Show Trajectories**: *Boolean* - Shows all particle paths with curves.
- **Trajectory Samples**: *IntProperty* - Determines sample count of curve (thus accuracy).

![image](https://github.com/user-attachments/assets/322c7ff6-9bc1-472d-bf6c-15ade7458064)
