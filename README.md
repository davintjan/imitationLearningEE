# Imitation Learning for 7-DOF Robot Trajectories
We used feedforward MLP to learn the trajectories
## Models Overview

1. **Trajectory Imitation Without Noise and Offset**
   - Learns and replicates robot end-effector trajectories.
   - No external disturbances or offsets are introduced.

2. **Trajectory Imitation With Noise and Offset**
   - Learns and replicates robot end-effector trajectories.
   - Includes noise and offset to simulate real-world variations.

3. **Trajectory + Wrench Imitation Without Noise and Offset**
   - Learns both end-effector trajectories and associated wrench data.
   - No noise or offset is introduced.

4. **Trajectory + Wrench Imitation With Noise and Offset**
   - Learns and replicates robot end-effector trajectories and associated wrench data.
   - Includes noise and offset to simulate real-world conditions.

