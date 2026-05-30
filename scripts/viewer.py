import mujoco
import mujoco.viewer
import time
from pathlib import Path

# Resolve XML path relative to this script's location
ASSET_DIR = Path(__file__).resolve().parent.parent / "src" / "smolvla_recap" / "env" / "assets"
xml_path = str(ASSET_DIR / "so101_tabletop.xml")

# Load the model and create the data structure
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Retrieve and apply the "ready" keyframe to initialize joint positions
# Forward kinematics to update site and sensor states before stepping
ready_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
if ready_key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, ready_key_id)
    mujoco.mj_forward(model, data)

# Launch the passive viewer and run the simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)