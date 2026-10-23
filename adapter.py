"""Any2Track generalist v2, following OpenTrack's released MuJoCo player.

The native player advances the target before inference; feet heights refer to the
previous target. The last-action term stores final joint position targets.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController
from motion_challenge.sdk import ControlCommand


class Any2Track(ONNXController):
    action_output='continuous_actions'

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        self.targets=np.asarray(state.joint_position_rad)[self.order].astype(np.float32)

    def observations(self,state,reference):
        current,target=reference
        q,qd,w,g,_=self.proprio(state)
        bodies={b.name:b for b in current.bodies}
        heights=[]
        for name,offset in [('left_ankle_roll_link',(0,0,0)),('right_ankle_roll_link',(0,0,0)),
                            ('left_ankle_roll_link',(.13,0,-.029)),('right_ankle_roll_link',(.13,0,-.029))]:
            b=bodies[name]
            heights.append((np.asarray(b.position_world_m)+R.from_quat(b.orientation_xyzw).apply(offset))[2])
        tq=np.asarray(target.joint_position_rad)[self.order]
        tv=np.asarray(target.joint_velocity_rad_s)[self.order]
        obs=np.r_[tq-q,(tv-qd)*.05,g,w*.05,q-self.default,qd*.05,self.targets,heights,target.root_position_world_m[2]]
        return {'obs':obs[None].astype(np.float32)}

    def act(self,state,reference):
        raw=self.infer(self.observations(state,reference))[self.action_output].reshape(29)
        if not np.isfinite(raw).all():raise ValueError('nonfinite action')
        self.targets=(np.asarray(reference[1].joint_position_rad)[self.order]+raw*self.scale).astype(np.float32)
        def order(a):return tuple(float(v) for v in a[self.inverse])
        return ControlCommand(order(self.targets),(0.,)*29,order(self.kp),order(self.kd),(0.,)*29)


def create(root,runtime):return Any2Track(root,runtime)
