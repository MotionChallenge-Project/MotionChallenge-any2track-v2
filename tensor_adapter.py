"""Any2Track's residual joint targets and adapter-owned reference foot heights."""
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import warp as wp
from motion_challenge.validation import sha256_file
from tensor_common import TensorPD,torch,rotate


class TensorAny2Track(TensorPD):
    action_output='continuous_actions'
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        super().__init__(root,runtime,joint_names,slots,device,converted_weights)
        path=Path(root)/runtime['reference_kinematics']
        if sha256_file(path)!=runtime['reference_kinematics_sha256']:raise ValueError('Reference geometry changed')
        self.cpu=mujoco.MjModel.from_xml_path(str(path));self.qids=[int(self.cpu.joint(n).qposadr[0]) for n in joint_names]
        self.feet=[self.cpu.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
        with wp.ScopedDevice(str(device)),self.stream():
            self.ref_model=mjw.put_model(self.cpu);self.ref_data=mjw.make_data(self.cpu,nworld=slots,nconmax=1,njmax=1)
            self.ref_qpos=wp.to_torch(self.ref_data.qpos);self.ref_qpos[:]=torch.as_tensor(self.cpu.qpos0,device=device,dtype=torch.float32)
        self.targets=torch.zeros((slots,29),device=device)

    def stream(self):return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def observations(self,state,reference):
        self.ref_qpos[:,:3]=reference.root_position_world_m[:,0]
        self.ref_qpos[:,3:7]=reference.root_orientation_xyzw[:,0,[3,0,1,2]]
        self.ref_qpos[:,self.qids]=reference.joint_position_rad[:,0].float()
        with wp.ScopedDevice(str(self.device)),self.stream():mjw.kinematics(self.ref_model,self.ref_data)
        positions=wp.to_torch(self.ref_data.xpos)[:,self.feet].double();rotations=wp.to_torch(self.ref_data.xquat)[:,self.feet][:,:,[1,2,3,0]].double()
        offset=torch.tensor([.13,0.,-.029],device=self.device,dtype=torch.float64)
        heights=torch.cat((positions[:,:,2],(positions+rotate(rotations,offset))[:,:,2]),dim=-1)
        q,qd,omega,gravity,_=self.proprio(state)
        self.targets.copy_(torch.where(self.fresh[:,None],q.float(),self.targets));self.fresh.zero_()
        tq=reference.joint_position_rad[:,1,self.order].double();tv=reference.joint_velocity_rad_s[:,1,self.order].double()
        obs=torch.cat((tq-q,(tv-qd)*.05,gravity,omega*.05,q-self.default,qd*.05,self.targets,heights,reference.root_position_world_m[:,1,2:3]),dim=-1)
        return {'obs':obs.float()}

    @torch.inference_mode()
    def act(self,state,reference):
        raw=self.model(**self.observations(state,reference))[self.action_output]
        self.targets.copy_((reference.joint_position_rad[:,1,self.order].double()+raw*self.scale).float())
        target=self.targets[:,self.inverse];zero=torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1).float()


ADAPTER_CLASS='TensorAny2Track'
NATIVE_CLASS='Any2Track'
