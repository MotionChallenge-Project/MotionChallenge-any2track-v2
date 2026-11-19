"""Independent original-adapter observations and ONNX output parity."""
import argparse,json,sys,types
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from motion_challenge.validation import sha256_file
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket,MotionReferenceBatch,ReferenceFrame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--converted',type=Path,required=True);parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=json.loads((root/'runtime.mjwarp.json').read_text());cfg=json.loads((root/'policy.json').read_text())
    from tensor_adapter import ADAPTER_CLASS,NATIVE_CLASS
    import tensor_adapter
    tensor_type=getattr(tensor_adapter,ADAPTER_CLASS)
    if 'onnxruntime' not in sys.modules:sys.modules['onnxruntime']=types.ModuleType('onnxruntime')
    import adapter as original_module
    from motion_challenge.adapters.onnx import vector
    original_type=getattr(original_module,NATIVE_CLASS)
    # Use reversed named order to exercise both mappings.
    joint_names=tuple(reversed(cfg['policy_joint_names']));slots=7;device=args.device
    batched=tensor_type(root,runtime,joint_names,slots,device,args.converted)
    batched.reset(torch.arange(slots,device=device));original=[]
    for _ in range(slots):
        a=original_type.__new__(original_type);a.cfg=cfg;a.runtime=runtime;a.names=cfg['policy_joint_names'];a.default=vector(cfg['default_joint_pos'],a.names)
        a.scale=vector(cfg['action_scale'],a.names,1.);original.append(a)
    from motion_challenge.simulation.physics import World
    from motion_challenge.resources import resource_root
    from motion_challenge.validation import sha256_file
    from scipy.spatial.transform import Rotation
    import mujoco,ast
    world=World(resource_root()/'robots/g1-29dof-mjwarp-v1.json');body_ids=[world.model.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
    order=[joint_names.index(n) for n in world.joint_names]
    def feet(position,quaternion,joints):
        world.data.qpos[:]=world.model.qpos0;world.data.qpos[:3]=position;world.data.qpos[3:7]=quaternion[[3,0,1,2]];world.data.qpos[world.qadr]=joints[order]
        mujoco.mj_kinematics(world.model,world.data)
        return tuple(types.SimpleNamespace(name=world.model.body(b).name,position_world_m=world.data.xpos[b].copy(),orientation_xyzw=world.data.xquat[b,[1,2,3,0]].copy()) for b in body_ids)
    source=root/'validation/upstream/play_g1_env_tracking_general.py';source_meta=json.loads((source.parent/'source.json').read_text())
    if sha256_file(source)!=source_meta['sha256']:raise ValueError('Author source checksum mismatch')
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PlayG1TrackingGeneralEnv');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_obs');quat_to_mat=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quat_to_mat')
    namespace={'np':np};exec(compile(ast.Module(body=[method,quat_to_mat],type_ignores=[]),str(source),'exec'),namespace)
    author_cfg=json.loads((source.parent/'author_config.json').read_text())['env_config']
    author_error=0.
    oracle=ReferenceEvaluator(str(args.weights));rng=np.random.default_rng(7913);max_observation=0.;max_action=0.;max_target=0.
    for step in range(24):
        q=rng.normal(0,.2,(slots,29));qd=rng.normal(0,.2,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);omega=rng.normal(0,.3,(slots,3))
        offsets=np.asarray(runtime['reference_offsets_s']);k=len(offsets);requested=step*.02+np.broadcast_to(offsets,(slots,k));times=requested.clip(0,1.)
        rj=rng.normal(0,.2,(slots,k,29));rp=rng.normal(0,.2,(slots,k,3));rp[...,2]+=.75;rq=rng.normal(size=(slots,k,4));rq/=np.linalg.norm(rq,axis=-1,keepdims=True)
        jv=rng.normal(0,.2,(slots,k,29));lv=rng.normal(0,.2,(slots,k,3));av=rng.normal(0,.2,(slots,k,3))
        def tensor(x):return torch.tensor(np.array(x),device=device)
        state=DeployableStateBatch(joint_names,tensor(np.full(slots,step*.02)),*[tensor(x) for x in (q,qd,quat,omega)])
        reference=MotionReferenceBatch(*[tensor(x) for x in (requested,times,rj,rp,rq,jv,lv,av)])
        if step==11:batched.reset(torch.tensor([1,4],device=device))
        got=batched.observations(state,reference);expected_actions=[]
        for row,a in enumerate(original):
            packet=DeployableStatePacket(step*.02,joint_names,tuple(q[row]),tuple(qd[row]),tuple(quat[row]),tuple(omega[row]))
            frames=tuple(ReferenceFrame(float(requested[row,i]),float(times[row,i]),tuple(rj[row,i]),tuple(rp[row,i]),tuple(rq[row,i]),tuple(jv[row,i]),tuple(lv[row,i]),tuple(av[row,i]),bodies=feet(rp[row,i],rq[row,i],rj[row,i])) for i in range(k))
            if step==0 or (step==11 and row in (1,4)):a.reset(packet,frames,0)
            expected={key:np.asarray(value,np.float32) for key,value in a.observations(packet,frames).items()}
            rotation=Rotation.from_quat(quat[row]);current,target=frames
            foot_values=current.bodies;heights=[b.position_world_m[2] for b in foot_values]+[(b.position_world_m+Rotation.from_quat(b.orientation_xyzw).apply([.13,0,-.029]))[2] for b in foot_values]
            fake=types.SimpleNamespace(get_gyro=lambda frame:rotation.inv().apply(omega[row]),mj_data=types.SimpleNamespace(qpos=np.r_[np.zeros(3),quat[row,[3,0,1,2]],q[row,a.order]],qvel=np.r_[np.zeros(6),qd[row,a.order]],site_xmat=rotation.as_matrix()[None]),ref_mj_data=types.SimpleNamespace(site_xpos=np.c_[np.zeros((4,2)),heights]),_pelvis_imu_site_id=0,_feet_all_site_id=np.arange(4),_default_qpos=a.default,obs_joint_ids=np.arange(29),_config=types.SimpleNamespace(obs_scales_config=types.SimpleNamespace(**author_cfg['obs_scales_config']),obs_keys=author_cfg['obs_keys'],history_keys=author_cfg['history_keys']))
            trajectory=types.SimpleNamespace(qpos=np.r_[target.root_position_world_m,np.array(target.root_orientation_xyzw)[[3,0,1,2]],np.array(target.joint_position_rad)[a.order]],qvel=np.r_[np.zeros(6),np.array(target.joint_velocity_rad_s)[a.order]])
            native=namespace['get_obs'](fake,trajectory,{'last_motor_targets':a.targets})[0]['state'].astype(np.float32)[None]
            author_error=max(author_error,float(np.max(abs(native-expected['obs']))));np.testing.assert_allclose(native,expected['obs'],atol=2e-6,rtol=2e-6)
            fake.mj_data.qpos[:3]=[1000,-1000,42];fake.mj_data.qvel[:6]=1000
            counterfactual=namespace['get_obs'](fake,trajectory,{'last_motor_targets':a.targets})[0]['state'].astype(np.float32)[None]
            np.testing.assert_array_equal(native,counterfactual)
            for key,value in got.items():
                actual=value[row:row+1].cpu().numpy();err=float(np.max(abs(actual-expected[key])));max_observation=max(max_observation,err)
                np.testing.assert_allclose(actual,expected[key],atol=2e-6,rtol=2e-6,err_msg=key)
            action=oracle.run([a.action_output],expected)[0].reshape(-1);expected_actions.append(action);a.targets=(np.asarray(target.joint_position_rad)[a.order]+action*a.scale).astype(np.float32)
        with torch.inference_mode():actual=batched.model(**got)[batched.action_output]
        expected=np.stack(expected_actions);value=actual.cpu().numpy();max_action=max(max_action,float(np.max(abs(value-expected))))
        np.testing.assert_allclose(value,expected,atol=5e-5,rtol=2e-5)
        clip=cfg.get('action_clip');a_value=np.clip(value,-clip,clip) if clip is not None else value;e_value=np.clip(expected,-clip,clip) if clip is not None else expected
        target_error=float(np.max(abs((a_value-e_value)*original[0].scale)));max_target=max(max_target,target_error)
        if target_error>5e-5:raise AssertionError('Joint position target error exceeds 5e-5 rad')
        # Same raw action histories isolate the observation and inference tests.
        batched.targets.copy_(tensor(np.stack([a.targets for a in original])))
    report={'checkpoint_sha256':sha256_file(args.weights),'adapter_source_sha256':{p.name:sha256_file(p) for p in root.glob('tensor*.py')},'graph_sha256':sha256_file(args.converted/'graph.json'),'status':'passed','steps':24,'batch':slots,'joint_order':'reversed canonical names','partial_reset_step':11,'partial_reset_slots':[1,4],
            'observation_max_abs_error':max_observation,'network_max_abs_error':max_action,'joint_target_max_abs_error_rad':max_target,
            'reference':'Original adapter.py observations and original ONNX per sample','history':'Matched final joint targets isolate observation and inference parity','author_source':source_meta,'author_observation_max_abs_error':author_error,'forbidden_actual_translation_velocity_counterfactual':'passed'}
    (root/'validation').mkdir(exist_ok=True);(root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
