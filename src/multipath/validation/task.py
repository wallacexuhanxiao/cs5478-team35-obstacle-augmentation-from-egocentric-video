import numpy as np
from .physics import core_pose_errors
from multipath.execution.mounted_scene import pose_matrix


def evaluate_task(state,manifest,contract,anchor_reports,records=None):
    err=core_pose_errors(state,pose_matrix(manifest['final_objects']['tool']))
    speed=float(np.linalg.norm(state['objects']['tool']['velocity'][:3]))
    # The source decides the final contact: a demonstration that ends still
    # holding the placed object is not asked to release it.
    final_contact=contract.get('final_contact','free')
    # Held at the end, the grip passes contact chatter to the object: stillness is then the measured displacement over
    # the last source frame (per-frame records); the instantaneous PhysX velocity is reported alongside.
    held_speed=None
    if final_contact!='free' and records and len(records)>=2 and records[-1]['time_s']>records[-2]['time_s']:
        a,b=records[-2],records[-1]
        held_speed=float(np.linalg.norm(np.subtract(b['objects']['tool']['pose'][:3],a['objects']['tool']['pose'][:3]))
                         /(b['time_s']-a['time_s']))
    still=(speed if held_speed is None else held_speed)<contract['stable_speed_m_s']
    passed=bool(anchor_reports and all(a['local_rollout_success'] and a['handoff_verified'] for a in anchor_reports)
                and err['position_m']<contract['position_tolerance_m']
                and err['rotation_rad']<contract['rotation_tolerance_rad']
                and still and state['contact']['mode']==final_contact)
    return dict(success=passed,final_errors=err,final_speed_m_s=speed,released=state['contact']['mode']=='free',
                required_final_contact=final_contact,
                **({} if held_speed is None else dict(held_final_frame_speed_m_s=held_speed,
                   stillness='held: measured displacement over the last source frame')))
