
import numpy as np
import scenic

def get_verifai():
    """
    Defines the parameters for the attack   
    """
    scenario = scenic.scenarioFromString("""
param verifaiSamplerType = 'ce'
param amp = VerifaiRange(0,1)
param amp2 = VerifaiRange(0,1)
param freq = VerifaiRange(0, 5)
param speed = VerifaiRange(0.5,3.6)

                                         
                                         """)
    return scenario

def direct_to_waypoint(current_pos, waypoint, kp=1.0, vel_lim=1.0):

    error = waypoint - current_pos
    target_pos = current_pos + kp * error
    vel = kp * error
    speed = np.linalg.norm(vel)
    if speed > vel_lim:
        vel = vel / speed * vel_lim
        
    return target_pos, vel

def attacker_3d_mov(pos, waypoint, t, v_forward=0.7, amp=0.5, amp2=0.5, freq=1.0, v_max=2.5, dt=0.01):
    """
    Generates an attacking trajectiory towards a waypoint with oscillatory motion.
    """
    direction = waypoint - pos
    dist = np.linalg.norm(direction)
    
    if dist < 0.05:
        return pos
    #reference vector
    z_vector = np.array([0.0, 0.0, 1.0])

    e_par = direction / dist 
    if abs(np.dot(e_par, z_vector)) > 0.95:
        z_vector = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(z_vector, e_par)
    e1/= np.linalg.norm(e1)

    e2 = np.cross(e_par, e1)


    # e_perp = np.array([-e_par[1], e_par[0], 0.0])

    
    wp_vel = (v_forward*e_par + amp*np.sin(freq*t)*e1 + amp2*np.cos(freq*t)*e2)

    
    return pos + wp_vel* dt

def opt_function(def_p, obj_p, attackers_p, eps=1e-2):
    distance_threshold = 1.0
    print("Defender pos:", def_p)
    print("Object pos:", obj_p)
    print("Attackers pos:", attackers_p)

    diff_att =  attackers_p - obj_p
    d_att = np.linalg.norm(diff_att, axis=1)
    cost = distance_threshold - d_att
    cost = np.max(cost)
    return cost

def distance_to_object(obj_p, attackers_p):
    diff_att =  attackers_p - obj_p
    d_att = np.linalg.norm(diff_att, axis=1)
    return np.min(d_att)
    