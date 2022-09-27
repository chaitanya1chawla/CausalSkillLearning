import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from PIL import Image

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    e = GymEnv(env_name)
    e.reset()
    
    for path in demo_paths:
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        
        image_list = []
        for t in range(actions.shape[0]):
            e.step(actions[t])
            # e.env.mj_render() 
            
            # Trying to use the sim render instead of the display based rendering, so that we can grab images.. 
            img = np.flipud(e.env.sim.render(600, 600))
            print("Successfully got image from sim renderer.")
            image_object = Image.fromarray(img)
            image_object.save("DextrousHand.jpg")
            # image_list.append(img)
            
    print("About to save image list.")
    # np.save("Trial_Image_List.npy", image_list)

if __name__ == '__main__':
    main()
