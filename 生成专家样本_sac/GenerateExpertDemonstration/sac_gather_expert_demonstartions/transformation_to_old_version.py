import torch
import os

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]


if __name__ == "__main__":
    env_name = "Hopper-v2"
    file_names = get_imlist("models/" + env_name)
    if not os.path.exists('models/{}/{}'.format(env_name, "transformation_models")):
        os.makedirs('models/{}/{}'.format(env_name, "transformation_models"))
    for name in file_names:
        policy, critic = torch.load(name)
        s_name = name.split("/")
        torch.save([policy, critic],
                   'models/{}/{}/'.format(env_name, "transformation_models") + s_name[2],
                   _use_new_zipfile_serialization=False)
