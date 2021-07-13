#!/usr/bin/env bash

#cd gather_expert_demonstration/

## generate Hopper-v2 expert demonstration
#python -m gather_expert_demonstration.generate_expert_trajectory \
#       --env_name "Hopper-v2" \
#       --actor_path "./models/Hopper/sac_actor_Hopper" \
#       --critic_path  "./models/Hopper/sac_critic_Hopper"
## Average length: 1000.0
## Average return: 3450.0


## generate ANt-v2 expert demonstration
#python -m gather_expert_demonstration.generate_expert_trajectory \
#       --env_name "Ant-v2" \
#       --actor_path "./models/Ant/sac_actor_Ant" \
#       --critic_path  "./models/Ant/sac_critic_Ant"
##Average length: 1000.0
##Average return: 5688.6615694237635


#generate HalfCheetah-v2 expert demonstration
python  -m gather_expert_demonstration.generate_expert_trajectory \
        --env_name "HalfCheetah-v2" \
        --actor_path "../models/sac_actor_HalfCheetah-v2_11938" \
        --critic_path  "../models/sac_critic_HalfCheetah-v2_11938"
#Average length: 1000.0
#Average return: 11588.00914054808



# generate Humanoid-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "Humanoid-v2" \
#        --actor_path "./models/Humanoid/sac_actor_Humanoid" \
#        --critic_path  "./models/Humanoid/sac_critic_Humanoid"




## generate Walker2d-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "Walker2d-v2" \
#        --actor_path "./models/Walker2d/sac_actor_Walker2d" \
#        --critic_path "./models/Walker2d/sac_critic_Walker2d"
##Average length: 1000.0
##Average return: 4913.904440835768
