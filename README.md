# 17-07 how to run


docker-compose up -d

## T1
docker exec -it tamlin bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch engage perceice.launch

## T2
docker exec -it compare bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun compare run.py

## T3
docker exec -it compare bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rostopic pub /trigger_topic ... start
rostopic pub /trigger_topic ... stop


# Compare various engagement metrics for HRI

Create a docker container in each engagement folder so that each module can be run independently.

Available repositories:

- Anshul https://github.com/vignif/eng_0
- Del Duchetto https://github.com/vignif/eng_1
- Migrave https://github.com/vignif/eng_2
- Amogh https://github.com/vignif/eng_3 

## Run each component independently

run the roscore
```
cd core
docker compose up --build --remove-orphans -d
```


run the python script for playing the video
```
cd main
docker compose up --build --remove-orphans -d
```


## Run the whole pipeline all together

```
docker compose up --build --remove-orphans -d
```


## TODO

rosrun tf2_tools view_frames.py

Make a node that is collecting engagement from different models. 
1. the topic /humans/interactions/engagements of type engage/EngagementValue with the structure
```
header: 
  seq: 545
  stamp: 
    secs: 1496944479
    nsecs: 389547164
  frame_id: ''
person_a: "ihzop"
person_b: ''
distance: 1.4045894145965576
mutual_gaze: 0.8617962598800659
engagement: 0.6135574579238892
confidence_a: 0.6241523623466492
confidence_b: 1.0
```
2. the engagement topic from GRACE


# Build only one service
docker-compose up -d --no-deps --build <name>

# Run a docker with custom environment
docker run --network eng_ros_network -it --gpus all -e ROS_MASTER_URI=http://172.21.0.2:11311 -e ROS_HOSTNAME=172.21.0.6 --name grace grace

# compare_engagement_labserver
# compare_engagement_labserver
