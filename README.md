# Oneshot Localization - LiDAR-only Method


## Quick Start
```bash
cd Localization/CVL
bash run_cvl_2.sh   # Drone 2
bash run_cvl_1.sh   # Drone 1
```
- Note: CVL estimates the pose within ±6 m and ±30° of the initial pose. If the initial pose changes significantly, please update INITIAL_X, INITIAL_Y, and INITIAL_YAW in sh files.

## Directory Layout

```text
/home/emnavi/X280/src/localization
├── localisation_fg2.tar.gz
└── CVL/
    ├── maps/mph/
    ├── test_sample/mph/
    ├── calib/
    └── run_cvl_1...6.sh
```

Docker image archive: `localisation_fg2.tar.gz`  
Docker image tag after loading: `localisation:fg2`

## Load Docker Image

Run once before the first test:

```bash
cd /home/emnavi/X280/src/localization
sudo docker load -i localisation_fg2.tar.gz
sudo docker image inspect localisation:fg2
```

## Test with ROS Topics

```bash
cd /home/emnavi/X280/src/localization/CVL
./run_cvl_1.sh
```

Override the initial metric map pose with environment variables, for example:

```bash
INITIAL_X=8.3 INITIAL_Y=8.3 INITIAL_YAW=45 ./run_cvl_1.sh
```

Output topic: `/agent001/oneshot_localization_result`
