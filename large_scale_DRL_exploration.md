### 两篇关联论文

- Cao Y, Hou T, Wang Y, et al. Ariadne: A reinforcement learning approach using attention-based deep networks for exploration[C]//2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023: 10219-10225.
  - https://github.com/marmotlab/ARiADNE.git
  - 同时有训练代码 https://github.com/marmotlab/ARiADNE
  - 安装与使用：按步骤安装即可。
    - 注意额外安装以下： ros-noetic-octomap-server
    - 在虚拟环境中，使用 ``catkin_make -j10 -DPYTHON_EXECUTABLE=/usr/bin/python3``
    - 在运行时，要将 ARiADNE 运行在指定虚拟环境中。
  - 经典 SAC，均匀分布 viewpoint
  - 策略网络输出机器人当前位置的临近节点
  - sensor.py 中的参数 max_collision = 10 会影响探索效率，此处此处是为了多探索一点障碍物边缘, 避免探索率不到100%，似乎对于训练十分重要。此处，参数必须大于1,否则无障碍物被设置，无法在地图中设置静态障碍物。对于 test_dirver.py 中 episode_number = 100, max_collision = 10 时探索路径长度为 295.8米，
- Cao Y, Zhao R, Wang Y, et al. Deep Reinforcement Learning-based Large-scale Robot Exploration[J]. IEEE Robotics and Automation Letters, 2024.
  - https://arxiv.org/pdf/2403.10833
  - https://github.com/marmotlab/large-scale-DRL-exploration 和 ros 接口测试程序 https://github.com/marmotlab/ARiADNE-ROS-Planner.git 。
  - ARiADNE 的升级版，在其基础上使用 ground_truth 进行训练，并提出了图稀疏化。首先将 utility 为 0 的 view-point删除，将非零的分组，通过 A* 查找机器人位置到每组的最短路径，然后检查这些路径上每一个 line of sight 构成最小结点集作为网络输入。 utility: 在 view-point 处所能观察到的前沿点的数量。
  - 策略网络计算
    - 编码器输入： (1,NODE_PADDING_SIZE,4) -> initial_embedding (4,EMBEDDING_DIM) ->
    - 解码器输出： （EMBEDDING_DIM）
    - 策略网络输出： Pointer networks, (1, K_SIZE)
    - 注意： EncoderLayer 和 DecoderLayer 有问题啊，**与原始论文中的normalization 顺序和位置不同**，与DCMRTA 中的实现也不同。 标准的 transformer 实现： https://github.com/hyunwoongko/transformer

### 潜在改进点

- 均匀分布网格，对于模型窄区域部分，可能导致无法通行。
