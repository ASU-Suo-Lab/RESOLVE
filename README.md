# RESOLVE: A Multi-Resolution and Multi-Modal Dataset for Roadside Cooperative Perception
[ECCV 2026] The official codebase for the paper "RESOLVE: A Multi-Resolution and Multi-Modal Dataset for Roadside Cooperative Perception"

![RESOLVE Overview](assets/overview.jpg)

---

## 📢 Announcements
Stay up to date with the latest news, updates, and important notices regarding RESOLVE:

- **`2026/06/18`**: Paper accepted at ECCV 2026 🎉.
- **`2026/06/21`**: The v0.1 dataset and benchmark code for 3D object detection has been released. The currently supported 3D object detection models are as follows: [PointPillars](https://arxiv.org/abs/1812.05784), [SECOND](https://www.mdpi.com/1424-8220/18/10/3337), [CenterPoint](https://arxiv.org/abs/2006.11275), [TransFusion](https://arxiv.org/abs/2203.11496), [VoxSeT](https://arxiv.org/abs/2203.10314), [DSVT](https://arxiv.org/abs/2301.06051), [Voxel Mamba](https://arxiv.org/abs/2406.10700), [LION](https://arxiv.org/abs/2407.18232), [BEVFusion](https://arxiv.org/abs/2205.13542), [UniTR](https://arxiv.org/abs/2308.07732)
- **`2026/06/22`**: The benchmark code for 3D multi-object tracking has been released. The currently supported 3D multi-object tracking models are as follows: [AB3DMOT](https://arxiv.org/abs/2008.08063), [CenterPoint](https://arxiv.org/abs/2006.11275), [SimpleTrack](https://arxiv.org/abs/2111.09621), [Poly-MOT](https://arxiv.org/abs/2307.16675), [MCTrack](https://arxiv.org/abs/2409.16149)

## ✅ TODO

- [x] Release the RESOLVE dataset
- [x] Release the benchmark code for 3D object detection
- [x] Release the benchmark code for 3D multi-object tracking
- [ ] Release the benchmark code for cooperative perception


## Data Download
You can download the v0.1 dataset via this [dropbox link](https://www.dropbox.com/scl/fi/uym5i78ih5fmr7lxxyu9g/202_scenes.zip?rlkey=x63s104j8bapaqpx4io47chi9&st=9396mm29&dl=0).

After downloading the data, please put the data in the following structure:
```shell
├── v2xreal
│   ├── train
|      |── 2023-03-17-15-53-02_1_0
│   ├── validate
│   ├── test
```

## Quick Start
### 3D Object Detection

### 3D Multi-Object Tracking

### Cooperative Perception


## Models Zoo

###  3D Object Detection Benchmarks

| Model        | Modality       | LiDAR Backbone       |   Low Resolution     | Mid Resolution     | High Resolution    |
|------------- |----------------|----------------------|----------------------|--------------------|--------------------|
| [PointPillars](tools/cfgs/sunlakes_models/pointpillar.yaml) | LiDAR          | Sparse Convolution   | 75.1 / 71.2          | 79.7 / 72.6        | 80.6 / 73.6        |
| [SECOND](tools/cfgs/sunlakes_models/second.yaml)       | LiDAR          | Sparse Convolution   | 68.2 / 66.5          | 76.6 / 71.2        | 78.0 / 73.1        |
| [CenterPoint](tools/cfgs/sunlakes_models/centerpoint.yaml)  | LiDAR          | Sparse Convolution   | 79.9 / 73.9          | 86.4 / 79.9        | 87.4 / 80.9        | 
| [TransFusion-L](tools/cfgs/sunlakes_models/transfusion.yaml)| LiDAR          | Sparse Convolution   | 82.5 / 76.3          | 86.6 / 80.1        | 89.1 / 82.4        | 
| [VoxSeT](tools/cfgs/sunlakes_models/voxset.yaml)       | LiDAR          | Transformer          | 87.4 / 82.9          | 89.1 / 81.6        | 90.1 / 83.1        |
| [DSVT](tools/cfgs/sunlakes_models/dsvt.yaml)         | LiDAR          | Transformer          | 85.9 / 81.5          | 94.1 / 87.0        | 94.6 / 87.5        | 
| [Voxel Mamba](tools/cfgs/sunlakes_models/voxel_mamba.yaml)  | LiDAR          | Mamba                | 85.7 / 81.7          | 94.9 / 88.5        | 95.4 / 89.1        | 
| [LION](tools/cfgs/sunlakes_models/lion.yaml)| LiDAR          | Mamba                | 88.1 / 84.2          | 95.1 / 89.8        | 95.9 / 91.1        |
| [BEVFusion](tools/cfgs/sunlakes_models/bevfusion.yaml)    | LiDAR + Camera | Sparse Convolution   | 86.3 / 79.8          | 92.6 / 84.8        | 93.1 / 86.4        | 
| [UniTR](tools/cfgs/sunlakes_models/unitr.yaml)        | LiDAR + Camera | Transformer          | 89.7 / 83.7          | 94.4 / 87.3        | 94.9 / 87.4        |


--- 
## 📝 License

- **Code**: Licensed under the **MIT** License. See [LICENSE](LICENSE) file for details.

- **Dataset**: Licensed under the Creative Commons Attribution 4.0 International [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en). You must give appropriate credit; Cannot be used for commercial purposes; You may not distribute modified versions of the dataset.
--- 
