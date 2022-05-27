import numpy as np
import open3d as o3d
from time import time


def show_ptxyz(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd])


class VoxelDSampleSort:
    def __init__(self, voxel_size, mode="center"):
        self.voxel_size = voxel_size # [x_size, y_size, z_size]
        self.mode = mode

    def downsample(self, pcd):
        # 1. min/max
        max_range = np.max(pcd, axis=0)
        min_range = np.min(pcd, axis=0)

        # 2. dims
        dims = np.floor((max_range - min_range) / self.voxel_size).astype(int)

        # 3. com idx
        idx = np.floor((pcd - min_range) / self.voxel_size).astype(int)
        idx = idx[:, 0] + idx[:, 1] * dims[0] + idx[:, 2] * dims[0] * dims[1]

        # 4. sort
        sorted_idx = idx.argsort()
        coords = idx[sorted_idx]
        sorted_pcd = pcd[sorted_idx]

        # 5. center sample or random sample
        if self.mode == "center":
            return self.center_sample(sorted_pcd, coords, sorted_idx)
        else:
            return self.random_sample(sorted_pcd, coords, sorted_idx)

    def center_sample(self, sorted_pcd, coords, sorted_idx):
        valid_voxels = np.unique(coords)
        ret = np.zeros(shape=[valid_voxels.shape[0], 3], dtype=np.float32)
        pt_idx = 0
        for i, voxel in enumerate(valid_voxels):
            num_pt_in_voxel = 0
            while pt_idx < coords.shape[0] and voxel == coords[pt_idx]:
                ret[i] += sorted_pcd[pt_idx]
                pt_idx += 1
                num_pt_in_voxel += 1
            ret[i] = ret[i] / num_pt_in_voxel
        return ret

    def random_sample(self, sorted_pcd, coords, sorted_idx):
        pass


class VoxelDSampleHashMap:
    def __init__(self):
        pass


if __name__ == '__main__':
    pc = np.fromfile("/home/lishiqi/develop/Mypcl/sampledata/H01_1634009186.127198.bin", np.float32).reshape(-1, 5)
    pc = pc[:, :3]

    print("pointnum before dowmsample: {}".format(pc.shape[0]))
    dsampler = VoxelDSampleSort([0.2, 0.2, 0.2])
    tic = time()
    ret = dsampler.downsample(pc)
    toc = time()
    print("Voxel Downsampl based center: {} ms".format((toc - tic) * 1000))
    print("pointnum after voxel downsample: {}".format(ret.shape[0]))
    # show_ptxyz(ret)

