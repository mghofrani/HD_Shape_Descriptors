import numpy as np
from scipy import ndimage
import torch

# def normalize_point_cloud(point_cloud):
#     # Compute mean along each axis
#     mean_xyz = np.mean(point_cloud, axis=1, keepdims=True)

#     # Compute standard deviation along each axis
#     std_xyz = np.std(point_cloud, axis=1, keepdims=True)

#     # Normalize each coordinate independently using its own mean and standard deviation
#     normalized_point_cloud = (point_cloud - mean_xyz) / std_xyz

#     return normalized_point_cloud

def normalize_point_cloud(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points

# Example usage
def find_surface_indices(array, num_points):
    labeled_array, num_features = ndimage.label(array)
    surface_indices = []

    for label in range(1, num_features + 1):
        component = (labeled_array == label)
        borders = ndimage.find_objects(component)

        for border in borders:
            x_slices, y_slices, z_slices = border
            x_vals, y_vals, z_vals = np.where(component[x_slices, y_slices, z_slices])
            x_coords = x_slices.start + x_vals
            y_coords = y_slices.start + y_vals
            z_coords = z_slices.start + z_vals

            coordinates = np.column_stack((x_coords, y_coords, z_coords))
            surface_indices.extend(coordinates)
            
    # Normalize the surface points
    surface_indices = normalize_point_cloud(surface_indices)
    
    num_surface_points = len(surface_indices)

    if num_points < num_surface_points:
        sampled_surface_indices = np.random.choice(num_surface_points, size=num_points, replace=False)
    else:
        sampled_surface_indices = np.random.choice(num_surface_points, size=num_points, replace=True)

    sampled_surface_points = np.array(surface_indices)[sampled_surface_indices]
    sampled_surface_points = np.transpose(sampled_surface_points)

    return  torch.Tensor(sampled_surface_points)

def chamfer_distance(pointcloud1, pointcloud2):
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
    - pointcloud1: Tensor of shape (B, 3, N) representing the first point cloud
    - pointcloud2: Tensor of shape (B, 3, M) representing the second point cloud
    
    Returns:
    - chamfer_dist: Tensor of shape (B,) containing the Chamfer Distance for each batch element
    """
    B1, D1 , N = pointcloud1.shape
    # print(pointcloud1.shape)
    B2, D2 , M = pointcloud2.shape
    # print(pointcloud2.shape)

    if D1 !=3 or D2 !=3:
        raise Exception("dimension is not 3")

    if B1 != B2 :
        raise Exception("batch sizes do not match")

  # Reshape point clouds for computation
    pointcloud1 = pointcloud1.permute(0, 2, 1).contiguous()  # (B, N, 3)
    # print(pointcloud1.shape)
    
    pointcloud2 = pointcloud2.permute(0, 2, 1).contiguous()  # (B, M, 3)
    # print(pointcloud2.shape)

    # Compute pairwise distances
    dist1 = torch.cdist(pointcloud1, pointcloud2, p=2)  # (B, N, M)
    # print(dist1.shape)
    
    dist2 = torch.cdist(pointcloud2, pointcloud1, p=2)  # (B, M, N)
    # print(dist2.shape)

    # Find nearest neighbors
    min_dist1, _ = torch.min(dist1, dim=2)  # (B, N)
    # print(min_dist1.shape)

    min_dist2, _ = torch.min(dist2, dim=2)  # (B, M)
    # print(min_dist2.shape)

    # Compute Chamfer distance
    chamfer_dist = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)
    # print(chamfer_dist.shape)

    return chamfer_dist

import torch

def earth_mover_distance(pointclouds1, pointclouds2):
    """
    Computes the Earth Mover's Distance (EMD) between batches of point clouds using PyTorch.

    Args:
    - pointclouds1: Tensor of shape (B, 3, N) representing the first batch of point clouds
    - pointclouds2: Tensor of shape (B, 3, M) representing the second batch of point clouds

    Returns:
    - emd: Tensor of shape (B,) containing the Earth Mover's Distance for each batch element
    """
    B, _, N = pointclouds1.shape
    _, _, M = pointclouds2.shape

    emd = torch.zeros(B, dtype=torch.float32)

    for b in range(B):
        distances = torch.cdist(pointclouds1[b].T, pointclouds2[b].T)
        emd[b] = torch.sum(distances) / (N * M)

    emd = emd.sum()

    return emd

