import numpy as np

def calculate_distances_and_correspondences(
        target, source,
        max_correspondance_distance):
        
    return 

def calculate_best_fit_transform(source, target, correspondances):
    
    return 

    
def transform_points(points, transformation):
    
    return 


def calculate_rmse(distances):

    return 

def icp(target, source,
        max_correspondance_distance = 1000,
        max_iterations = 4,
        metric_delta_threshold = 1e-20):
    src = source.copy()
    prev_metric = float('inf')
    history = []
    dim = source.shape[1]
    total_transformation = np.eye(dim + 1)

    for i in range(max_iterations):
        # Step 1:
        distances, correspondances = calculate_distances_and_correspondences(
            target, src,
            max_correspondance_distance
            )
        
        # Step 2:
        iteration_transformation = calculate_best_fit_transform(src, target, correspondances)
        
        # Step 3:
        total_transformation = iteration_transformation @ total_transformation
        
        # Step 4:
        src = transform_points(src, iteration_transformation)
        
        # Step 5:
        metric = calculate_rmse(distances)
        
        history.append((metric, total_transformation))
        if abs(prev_metric - metric) < metric_delta_threshold:
            break
        prev_metric = metric
    
    return total_transformation, history