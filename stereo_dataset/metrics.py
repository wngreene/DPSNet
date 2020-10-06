import numpy as np

def write_metrics_header(output_file, metrics_dict):
    """Write metrics header.
    """
    with open(output_file, "w") as ff:
        ff.write("file ")
        for key, value in metrics_dict.items():
            ff.write("{} ".format(key))
        ff.write("\n")
    return

def write_metrics(output_file, input_file, metrics_dict):
    """Write metrics as a line in output file.
    """
    with open(output_file, "a") as ff:
        ff.write("{} ".format(input_file))
        for key, value in metrics_dict.items():
            ff.write("{} ".format(value))
        ff.write("\n")

    return

def compute_avg_metrics(metrics_file):
    """Compute average metrics from metrics file.
    """
    keys = None
    with open(metrics_file, "r") as ff:
        header = ff.readline()
        keys = header.split()
    keys = keys[1:] # Skip filename.

    metrics = np.loadtxt(metrics_file, skiprows=1, usecols=range(1, len(keys) + 1))
    avg_metrics = np.mean(metrics, axis=0)

    avg_metrics_dict = {}
    for idx in range(len(keys)):
        avg_metrics_dict[keys[idx]] = avg_metrics[idx]

    avg_metrics_dict["num_samples"] = metrics.shape[0]

    return avg_metrics_dict

def get_depth_prediction_metrics(depthmap_true, depthmap_est):
    """Compute metrics commonly reported for KITTI depth prediction.

    Assumes no invalid inputs (i.e. mask has already been applied).

    Based on Monodepth.
    """
    thresh = np.maximum((depthmap_true / depthmap_est), (depthmap_est / depthmap_true))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (depthmap_true - depthmap_est) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(depthmap_true) - np.log(depthmap_est)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(depthmap_true - depthmap_est) / depthmap_true)

    sq_rel = np.mean(((depthmap_true - depthmap_est)**2) / depthmap_true)

    metrics = {"abs_rel": abs_rel,
               "sq_rel": sq_rel,
               "rmse": rmse,
               "rmse_log": rmse_log,
               "a1": a1,
               "a2": a2,
               "a3": a3}

    return metrics

def get_disparity_metrics(disparity_true, disparity_est, mask):
    """Compute disparity metrics reported for KITTI Stereo 2012 and 2015.

    Based on GANet.
    """
    num_samples = np.sum(mask)
    diff = np.abs(disparity_true[mask] - disparity_est[mask])
    rel_diff = diff / disparity_true[mask]

    # End-point-error (EPE) is simply the average disparity error.
    epe = np.mean(diff)

    # Percentage of outlier pixels (used in Stereo2012).
    outlier_rate1 = np.sum(diff > 1.0) / num_samples
    outlier_rate2 = np.sum(diff > 2.0) / num_samples
    outlier_rate3 = np.sum(diff > 3.0) / num_samples

    # D1 is the percentage of pixels with disparity error >3px *and*
    # disparity_rel_error >5% (used in Stereo2015).
    d1_all = np.sum((diff > 3.0) & (rel_diff > 0.05)) / num_samples

    metrics = {"epe": epe,
               "outlier_rate1": outlier_rate1,
               "outlier_rate2": outlier_rate2,
               "outlier_rate3": outlier_rate3,
               "d1_all": d1_all}

    return metrics
