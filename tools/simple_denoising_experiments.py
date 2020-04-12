"""
This file contains a basic set of label denoising experiments.
Generally, these functions produce a cache file that is better
displayed inside of a Jupyter notebook.
"""
import os.path as osp


def exp_bernoulli_grid_label_noise_knn_k_max_vote():
    cache_fn = "./output/cache/simple_bernoulli_grid_label_noise_knn_k_max_vote.pkl"
    if osp.exists(cache_fn):
        results_bergrid_knn = read_pickle(cache_fn)
    else:
        # run experiment with various knn k
        results_bergrid_knn = []
        k_values = [3, 5, 8, 10, 12, 15, 20, 25, 30, 35, 50]
        for knn_k in k_values:
            bernoulli_thresh_cfg.knn.k = knn_k
            label_noise_cfg_list = rs.create_label_noise_experiment_cfg(bernoulli_thresh_cfg,label_noise_levels)
            results_bgk = rs.run_experiments(label_noise_cfg_list,ld.maximum_vote_grid_knn,nrepeats=50)
            results_bergrid_knn.append(results_bgk)
        write_pickle(results_bergrid_knn,cache_fn)    
    print("Project code complete")
    return results_bergrid_knn

def main():
    exp_bernoulli_grid_label_noise_knn_k_max_vote()

if __name__ == "__main__":
    main()
