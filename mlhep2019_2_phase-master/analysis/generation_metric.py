from prd_score import compute_prd_from_embedding
from calogan_metrics import get_assymetry, get_shower_width, get_sparsity_level
from calogan_metrics import get_physical_stats
from sklearn.metrics import auc
import numpy as np
def scoring_function(solution_file, predict_file):
    np.random.seed(1337)
    score = 0.
    
    solution = np.load(solution_file, allow_pickle=True)
    predict = np.load(predict_file, allow_pickle=True)
    
    EnergyDeposit_sol = solution['EnergyDeposit']
    ParticleMomentum_sol = solution['ParticleMomentum']
    ParticlePoint_sol = solution['ParticlePoint']
    
    EnergyDeposit_pred = predict['EnergyDeposit']
    
    precision, recall = compute_prd_from_embedding(
                        EnergyDeposit_sol.reshape(len(EnergyDeposit_sol), -1), 
                        EnergyDeposit_pred.reshape(len(EnergyDeposit_sol), -1),,
                        num_clusters=100,
                        num_runs=100)
    
    auc_img = auc(precision, recall)

    
    physical_metrics_sol = get_physical_stats(
        EnergyDeposit_sol, 
        ParticleMomentum_sol,
        ParticlePoint_sol)

    physical_metrics_pred = get_physical_stats(
        EnergyDeposit_pred, 
        ParticleMomentum_sol,
        ParticlePoint_sol)
    
    precision, recall = compute_prd_from_embedding(
        physical_metrics_sol, 
        physical_metrics_pred,
        num_clusters=100,
        num_runs=100)
    
    auc_physical_metrics = auc(precision, recall)

    
    return min(auc_img, auc_physical_metrics)
