import numpy as np

ParticleMomentum_MEAN = np.array([0., 0.])
ParticlePoint_MEAN = np.array([0., 0.])

def scoring_function(solution_file, predict_file):
    score = 0.
    
    solution = np.load(solution_file, allow_pickle=True)
    predict = np.load(predict_file, allow_pickle=True)
    ParticleMomentum_sol = solution['ParticleMomentum'][:, :2]
    ParticlePoint_sol = solution['ParticlePoint'][:, :2]
    
    ParticleMomentum_pred = predict['ParticleMomentum'][:, :2]
    ParticlePoint_pred = predict['ParticlePoint'][:, :2]
    
    score += np.sum(np.square(ParticleMomentum_sol - ParticleMomentum_pred).mean(axis=0) / np.square(ParticleMomentum_sol - ParticleMomentum_MEAN).mean(axis=0))
    score += np.sum(np.square(ParticlePoint_sol - ParticlePoint_pred).mean(axis=0) / np.square(ParticlePoint_sol - ParticlePoint_MEAN).mean(axis=0))
    return np.sqrt(score)
