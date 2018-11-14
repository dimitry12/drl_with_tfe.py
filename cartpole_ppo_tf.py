from sklearn.model_selection import GridSearchCV
import pandas as pd
import namesgenerator
from ppo import main


default_hyperparameters = {
    'VERSION': '2.1.0',
    'ENV': 'CartPole-v0',
    'RANDOM_SEED': 42,
    'RENDER': True,

    'MLP_UNITS': 16,
    'MLP_LAYERS': 1,  # one shared hidden layer between V and P
    'P_MLP_LAYERS': 0,  # policy network is linear
    'V_MLP_LAYERS': 1,  # V network is "deep"

    # the dimensionality for all these is: "MDP state transitions" (not observational frames)
    'GRADIENT_LEARNING_BATCH_SIZE': 32,
    'TRANSITIONS_IN_EXPERIENCE_BUFFER': 1024,
    'HORIZON': 1024,
    'TOTAL_ENV_STEPS': 1e5,  # 2e7,

    'EPOCHS_PER_UPDATE': 4,

    'CLIP_RANGE': .2,
    'GAMMA': .99,
    'ADVANTAGE_LAMBDA': .97,
    'VALUE_LAMBDA': .99,
    'MAX_GRAD_NORM': .5,
    'VALUE_LOSS_WEIGHT': .25,
    'LR': 3e-4,
}


class RLEstimator():
    params_default = default_hyperparameters
    params_values = {}
    score_ = None
    _flushed_score = False
    _random_name = ''

    def __init__(self, *args, **kwargs):
        self.set_params(**kwargs)

    def get_params(self, *args, **kwargs):
        return self.params_values

    def set_params(self, *args, **kwargs):
        for param, param_default_value in self.params_default.items():
            if param in kwargs:
                self.params_values[param] = kwargs[param]
            elif param in self.params_values:
                pass
            else:
                self.params_values[param] = param_default_value

    def fit(self, X):
        self._random_name = namesgenerator.get_random_name()
        self.score_ = main(hparams=self.params_values,
                           random_name=self._random_name)
        return self

    def score(self, X):
        if not self._flushed_score:
            self._flushed_score = True
            with open('score.log', 'a') as log:
                log.write(json.dumps(
                    {'score': self.score_, 'random_name': self._random_name, **self.params_values}) + "\n")
        return self.score_


if __name__ == '__main__':
    main(hparams=default_hyperparameters)
    exit()
    INITS_PER_HYPERSET = 2
    assert INITS_PER_HYPERSET % 2 == 0
    rle = RLEstimator()
    gs = GridSearchCV(rle, {
        'RENDER': [False],
        'RANDOM_SEEED': [False],
        'GAMMA': [0.95, 0.99],
        'ADVANTAGE_LAMBDA': [0.8],
        'TOTAL_ENV_STEPS': [10_000],
    }, cv=INITS_PER_HYPERSET, n_jobs=-1, refit=False)
    gs.fit([.0]*INITS_PER_HYPERSET)
    print(pd.DataFrame(gs.cv_results_).filter(
        regex='^(param_)|(mean_test_score)'))
