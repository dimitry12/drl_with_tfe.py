import datetime
import time
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import GridSearchCV
import pandas as pd
import json


class P_and_V_Model(tf.keras.Model):
    def __init__(self, classes, mlp_layers, mlp_units, v_mlp_layers, p_mlp_layers, heads='pv'):
        assert mlp_layers > 0  # need at least one layer
        super(P_and_V_Model, self).__init__(name="P_and_V_Model")
        # interestingly, model *must* be named if we use named layers
        # (Keras 2.1.6-tf)
        self._outputs = heads
        self.dense = tf.keras.layers.Dense(
            mlp_units, activation='tanh', name='first_shared_layer')
        self.hidden_denses = []
        for _ in range(mlp_layers - 1):
            self.hidden_denses.append(tf.keras.layers.Dense(
                mlp_units, activation='tanh', name='shared_hidden_layer' + str(_)))
        self.v_hidden_denses = []
        for _ in range(v_mlp_layers):
            self.v_hidden_denses.append(tf.keras.layers.Dense(
                mlp_units, activation='tanh', name='v_hidden_layer' + str(_)))
        self.p_hidden_denses = []
        for _ in range(v_mlp_layers):
            self.p_hidden_denses.append(tf.keras.layers.Dense(
                mlp_units, activation='tanh', name='p_hidden_layer' + str(_)))
        self.p_outputs = tf.keras.layers.Dense(
            classes, activation=None, name='p_logits')
        self.v_output = tf.keras.layers.Dense(
            1, activation=None, name='v_logit')

    def call(self, inputs):
        shared_latent = self.dense(inputs)
        for l in self.hidden_denses:
            shared_latent = l(shared_latent)

        v_latent = shared_latent
        for l in self.v_hidden_denses:
            v_latent = l(v_latent)

        p_latent = shared_latent
        for l in self.p_hidden_denses:
            p_latent = l(p_latent)

        p_logits = self.p_outputs(p_latent)
        v_logit = self.v_output(v_latent)

        if self._outputs == 'p':
            return p_logits
        elif self._outputs == 'v':
            return v_logit
        else:
            return p_logits, v_logit


def main(*, hparams):
    assert hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'] % hparams['GRADIENT_LEARNING_BATCH_SIZE'] == 0
    tf.enable_eager_execution()

    if hparams['RANDOM_SEED']:
        tf.set_random_seed(hparams['RANDOM_SEED'])
        np.random.seed(hparams['RANDOM_SEED'])
        random.seed(hparams['RANDOM_SEED'])

    env = gym.make('CartPole-v0')
    log_dir_name = './tf-logs/' + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
    writer = tf.contrib.summary.create_file_writer(log_dir_name)
    rl_writer = tf.contrib.summary.create_file_writer(log_dir_name + 'rl')
    writer.set_as_default()

    # only works for Box
    observations_space_dim_count = env.observation_space.shape[0]
    action_space_cardinality = env.action_space.n  # only works for Discrete

    pv_model = P_and_V_Model(classes=action_space_cardinality, mlp_layers=hparams['MLP_LAYERS'],
                             mlp_units=hparams['MLP_UNITS'], v_mlp_layers=hparams['V_MLP_LAYERS'], p_mlp_layers=hparams['P_MLP_LAYERS'])
    optimizer = tf.train.AdamOptimizer(
        learning_rate=hparams['LR'], epsilon=1e-5)

    learner_updates = int(
        hparams['TOTAL_ENV_STEPS'] // hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'])

    observation: np.ndarray = env.reset()
    episode_done: bool = False
    total_reward = 0
    total_episodes = 0

    for learner_update in range(1, learner_updates+1):
        with rl_writer.as_default(), tf.contrib.summary.always_record_summaries():
            current_update_started_at = time.time()
            episodes_in_current_update = 0
            total_rewards_in_current_update = []
            steps_in_current_update = []

            observations, rewards, taken_actions, predicted_values, neg_log_p_ac_s, episode_dones = [], [], [], [], [], []

            reward: float
            steps_in_current_episode = 0
            total_reward_in_current_episode = 0

            def vector_to_tf_constant(x): return tf.constant(
                x, dtype=tf.keras.backend.floatx(), shape=(1, len(x)))

            for _ in range(hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']):
                p_logits, v_logit = pv_model(
                    vector_to_tf_constant(observation))
                p_distribution = tf.distributions.Categorical(logits=p_logits)
                action = p_distribution.sample()
                tf.contrib.summary.histogram(
                    'policy_probabilities', p_distribution.probs)
                neg_log_p_ac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=p_logits, labels=action)

                # gym envs overwrite observations
                observations.append(observation.copy())
                episode_dones.append(episode_done)
                taken_actions.append(action)
                predicted_values.append(v_logit)
                neg_log_p_ac_s.append(neg_log_p_ac)

                observation, reward, episode_done, infos = env.step(
                    action.numpy()[0])
                rewards.append(reward)
                total_reward += reward
                if (hparams['RENDER']):
                    env.render()

                if episode_done:
                    observation = env.reset()
                    episodes_in_current_update += 1
                    total_episodes += 1
                    steps_in_current_update.append(steps_in_current_episode)
                    total_rewards_in_current_update.append(
                        total_reward_in_current_episode)
                    tf.contrib.summary.scalar('episode_reward',
                                              total_reward_in_current_episode)
                    tf.contrib.summary.scalar(
                        'episode_steps', steps_in_current_episode)
                    steps_in_current_episode = 0
                    total_reward_in_current_episode = 0

                steps_in_current_episode += 1
                total_reward_in_current_episode += reward

            _, last_v_logit = pv_model(vector_to_tf_constant(observation))

            # Each slice of experience now contains:
            # - observation
            # - flag indicating if this observation is
            #   the beginning of the new episode
            # - which action we took seeing this observation
            #   (plus neg_log_p_ac)
            # - which value we inferred seeing this observation
            # - which reward we received taking this action
            #
            # In other words, thnking causally:
            # - observation causes
            # - action, which causes
            # - reward
            # with all three being in the same experience-slice.
            #
            # Additionally we keep note of:
            # - the very last observation, which agent didn't take action
            #   on, yet agent can still infer the value of that last state.
            # - and if the very last observation was the beginning of the
            #   new episode.

            predicted_values = np.asarray(predicted_values, dtype=np.float32)
            gae_s = np.zeros_like(rewards, dtype=np.float32)
            v_targets = np.zeros_like(rewards, dtype=np.float32)
            # tail_of_gae  is lambda-discounted sum of per-slice surprises. We learn from surprises!
            # tail_of_returns is gamma-discounted sum of per-transition rewards (policy-value)
            for t in reversed(range(hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'])):
                is_this_last_slice = (
                    t == hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']-1)

                if is_this_last_slice:
                    did_this_slice_ended_episode = episode_done
                    next_slice_v_logit = last_v_logit
                    tail_of_gae = 0  # no heuristic for gamma-lambda-discounted sum of residuals
                    tail_of_returns = hparams['GAMMA'] * \
                        last_v_logit  # estimate of the tail
                else:
                    did_this_slice_ended_episode = episode_dones[t+1]
                    next_slice_v_logit = predicted_values[t+1]

                if did_this_slice_ended_episode:
                    tail_of_gae = 0  # zero-out the tail of generalized advantage estimation
                    tail_of_returns = 0
                    td_residual = rewards[t] - predicted_values[t]
                else:
                    td_residual = rewards[t] + hparams['GAMMA'] * \
                        next_slice_v_logit - predicted_values[t]

                gae_s[t] = td_residual + hparams['GAMMA'] * \
                    hparams['ADVANTAGE_LAMBDA'] * tail_of_gae
                v_targets[t] = rewards[t] + hparams['GAMMA']*tail_of_returns
                tail_of_gae = gae_s[t]
                tail_of_returns = v_targets[t]

            observations = np.asarray(observations, dtype=np.float32)
            for dim_i in range(0, observations_space_dim_count):
                tf.contrib.summary.histogram(
                    'visitation_state_dim_' + str(dim_i), observations[:, dim_i])
            taken_actions = np.asarray(taken_actions, dtype=np.int64)
            predicted_values = np.asarray(predicted_values, dtype=np.float32)
            neg_log_p_ac_s = np.asarray(neg_log_p_ac_s, dtype=np.float32)

            # GAE is lambda-exponentially-weighted sum of:
            # - k-step TD-residuals, and each of these is the difference of:
            #   - k-step value-estimate, and
            #   - baseline, which is policy-value of the current state
            # Baseline is constant throughout the expression, therefore
            # difference of GAE and baseline is lambda-exponentially-weighted
            # sum of k-step value-estimates.
            v_targets = gae_s + predicted_values

            tf.contrib.summary.histogram('GAEs', gae_s)

        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            dataset = tf.data.Dataset.from_tensor_slices(
                (observations, v_targets, taken_actions, predicted_values, neg_log_p_ac_s, gae_s))
            dataset = dataset.shuffle(
                hparams['EPOCHS_PER_UPDATE']*hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']).batch(hparams['GRADIENT_LEARNING_BATCH_SIZE'])
            v_losses = []
            for _ in range(hparams['EPOCHS_PER_UPDATE']):
                for batch_number, dataset_batch in enumerate(dataset):
                    observations_batch, v_targets_batch, old_taken_actions_batch, old_predicted_values_batch, old_neg_log_p_ac_s_batch, advantages_batch = dataset_batch

                    with tf.GradientTape() as tape:
                        train_p_logits, train_v_logit = pv_model(
                            observations_batch)
                        neg_log_p_ac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=train_p_logits, labels=old_taken_actions_batch)

                        # Only care about how proximate the new policy is
                        # relative to the probability of the *taken* action.
                        # New policy may offer very different distribution
                        # over untaken actions.
                        ratio = tf.exp(old_neg_log_p_ac_s_batch - neg_log_p_ac)
                        pg_losses = advantages_batch * ratio
                        pg_losses2 = advantages_batch * \
                            tf.clip_by_value(
                                ratio, 1.0 - hparams['CLIP_RANGE'], 1.0 + hparams['CLIP_RANGE'])
                        # effectively:
                        # - clip ratio 1.0+hparams['CLIP_RANGE'] when advantage is positive
                        # - clip ratio 1.0-hparams['CLIP_RANGE'] when advantage is negative
                        # (paper's wording is misleading, but graphs are correct)
                        pg_loss = tf.reduce_mean(
                            tf.minimum(pg_losses, pg_losses2))
                        tf.contrib.summary.scalar(
                            'batch_average_CLIP', pg_loss)
                        pg_loss = -pg_loss  # TF's optimizers minimize
                        tf.contrib.summary.histogram('untrusted_p_ratio_clipped_off', tf.to_float(
                            tf.greater(tf.abs(ratio-1.), hparams['CLIP_RANGE']))*tf.abs(ratio-hparams['CLIP_RANGE']))

                        v_pred_clipped = old_predicted_values_batch + \
                            tf.clip_by_value(
                                train_v_logit - old_predicted_values_batch, - hparams['CLIP_RANGE'], hparams['CLIP_RANGE'])
                        tf.contrib.summary.histogram('untrusted_v_diff_clipped_off', tf.abs(
                            v_pred_clipped - train_v_logit))
                        v_f_losses1 = tf.square(
                            train_v_logit - v_targets_batch)
                        tf.contrib.summary.scalar('batch_average_unclipped_v_loss',
                                                  tf.reduce_mean(v_f_losses1))
                        v_f_losses2 = tf.square(
                            v_pred_clipped - v_targets_batch)
                        # See several scenarios:
                        #
                        # 1) R.....O.C..V
                        # here V-R > C-R and thefore minimization will be applied to
                        # unclipped value-loss
                        #
                        # 2) R.V.C.O
                        # here C-R > V-R and therefore minimization will be applied
                        # to clipped value-loss
                        #
                        # To summarize, this objective will:
                        # - move predicted value closer to actual return
                        # - yet keep it within hparams['CLIP_RANGE'] of the old predicted value
                        v_f_loss = .5 * \
                            tf.reduce_mean(tf.maximum(
                                v_f_losses1, v_f_losses2))
                        tf.contrib.summary.scalar('batch_average_clipped_v_loss',
                                                  tf.reduce_mean(v_f_loss))
                        v_losses.append(v_f_loss.numpy())

                        # Total loss
                        loss = pg_loss + v_f_loss * \
                            hparams['VALUE_LOSS_WEIGHT']

                    grads = tape.gradient(loss, pv_model.variables)
                    grads, _grad_norm = tf.clip_by_global_norm(
                        grads, hparams['MAX_GRAD_NORM'])
                    optimizer.apply_gradients(zip(grads, pv_model.variables),
                                              global_step=tf.train.get_or_create_global_step())

            fps = int(hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'] /
                      (time.time() - current_update_started_at))

            print("update: ", learner_update)
            print("Hypers: ", hparams['GAMMA'], hparams['ADVANTAGE_LAMBDA'])
            print('episodes in update: ', episodes_in_current_update)
            print('average steps per episode: ', np.asarray(
                steps_in_current_update).mean())
            print('average reward per episode: ', np.asarray(
                total_rewards_in_current_update).mean())
            print('min reward per episode: ', np.asarray(
                total_rewards_in_current_update).min())
            print('max reward per episode: ', np.asarray(
                total_rewards_in_current_update).max())
            print("total timesteps so far", learner_update *
                  hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'])
            print("average of batch-averaged V losses",
                  np.array(v_losses).mean())
            tf.contrib.summary.scalar(
                'total_frames', tf.constant(learner_update*hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']))
            print("fps", fps)

    print('Total reward:', total_reward)
    print('Total episodes:', total_episodes)
    return float(total_reward)/float(total_episodes)


default_hyperparameters = {
    'VERSION': '0.1.0',
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
    'TOTAL_ENV_STEPS': 1e4,  # 2e7,

    'EPOCHS_PER_UPDATE': 4,

    'CLIP_RANGE': .2,
    'GAMMA': .99,
    'ADVANTAGE_LAMBDA': .97,
    'MAX_GRAD_NORM': .5,
    'VALUE_LOSS_WEIGHT': .25,
    'LR': 3e-4,
}


class RLEstimator():
    params_default = default_hyperparameters
    params_values = {}
    score_ = None

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
        self.score_ = main(hparams=self.params_values)
        return self

    def score(self, X):
        with open('score.log', 'a') as log:
            log.write(json.dumps({'score': self.score_, **self.params_values}))
        return self.score_


if __name__ == '__main__':
    main(hparams=default_hyperparameters)
    exit()
    INITS_PER_HYPERSET = 2
    assert INITS_PER_HYPERSET % 2 == 0
    rle = RLEstimator()
    gs = GridSearchCV(rle, {
        'RENDER': False,
        'RANDOM_SEEED': False,
        'GAMMA': [0.95, 0.99],
        'ADVANTAGE_LAMBDA': [0.8],
        'TOTAL_ENV_STEPS': [10_000],
    }, cv=INITS_PER_HYPERSET, n_jobs=-1, refit=False)
    gs.fit([.0]*INITS_PER_HYPERSET)
    print(pd.DataFrame(gs.cv_results_).filter(
        regex='^(param_)|(mean_test_score)'))
