import datetime
import time
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import json
import os


def get_actions_and_neglogp(p_logits, postprocess_preds, actions=None):
    p_logits = postprocess_preds(p_logits)
    p_distributions = [tf.distributions.Categorical(
        logits=_p_logits) for _p_logits in p_logits]

    if actions is None:
        # batch of 1
        actions = [tf.reshape(_p_distribution.sample(), (1,))
                   for _p_distribution in p_distributions]

        # only log if we are actively generating actions
        for i, _p_distribution in enumerate(p_distributions):
            tf.contrib.summary.histogram(
                'policy_probabilities_' + str(i), _p_distribution.probs)
    else:
        # actions must be a list of (batchsize,1) tensors
        # for zip-map to work
        actions = tf.split(actions, actions.shape[1], axis=1)
        actions = [tf.reshape(a, (a.shape[0],)) for a in actions]

    def neg_log_p_ac_func(p, a):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=a)

    neg_log_p_ac = tf.add_n([neg_log_p_ac_func(_p_logits, _action)
                             for _p_logits, _action in zip(p_logits, actions)])

    return actions, neg_log_p_ac


def get_transformations(env):
    def vector_to_tf_constant(x, dtype=tf.keras.backend.floatx()):
        return tf.constant(tf.cast(x, dtype), dtype=dtype)

    def passthrough(tensor):
        return tensor

    def one_hot(tensor, dims):
        return tf.layers.flatten(tf.one_hot(tensor, dims))

    if isinstance(env.observation_space, gym.spaces.Box):
        observations_space_dim_count = env.observation_space.shape[0]

        def preprocess_obs(x): return passthrough(vector_to_tf_constant(x))
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        observations_space_dim_count = 1

        def preprocess_obs(x): return one_hot(
            vector_to_tf_constant(x, dtype=tf.int32), env.observation_space.n)

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_space_cardinalities = [
            env.action_space.n]  # only works for Discrete
    elif isinstance(env.action_space, gym.spaces.tuple_space.Tuple) and not [s for s in env.action_space.spaces if not isinstance(s, gym.spaces.Discrete)]:
        action_space_cardinalities = [s.n for s in env.action_space.spaces]

    def postprocess_preds(x):
        return tf.split(x, action_space_cardinalities, axis=1)

    return (observations_space_dim_count, preprocess_obs, sum(action_space_cardinalities), postprocess_preds)


def calculate_gae(*, hparams, ADVANTAGE_LAMBDA, rewards, episode_dones, predicted_values, episode_done, last_v_logit):
    gae_s = np.zeros_like(rewards, dtype=np.float32)
    # tail_of_gae  is lambda-discounted sum of per-slice surprises. We learn from surprises!
    for t in reversed(range(hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'])):
        is_this_last_slice = (
            t == hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']-1)

        if is_this_last_slice:
            did_this_slice_ended_episode = episode_done
            next_slice_v_logit = last_v_logit
            tail_of_gae = 0  # no heuristic for gamma-lambda-discounted sum of residuals
        else:
            did_this_slice_ended_episode = episode_dones[t+1]
            next_slice_v_logit = predicted_values[t+1]

        if did_this_slice_ended_episode:
            tail_of_gae = 0  # zero-out the tail of generalized advantage estimation
            td_residual = rewards[t] - predicted_values[t]
        else:
            td_residual = rewards[t] + hparams['GAMMA'] * \
                next_slice_v_logit - predicted_values[t]

        gae_s[t] = td_residual + hparams['GAMMA'] * \
            ADVANTAGE_LAMBDA * tail_of_gae
        tail_of_gae = gae_s[t]

    return gae_s


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


def main(*, hparams, random_name=''):
    assert hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER'] % hparams['GRADIENT_LEARNING_BATCH_SIZE'] == 0
    tf.enable_eager_execution()

    if hparams['RANDOM_SEED']:
        tf.set_random_seed(hparams['RANDOM_SEED'])
        np.random.seed(hparams['RANDOM_SEED'])
        random.seed(hparams['RANDOM_SEED'])

    env = gym.make(hparams['ENV'])
    random_name = datetime.datetime.now().strftime("%Y%m%d%H%M") + '-' + random_name
    with open('./tf-logs/' + random_name + '.hparams.json', 'a') as log:
        log.write(json.dumps(hparams))
    log_dir_name = './tf-logs/' + random_name + '-'
    writer = tf.contrib.summary.create_file_writer(log_dir_name)
    rl_writer = tf.contrib.summary.create_file_writer(log_dir_name + 'rl')

    observations_space_dim_count, preprocess_obs, action_space_cardinality, postprocess_preds = get_transformations(
        env)

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

            for _ in range(hparams['TRANSITIONS_IN_EXPERIENCE_BUFFER']):
                if isinstance(observation, int):
                    observation = np.array([observation])
                p_logits, v_logit = pv_model(
                    preprocess_obs([observation]))
                action, neg_log_p_ac = get_actions_and_neglogp(
                    p_logits, postprocess_preds)

                # gym envs overwrite observations
                observations.append(observation.copy())
                episode_dones.append(episode_done)
                taken_actions.append(action)
                predicted_values.append(v_logit)
                neg_log_p_ac_s.append(neg_log_p_ac)

                observation, reward, episode_done, infos = env.step(
                    [a.numpy()[0] for a in action])
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

            _, last_v_logit = pv_model(preprocess_obs([observation]))

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
            tf.contrib.summary.histogram(
                'predicted_values_in_buffer', predicted_values)

            # GAE is lambda-exponentially-weighted sum of:
            # - k-step TD-residuals, and each of these is the difference of:
            #   - k-step value-estimate, and
            #   - baseline, which is policy-value of the current state
            # Baseline is constant throughout the expression, therefore
            # sum of GAE and baseline is lambda-exponentially-weighted
            # sum of k-step value-estimates.
            gae_s = calculate_gae(hparams=hparams, ADVANTAGE_LAMBDA=hparams['VALUE_LAMBDA'], rewards=rewards,
                                  episode_dones=episode_dones, predicted_values=predicted_values, episode_done=episode_done, last_v_logit=last_v_logit)
            v_targets = gae_s + predicted_values

            gae_s = calculate_gae(hparams=hparams, ADVANTAGE_LAMBDA=hparams['ADVANTAGE_LAMBDA'], rewards=rewards,
                                  episode_dones=episode_dones, predicted_values=predicted_values, episode_done=episode_done, last_v_logit=last_v_logit)

            observations = np.asarray(observations, dtype=np.float32)
            for dim_i in range(0, observations_space_dim_count):
                tf.contrib.summary.histogram(
                    'visitation_state_dim_' + str(dim_i), observations[:, dim_i])
            taken_actions = np.asarray(taken_actions, dtype=np.int64)
            neg_log_p_ac_s = np.asarray(neg_log_p_ac_s, dtype=np.float32)

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
                            preprocess_obs(observations_batch))
                        _, neg_log_p_ac = get_actions_and_neglogp(
                            train_p_logits, postprocess_preds, actions=old_taken_actions_batch)

                        # Only care about how proximate the updated policy is
                        # relative to the probability of the *taken* action.
                        # Updated policy may offer very different distribution
                        # over untaken actions.
                        ratio = tf.exp(old_neg_log_p_ac_s_batch - neg_log_p_ac)

                        pg_losses = advantages_batch * ratio
                        pg_losses2 = advantages_batch * \
                            tf.clip_by_value(
                                ratio, 1.0 - hparams['CLIP_RANGE'], 1.0 + hparams['CLIP_RANGE'])

                        # This effectively does:
                        # - clips ratio from going above 1.0+hparams['CLIP_RANGE'] when advantage is positive
                        #   (i.e. makes objective indifferent to changes in ratio above the upper-threshold)
                        # - clips ratio from going under 1.0-hparams['CLIP_RANGE'] when advantage is negative
                        #   (i.e. makes objective indifferent to changes in ratio below the lower-threshold)
                        # - YET, IMPORTANTLY:
                        #   - it DOES NOT clip the ratio from going under the lower-threshold when advantage is positive
                        #   - and therefore: objective IS sensitive (can suffer) if ratio goes deep under the
                        #     low-threshold.
                        # Using wording from the paper:
                        # "With this scheme, we only ignore the change in probability ratio when it would make the objective improve,
                        #  and we include it when it makes the objective worse."
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
