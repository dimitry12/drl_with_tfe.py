import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import random
import gym

import time

RANDOM_SEED = 42

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MLP_UNITS = 16
MLP_EXTRA_LAYERS = 3

TRANSITIONS_IN_EXPERIENCE_BUFFER = 1024
GRADIENT_LEARNING_BATCH_SIZE = 32
assert TRANSITIONS_IN_EXPERIENCE_BUFFER % GRADIENT_LEARNING_BATCH_SIZE == 0
EPOCHS_PER_UPDATE = 4

CLIP_RANGE = .2
GAMMA = .99
ADVANTAGE_LAMBDA = .97
MAX_GRAD_NORM = .5
VALUE_LOSS_WEIGHT = .25
LR = 3e-4

TOTAL_ENV_STEPS = 2e7

import datetime
TF_LOGS_DIR = './tf-logs/' + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")


class P_and_V_Model(tf.keras.Model):
    def __init__(self, *, classes, mlp_extra_layers=MLP_EXTRA_LAYERS, mlp_units=MLP_UNITS, heads='pv'):
        super(P_and_V_Model, self).__init__(name="P_and_V_Model")
        # interestingly, model *must* be named if we use named layers
        # (Keras 2.1.6-tf)
        self._outputs = heads
        self.dense = tf.keras.layers.Dense(mlp_units, activation='tanh')
        self.hidden_denses = []
        for _ in range(mlp_extra_layers):
            self.hidden_denses.append(tf.keras.layers.Dense(
                mlp_units, activation='tanh'))
        self.p_outputs = tf.keras.layers.Dense(
            classes, activation=None, name='p_logits')
        self.v_output = tf.keras.layers.Dense(
            1, activation=None, name='v_logit')

    def call(self, inputs):
        shared_latent = self.dense(inputs)
        for l in self.hidden_denses:
            shared_latent = l(shared_latent)

        p_logits = self.p_outputs(shared_latent)
        v_logit = self.v_output(shared_latent)

        if self._outputs == 'p':
            return p_logits
        elif self._outputs == 'v':
            return v_logit
        else:
            return p_logits, v_logit


def main():
    env = gym.make('CartPole-v0')
    writer = tf.contrib.summary.create_file_writer(TF_LOGS_DIR)
    rl_writer = tf.contrib.summary.create_file_writer(TF_LOGS_DIR + 'rl')
    writer.set_as_default()

    # only works for Box
    observations_space_dim_count = env.observation_space.shape[0]
    action_space_cardinality = env.action_space.n  # only works for Discrete

    pv_model = P_and_V_Model(classes=action_space_cardinality)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

    learner_updates = int(TOTAL_ENV_STEPS // TRANSITIONS_IN_EXPERIENCE_BUFFER)

    for learner_update in range(1, learner_updates+1):
        with rl_writer.as_default(), tf.contrib.summary.always_record_summaries():
            current_update_started_at = time.time()
            episodes_in_current_update = 0
            total_rewards_in_current_update = []
            steps_in_current_update = []

            observations, rewards, taken_actions, predicted_values, neg_log_p_ac_s, episode_dones = [], [], [], [], [], []

            observation: np.ndarray = env.reset()
            episode_done: bool = False
            reward: float
            steps_in_current_episode = 0
            total_reward_in_current_episode = 0

            def vector_to_tf_constant(x): return tf.constant(
                x, dtype=tf.keras.backend.floatx(), shape=(1, len(x)))

            for _ in range(TRANSITIONS_IN_EXPERIENCE_BUFFER):
                p_logits, v_logit = pv_model(
                    vector_to_tf_constant(observation))
                p_distribution = tf.distributions.Categorical(logits=p_logits)
                action = p_distribution.sample()
                tf.contrib.summary.hist(
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
                env.render()

                if episode_done:
                    observation = env.reset()
                    episodes_in_current_update += 1
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
            returns = np.zeros_like(rewards, dtype=np.float32)
            # tail_of_gae  is lambda-discounted sum of per-slice surprises. We learn from surprises!
            # tail_of_returns is gamma-discounted sum of per-transition rewards (policy-value)
            for t in reversed(range(TRANSITIONS_IN_EXPERIENCE_BUFFER)):
                is_this_last_slice = (t == TRANSITIONS_IN_EXPERIENCE_BUFFER-1)

                if is_this_last_slice:
                    did_this_slice_ended_episode = episode_done
                    next_slice_v_logit = last_v_logit
                    tail_of_gae = 0  # no heuristic for gamma-lambda-discounted sum of residuals
                    tail_of_returns = GAMMA*last_v_logit  # estimate of the tail
                else:
                    did_this_slice_ended_episode = episode_dones[t+1]
                    next_slice_v_logit = predicted_values[t+1]

                if did_this_slice_ended_episode:
                    tail_of_gae = 0  # zero-out the tail of generalized advantage estimation
                    tail_of_returns = 0
                    td_residual = rewards[t] - predicted_values[t]
                else:
                    td_residual = rewards[t] + GAMMA * \
                        next_slice_v_logit - predicted_values[t]

                gae_s[t] = td_residual + GAMMA * ADVANTAGE_LAMBDA * tail_of_gae
                returns[t] = rewards[t] + GAMMA*tail_of_returns
                tail_of_gae = gae_s[t]
                tail_of_returns = returns[t]

            observations = np.asarray(observations, dtype=np.float32)
            taken_actions = np.asarray(taken_actions, dtype=np.int64)
            predicted_values = np.asarray(predicted_values, dtype=np.float32)
            neg_log_p_ac_s = np.asarray(neg_log_p_ac_s, dtype=np.float32)

            tf.contrib.summary.hist('GAEs', gae_s)

        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            dataset = tf.data.Dataset.from_tensor_slices(
                (observations, returns, taken_actions, predicted_values, neg_log_p_ac_s, gae_s))
            dataset = dataset.shuffle(
                EPOCHS_PER_UPDATE*TRANSITIONS_IN_EXPERIENCE_BUFFER).batch(GRADIENT_LEARNING_BATCH_SIZE)
            v_losses = []
            for _ in range(EPOCHS_PER_UPDATE):
                for batch_number, dataset_batch in enumerate(dataset):
                    observations_batch, returns_batch, old_taken_actions_batch, old_predicted_values_batch, old_neg_log_p_ac_s_batch, advantages_batch = dataset_batch

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
                                ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                        # effectively:
                        # - clip ratio 1.0+CLIP_RANGE when advantage is positive
                        # - clip ratio 1.0-CLIP_RANGE when advantage is negative
                        # (paper's wording is misleading, but graphs are correct)
                        pg_loss = tf.reduce_mean(
                            tf.minimum(pg_losses, pg_losses2))
                        tf.contrib.summary.scalar(
                            'batch_average_CLIP', pg_loss)
                        pg_loss = -pg_loss  # TF's optimizers minimize

                        v_pred_clipped = old_predicted_values_batch + \
                            tf.clip_by_value(
                                train_v_logit - old_predicted_values_batch, - CLIP_RANGE, CLIP_RANGE)
                        v_f_losses1 = tf.square(train_v_logit - returns_batch)
                        tf.contrib.summary.scalar('batch_average_unclipped_v_loss',
                                                  tf.reduce_mean(v_f_losses1))
                        v_f_losses2 = tf.square(v_pred_clipped - returns_batch)
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
                        # - yet keep it within CLIP_RANGE of the old predicted value
                        v_f_loss = .5 * \
                            tf.reduce_mean(tf.maximum(
                                v_f_losses1, v_f_losses2))
                        tf.contrib.summary.scalar('batch_average_clipped_v_loss',
                                                  tf.reduce_mean(v_f_loss))
                        v_losses.append(v_f_loss.numpy())

                        # Total loss
                        loss = pg_loss + v_f_loss * VALUE_LOSS_WEIGHT

                    grads = tape.gradient(loss, pv_model.variables)
                    grads, _grad_norm = tf.clip_by_global_norm(
                        grads, MAX_GRAD_NORM)
                    optimizer.apply_gradients(zip(grads, pv_model.variables),
                                              global_step=tf.train.get_or_create_global_step())

            fps = int(TRANSITIONS_IN_EXPERIENCE_BUFFER /
                      (time.time() - current_update_started_at))

            print("update: ", learner_update)
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
                  TRANSITIONS_IN_EXPERIENCE_BUFFER)
            print("average of batch-averaged V losses",
                  np.array(v_losses).mean())
            tf.contrib.summary.scalar(
                'total_frames', tf.constant(learner_update*TRANSITIONS_IN_EXPERIENCE_BUFFER))
            print("fps", fps)


if __name__ == '__main__':
    main()
