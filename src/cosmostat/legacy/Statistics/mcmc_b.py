# # # # # # # # metropolis hasting for cosmology  # # # # # # # # # # # # # # # # # # # # # #
# # # Mariana Jaber  2019
# # # This part of the program contains the basic implementation of
# # # a Monte Carlo Marcos Chain using Metropolis Hasting
# # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import datetime
import os
import sys
from multiprocessing import Pool, current_process

import numpy as np

MARKOV_CHAIN_MSG = '''[Process {pid}] Executing Markov Chain for {name} with
{steps:d} chain steps.'''

# todo: convergence criteria: R-1 estimator
HEADER_CHAIN = '''--'''
#'''Chain for 6-D space in the format: ( ---)
#for the N steps.

#Acceptance ratio: {ratio}
#'''

HEADER_LIKE_CHAIN = '''---'''
# Value for the log(Likelihood) calculated
# for each step in the 6D space: (w0, wi, q, zt, h, OmegaM)
# for the N steps.
#
# Acceptance ratio: {ratio}
# '''

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

from cosmostat.legacy.Statistics import loglikeSNe, loglikeBAOSNeHz, loglikeBAO, loglikehz
# TODO check the dependence on the cosmological model through the loglike etc

BAO = 0
Hz = 1
SNe = 2
BAOSNeHz = 3

loglike_set = (
    ('BAO', loglikeBAO),
    ('Hz', loglikehz),
    ('SNe', loglikeSNe),
    ('BAO-SNe-Hz', loglikeBAOSNeHz)
)


def markov_chain(steps, step_width, ini_step, which=BAOSNeHz):
    """
    :param steps: number of steps
    :param step_width: (delta_w0, delta_wi, delta_q, delta_zt, delta_h, delta_OmegaM)
    :param ini_step: (w0_ini, wi_ini, q_ini, zt_ini, h_ini, OmegaM_ini)
    :param which: 0:BAO, 1:BAOHz, 2:BAOSNeHz
    :return:
    """

    model_params = ini_step[0], ini_step[1], ini_step[2], ini_step[3]
    cosmo_params = ini_step[4], ini_step[5]

    like_name, loglike = loglike_set[which]

    chain = [ini_step]
    like_chain = [loglike(model_params, cosmo_params)]

    cp = current_process()
    print(MARKOV_CHAIN_MSG.format(pid=cp.pid, name=like_name, steps=steps))

    each = 1000
    accepted = 0
    for i in range(steps):
        my_rand = np.random.normal(0., 1., len(ini_step))
        # my_rand = np.random.random_sample(size=len(ini_step)) - 0.5

        # Uniform random movement centered in the current point contained
        # in a 6-dimensional cube of size given by step_width.
        new_point = chain[i] + step_width * my_rand

        w0p, w1p, w2p, w3p, hp, OmegaMp = new_point
        model_params = w0p, w1p, w2p, w3p
        cosmo_params = hp, OmegaMp

        # Print message every half of the steps
        residual = (i + 1) % each
        if not residual:
            print('[Process {}] Markov Chain first {:d} steps '
                  'completed'.format(cp.pid, i))

        #if not (-2 < w1p < 1):
        #    accept_like = -1E50
         #   accept_prob = 0
          #  like_try = -1e50
        if not (0 < OmegaMp < 1):
            accept_like = -1E50
            accept_prob = 0
            like_try = -1e50
        elif not (0 < hp < 1):
            accept_like = -1E50
            accept_prob = 0
            like_try = -1e50
       # elif not (10 > w2p > 0):
       #     accept_like = -1E50
        #    accept_prob = 0
        #    like_try = -1e50
        #elif not (3 > w3p > 0):
        #    accept_like = -1E50
        #    accept_prob = 0
         #   like_try = -1e50
        else:
            like_try = loglike(model_params, cosmo_params)

            if np.isnan(like_try):
                print('[Process {}] Something weird just '
                      'happened :('.format(cp.pid))
                accept_like = -1E50
                accept_prob = 0
            elif like_try > like_chain[i]:
                accept_like = 0
                accept_prob = 1
            else:
                accept_like = like_try - like_chain[i]
                accept_prob = np.exp(like_try - like_chain[i])

        # if accept_prob >= np.random.uniform(0., 1.):
        rand_uni = np.random.uniform(0., 1.)
        if accept_like >= np.log(rand_uni):
            chain.append(new_point)
            like_chain.append(like_try)
            accepted += 1
        else:
            chain.append(chain[i])
            like_chain.append(like_chain[i])

    chain = np.array(chain)
    like_chain = np.array(like_chain)
    accept_ratio = float(accepted) / float(steps)
    # chain_array = np.array(chain_array)
    # full_array = np.array(chain, like_chain)

    print('[Process {}] Markov chain acceptance '
          'ratio: {}'.format(cp.pid, accept_ratio))

    return chain, like_chain, accept_ratio


# todo: add prior to the range in parameter space for w0, w1, w2, w3 (necessary?)



def markov_chain_kernel(spec):
    """

    :param spec:
    :return:
    """
    steps, width, init_step, which = spec
    return markov_chain(steps, width, init_step, which)


def parallel_markov_chain(markov_chains_spec, processes=None):
    """

    :param markov_chains_spec:
    :param processes:
    :return:
    """
    # data_values = data_array.shape[0]

    with Pool(processes=processes) as pool:
        # The data accepted by the map method must be an iterable, like
        # a list, a tuple or a numpy array. The function is applied over
        # each element of the iterable. The result is another list with
        # the values returned by the function after being evaluated.
        #
        # [a, b, c, ...]  -> [f(a), f(b), f(c), ...]
        #
        # Here we use the imap method, so we need to create a list to
        # gather the results.
        results = []
        pool_imap = pool.imap(markov_chain_kernel, markov_chains_spec)
        # progress = 0
        for result in pool_imap:
            results.append(result)

        return results


def export_chains(base_path, chains_specs, chains_results):
    """

    :param base_path:
    :param chains_specs:
    :param chains_results:
    :return:
    """
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d-%H%M')

    chains_data_zip = zip(chains_specs, chains_results)
    for index, data in enumerate(chains_data_zip):
        chain_spec, chain_result = data
        steps, width, ini_step, which = chain_spec
        chain, like_chain, accept_ratio = chain_result

        like_name, _ = loglike_set[which]
        steps_str = str(steps)

        # Create a subdirectory for the model and the date of the
        # execution.
        chain_model_path = os.path.abspath(
                os.path.join(base_path, like_name, date_str)
        )
        os.makedirs(chain_model_path, exist_ok=True)

        chain_file = 'chain-N_{:s}[{}].txt'.format(steps_str, index)
        likechain_file = 'likechain-N_{:s}[{}].txt'.format(steps_str, index)
        chain_header = HEADER_CHAIN.format(ratio=accept_ratio)
        like_chain_header = HEADER_LIKE_CHAIN.format(ratio=accept_ratio)

        np.savetxt(os.path.join(chain_model_path, chain_file),
                   chain, header=chain_header)
        np.savetxt(os.path.join(chain_model_path, likechain_file),
                   like_chain, header=like_chain_header)
        # np.savetxt(os.path.join(current_dir, 'chains', filetxtchain), chain,
        #           header=HEADERchain)
