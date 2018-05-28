import numpy as np
import pandas as pd

def test_policy(env, pol, num_episodes, results, reset_kwargs):
    '''Takes python gym style environment and tests
        given policy (pol) num_episodes amount of times.
        results is a list to which results of test will be appended'''
    for episode in range(num_episodes):
        env.reset(**reset_kwargs)
        pol.reset(env)
        done = False

        while not done:
            advice = env.get_advice()
            action = pol.act(advice)
            observation, cost, done, info = env.step(action)
            pol.update(advice, observation)

        result = [env.get_name(), pol.get_name(), env.num_experts, env.num_steps]
        result.append(env.regret_per_step())

        results.append(result)

def test_and_save(env, pol, folder, num_experts, min_step, max_step, num_simulations, append = False):
    '''tests policy, environment combination num_simulations amount of times and
        saves the result to a file in the specified folder.'''
    steps = np.random.randint(10, 1000, 1000)
    results = []

    for num_steps in steps:
        reset_kwargs = { 'num_experts' : num_experts, 'num_steps' : num_steps }
        test_policy(env, pol, 1, results, reset_kwargs)

    file_name = folder + env.get_name() + '_' + pol.get_name() + '.csv'

    save_list(results, file_name, append)


def save_list(lst, file_name, append = False, columns = ['environment', 'policy', 'experts', 'steps', 'regret_per_step']):
    df = pd.DataFrame(lst, columns = columns)
    if not append:
        df.to_csv(file_name, index = False)
    else:
        df.to_csv(file_name, append = True, index = False, header = False)
