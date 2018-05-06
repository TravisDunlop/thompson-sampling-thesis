
def test_policy(env, pol, num_episodes, results, reset_kwargs):
    '''Takes python gym style environment and tests
        given policy (pol) num_episodes amount of times
        then appends to results'''
    keys = sorted(reset_kwargs.keys())
    for episode in range(num_episodes):
        total_cost, cost = 0, 0
        env.reset(**reset_kwargs)
        pol.reset(env)
        done = False
        observation = None

        while not done:
            action = pol.act(observation)
            total_cost += cost

            observation, cost, done, info = env.step(action)
            pol.update(observation, cost)

        result = [env.get_name(), pol.get_name()]
        for key in keys: result.append(reset_kwargs[key])
        result.append(total_cost / env.num_steps)

        results.append(result)
