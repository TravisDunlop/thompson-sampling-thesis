

def estimate_regret(loss_matrix, agent):
    agent.reset()
    loss = 0

    for loss_vector in loss_matrix:
        action = agent.act()
        loss += loss_vector[action]
        agent.update(loss_vector)

    regret = loss - loss_matrix
