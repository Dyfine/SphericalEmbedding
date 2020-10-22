import logging
from config import get_config
from learner import metric_learner

if __name__ == '__main__':
    conf = get_config()

    test_path = 'work_space/' + conf.test_sop_model + '/models'
    logging.info(test_path)
    test_name = 'SOP'

    conf.use_dataset = test_name
    learner = metric_learner(conf, inference=True)
    learner.load_state(conf, resume_path=test_path)

    nmi, f1, recall_ks = learner.test_sop_complete(conf)

    ks_dict = dict()
    ks_dict['CUB'] = [1, 2, 4, 8, 16, 32]
    ks_dict['Cars'] = [1, 2, 4, 8, 16, 32]
    ks_dict['SOP'] = [1, 10, 100, 1000, 10000]
    ks_dict['Inshop'] = [1, 10, 20, 30, 40, 50]
    k_s = ks_dict[test_name]

    logging.info(f'nmi: {nmi}')
    logging.info(f'f1: {f1}')
    for i in range(len(recall_ks)):
        logging.info(f'R{k_s[i]} {recall_ks[i]}')
