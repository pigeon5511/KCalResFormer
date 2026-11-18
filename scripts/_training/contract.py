"""Validate the shared A/C fitting contract before any data are read."""


def validate(config):
    if config['dataset_version'] != 'formal_dmrs_ber_20261023_v1':
        raise ValueError('Use the declared formal DM-RS dataset')
    if config['shared_source_layouts'] != ['A', 'C'] or config['specialist_layouts']:
        raise ValueError('This entry trains one shared A/C model')
    if set(config['maximum_epochs']) != {'exact_klmmse_residual'}:
        raise ValueError('Only the exact-K model can be trained here')
    if config['component_kwargs']['grid_shape'] != [120, 14]:
        raise ValueError('The released architecture requires the fixed 120x14 grid')
    if config['loss'] != 'mean_per_observation_complex_MSE':
        raise ValueError('The fitting loss must be complex MSE')
    if config['alpha']['split'] != 'validation_alpha':
        raise ValueError('Fit alpha on the separate source alpha pool')
    if config['batch_size'] < 1 or config['early_stopping_patience'] < 1:
        raise ValueError('Positive batch size and patience are required')
    return dict(status='SHARED_EXACT_AC_CONTRACT', training_runs=len(config['training_seeds']))
