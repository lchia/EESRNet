from .single import single 

from tensorflow import keras

def create_model(args):
    which_model = args['networks']['which_model']
    s = args['networks']['scale']
    c = args['networks']['in_channels']
    out_channels = args['networks']['out_channels']

    if which_model == 'single':
        netname = args['networks']['netname']
        model = single(s, c, args['networks']['num_fea'], args['networks']['m'], out_channels, netname)
 
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))

    return model
