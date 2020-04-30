from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time 

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm, get_ckpt_inf

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('which_datatest', 'lfw', 'path to dataset test') # 'lfw_align_112/lfw' 'agedb_align_112/agedb_30' 'cfp_align_112/cfp_fp'
flags.DEFINE_string('v_tf', '2', 'which version of tensorflow to use : 1.2.0 or 1.13.1')

def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=60,
                            timeout=None):
    print('Waiting for new checkpoint at {}'.format(checkpoint_dir))
    #stop_time = time.time() + timeout if timeout is not None else None
    while True:
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or checkpoint_path == last_checkpoint:
            #if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
                #return None
            time.sleep(seconds_to_sleep)
        else:
            print('Found new checkpoint at {}'.format(checkpoint_path))
            return checkpoint_path

if __name__ == '__main__':

    cfg_path = './configs/arc_res100_fit.yaml'
    which_datatest = 'agedb'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    logger = tf.get_logger()


    #logger.disabled = True
  
    #logger.setLevel(logging.FATAL)
   
    set_memory_growth()
  
    cfg = load_yaml(cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                            backbone_type=cfg['backbone_type'],
                            training=False)


    if which_datatest == 'lfw':
        dataset_test_path = 'lfw_align_112/lfw'
    elif which_datatest == 'agedb' :
        dataset_test_path = 'agedb_align_112/agedb_30'
    elif which_datatest == 'cfp' :
        dataset_test_path = 'cfp_align_112/cfp_fp'
    else : 
        dataset_test_path = which_datatest

    print("[*] Loading dataset test {}...".format(which_datatest))

    dataset_test, dataset_test_issame = \
            get_val_data(cfg['test_dataset'],dataset_test_path)

    test_summary_writer = tf.summary.create_file_writer(cfg['logs_dir'] + cfg['sub_name'] + '/test_'+which_datatest)

    checkpoint_dir = os.path.join(cfg['checkpoints_dir'],cfg['sub_name'])
    ckpt_path = None

    steps_per_epoch = cfg['num_samples'] // cfg['batch_size']

    while True:
        
        new_ckpt_path = wait_for_new_checkpoint(
            checkpoint_dir, ckpt_path)
        ckpt_path = new_ckpt_path
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)

        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        
        print("[*] Perform Evaluation ...")
        acc, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, dataset_test, dataset_test_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    Step {} , acc {:.1f}, th: {:.2f}".format(steps, acc*100 , best_th))

        with test_summary_writer.as_default():
                tf.summary.scalar('best_threshold', best_th, step=steps)
                tf.summary.scalar('accuracy', acc, step=steps)
