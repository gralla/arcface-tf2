from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.dataset as dataset


flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('v_tf', '2', 'which version of tensorflow to use : 1.2.0 or 1.13.1')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

def step_decay(epoch):
    initial_lr = 0.1
    if epoch < 9 :
        lrate = 0.1
    elif epoch >= 9 and epoch < 13:
        lrate = 0.1/10
    elif epoch >=13 :
        lrate = 0.1/100
    return lrate

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    if FLAGS.v_tf == '2' :
        set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True,
                         version_tensorflow=FLAGS.v_tf)
    model.summary(line_length=80)

    if cfg['train_dataset']:
        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])


    ckpt_path = tf.train.latest_checkpoint(cfg['checkpoints_dir'] + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    if FLAGS.mode == 'eager_tf':

        learning_rate = 0.1

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
        loss_fn = SoftmaxLoss()

        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training

        summary_writer = tf.summary.create_file_writer(
            cfg['logs_dir'] + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            inputs, labels = next(train_dataset)

            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)
                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist)
                total_loss = pred_loss + reg_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.2f}"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      optimizer.lr.numpy()))

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                model.save_weights('{}{}/e_{}_b_{}.ckpt'.format(
                    cfg['checkpoints_dir'],cfg['sub_name'], epochs, steps % steps_per_epoch))

            steps += 1
            epochs = steps // steps_per_epoch + 1


    else:

        learning_rate = 0.1

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
        loss_fn = SoftmaxLoss()

        model.compile(optimizer=optimizer, loss=loss_fn)

        mc_callback = ModelCheckpoint(
            cfg['checkpoints_dir'] + cfg['sub_name'] + '/e_{epoch}_b_{batch}.ckpt',
            save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
            save_weights_only=True)
        tb_callback = TensorBoard(log_dir=cfg['logs_dir'] + cfg['sub_name'],
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        #lrate = LearningRateScheduler(step_decay)
        #callbacks = [mc_callback, tb_callback, lrate]

        callbacks = [mc_callback, tb_callback]

        model.fit(train_dataset,
                  epochs=cfg['epochs'],
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=epochs - 1)

    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
