import argparse
import os
import sys
import numpy as np
from PIL import Image
import subprocess

# Try to import from CIHP_PGN if available in PYTHONPATH
try:
    sys.path.append(os.path.expanduser('~/CIHP_PGN'))
    from utils import decode_labels, PGNModel, load
    import tensorflow as tf
    N_CLASSES = 20
    RESTORE_FROM = os.path.expanduser('~/CIHP_PGN/checkpoint/CIHP_pgn')
    def run_single_image(image_path, output_path):
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        img_input = img_np[np.newaxis, ...]
        tf.reset_default_graph()
        image_ph = tf.placeholder(tf.uint8, shape=[1, h, w, 3])
        image_float = tf.cast(image_ph, tf.float32)
        image_batch = image_float
        with tf.variable_scope('', reuse=False):
            net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
        parsing_out = net.layers['parsing_fc']
        parsing_out = tf.argmax(parsing_out, axis=3)
        parsing_out = tf.cast(parsing_out, tf.uint8)
        restore_var = tf.global_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        loader = tf.compat.v1.train.Saver(var_list=restore_var)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if RESTORE_FROM is not None:
            if load(loader, sess, RESTORE_FROM):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        parsing = sess.run(parsing_out, feed_dict={image_ph: img_input})
        parsing_img = Image.fromarray(parsing[0])
        parsing_img.save(output_path)
        print(f"Saved mask to {output_path}")
        sess.close()
except Exception as e:
    print("Could not import CIHP_PGN modules directly, will fallback to subprocess.")
    run_single_image = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output mask')
    args = parser.parse_args()
    if run_single_image is not None:
        run_single_image(args.image, args.output)
    else:
        # Fallback: call CIHP_PGN/test_pgn.py as a subprocess, but this will fail if the top-level bug is present
        cihp_pgn_script = os.path.expanduser('~/CIHP_PGN/test_pgn.py')
        subprocess.run([
            'conda', 'run', '-n', 'cihp_pgn', 'python', cihp_pgn_script,
            '--image', args.image,
            '--output', args.output
        ], check=True)

if __name__ == '__main__':
    main() 