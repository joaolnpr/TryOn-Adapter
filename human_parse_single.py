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
    N_CLASSES = 20 # 20 classes for CIHP_PGN
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

def run_cihp_pgn_mask(person_img_path, output_mask_path):
    import uuid
    import shutil
    # 1. Create temp dirs
    temp_id = str(uuid.uuid4())
    temp_input_dir = f"/tmp/cihp_input_{temp_id}"
    temp_output_dir = f"/tmp/cihp_output_{temp_id}"
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    # 2. Copy person image to temp input dir
    base_name = os.path.basename(person_img_path)
    temp_input_img = os.path.join(temp_input_dir, base_name)
    shutil.copy2(person_img_path, temp_input_img)
    # 3. Run inf_pgn.py
    cihp_pgn_dir = os.path.expanduser('~/CIHP_PGN')
    cihp_pgn_script = os.path.join(cihp_pgn_dir, 'inf_pgn.py')
    subprocess.run([
        'conda', 'run', '-n', 'cihp_pgn', 'python', cihp_pgn_script,
        '-i', temp_input_dir,
        '-o', temp_output_dir
    ], check=True, cwd=cihp_pgn_dir)
    # 4. Move output mask to expected location
    mask_name = os.path.splitext(base_name)[0] + '.png'
    temp_mask_path = os.path.join(temp_output_dir, 'cihp_parsing_maps', mask_name)
    if not os.path.exists(temp_mask_path):
        raise FileNotFoundError(f"Expected mask not found: {temp_mask_path}")
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    shutil.move(temp_mask_path, output_mask_path)
    # 5. Cleanup
    shutil.rmtree(temp_input_dir)
    shutil.rmtree(temp_output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output mask')
    args = parser.parse_args()
    if run_single_image is not None:
        run_single_image(args.image, args.output)
    else:
        run_cihp_pgn_mask(args.image, args.output)

if __name__ == '__main__':
    main() 