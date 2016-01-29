import yaml
from scn_victor2 import run_model
import tensorflow as tf


def main():
    with open('scn_victor2/eval.yaml', 'r') as f:
        conf = yaml.load(f)
    
    results = {}
    path_tes = ['data/Set5', 'data/Set14', 'data/BSD100', 'data/Urban100']        
    srs = [2, 3, 4]
    for sr in srs:
        # Train the model for specified super-resolution factor `sr`
        results[sr] = {}
        conf['sr'] = sr
        run_model.train(conf)
        # Get the checkpoint
        ckpt = tf.train.get_checkpoint_state(conf['path_tmp']).model_checkpoint_path
        # Evaluate on all testsets
        for path_te in path_tes:
            te_name = path_te.split('/')[-1]
            conf['path_te'] = path_te
            psnr, bl_psnr = run_model.eval_te(conf, ckpt)
            results[sr][te_name] = (psnr, bl_psnr)
            
    with open('results.txt', 'w') as f:
        for sr in srs:
            for path_te in path_tes:
                te_name = path_te.split('/')[-1]
                f.write('sr: %d %s psnr: %.3f bl_psnr: %.3f\n' % (sr, te_name,
                    results[sr][te_name][0], results[sr][te_name][1]))


if __name__ == '__main__':
    main()
