import tensorflow as tf
from .model import Generator, Encoder, Discriminator
import numpy as np
from .util import Gaussian_sampler, Multiome_loader_TI
import dateutil.tz
import datetime
import sys
import copy
import os
import json
#tf.keras.utils.set_random_seed(123)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

class scMTG(object):
    """scMTG model for clustering.
    """
    def __init__(self, params):
        super(scMTG, self).__init__()
        self.params = params
        self.g24_net = Generator(input_dim=params['pca_dim']+params['z_dim'],z_dim = params['z_dim'], 
            output_dim = params['pca_dim'],model_name='g24_net',nb_layers=8, nb_units=256, concat_every_fcl=False)
        self.d4_net = Discriminator(input_dim=params['pca_dim'],model_name='d4_net',nb_layers=3,nb_units=128)

        self.g46_net = Generator(input_dim=params['pca_dim']+params['z_dim'],z_dim = params['z_dim'], 
            output_dim = params['pca_dim'],model_name='g46_net',nb_layers=8, nb_units=256, concat_every_fcl=False)
        self.d6_net = Discriminator(input_dim=params['pca_dim'],model_name='d6_net',nb_layers=3,nb_units=128)

        self.g24_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d4_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)

        self.g46_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d6_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)

        self.z_sampler = Gaussian_sampler(N=20000, mean=np.zeros(params['z_dim']), sd=1)
        self.x_sampler = Multiome_loader_TI()

        self.initilize_nets()
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "checkpoints/%s" % self.timestamp
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "results/%s" % self.timestamp
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        json.dump(params, open( "%s/params.json"%self.save_dir, 'w' ))
        self.f_log = open('%s/log.txt'%self.save_dir,'a+')   
        ckpt = tf.train.Checkpoint(g24_net = self.g24_net,
                                   d4_net = self.d4_net,
                                   g46_net = self.g46_net,
                                   d6_net = self.d6_net,
                                   g24_optimizer = self.g24_optimizer,
                                   d4_optimizer = self.d4_optimizer,
                                   g46_optimizer = self.g46_optimizer,
                                   d6_optimizer = self.d6_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    def get_config(self):
        return {
                "params": self.params,
        }

    def initilize_nets(self, print_summary = True):
        self.g24_net(np.zeros((1, self.params['pca_dim']+self.params['z_dim'])))
        self.d4_net(np.zeros((1, self.params['pca_dim'])))
        self.g46_net(np.zeros((1, self.params['pca_dim']+self.params['z_dim'])))
        self.d6_net(np.zeros((1, self.params['pca_dim'])))
        if print_summary:
            print(self.g24_net.summary())
            print(self.d4_net.summary())
            print(self.g46_net.summary())
            print(self.d6_net.summary())

    @tf.function
    def train_gen24_step(self, data_z, data_x2, data_x4):
        with tf.GradientTape(persistent=True) as gen24_tape:
            data_z_combine = tf.concat([data_z, data_x2], axis=-1)
            data_x4_ = self.g24_net(data_z_combine)
            data_dx4_ = self.d4_net(data_x4_)
            g24_loss_adv = -tf.reduce_mean(data_dx4_)
            g24_displace_loss = tf.reduce_mean((data_x4 - data_x4_)**2)
            g24_loss = g24_loss_adv + self.params['alpha'] * g24_displace_loss
        g24_gradients = gen24_tape.gradient(g24_loss, self.g24_net.trainable_variables)
        # Apply the gradients to the optimizer
        self.g24_optimizer.apply_gradients(zip(g24_gradients, self.g24_net.trainable_variables))
        return g24_loss_adv, g24_displace_loss

    @tf.function
    def train_gen46_step(self, data_z, data_x4, data_x6):
        with tf.GradientTape(persistent=True) as gen46_tape:
            data_z_combine = tf.concat([data_z, data_x4], axis=-1)
            data_x6_ = self.g46_net(data_z_combine)
            data_dx6_ = self.d6_net(data_x6_)
            g46_loss_adv = -tf.reduce_mean(data_dx6_)
            g46_displace_loss = tf.reduce_mean((data_x6 - data_x6_)**2)
            g46_loss = g46_loss_adv + self.params['alpha'] * g46_displace_loss
        g46_gradients = gen46_tape.gradient(g46_loss, self.g46_net.trainable_variables)
        # Apply the gradients to the optimizer
        self.g46_optimizer.apply_gradients(zip(g46_gradients, self.g46_net.trainable_variables))
        return g46_loss_adv, g46_displace_loss

    @tf.function
    def train_disc4_step(self, data_z, data_x2, data_x4):
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc4_tape:
            data_z_combine = tf.concat([data_z, data_x2], axis=-1)
            data_x4_ = self.g24_net(data_z_combine)
            data_dx4_ = self.d4_net(data_x4_)
            data_dx4 = self.d4_net(data_x4)
            dx4_loss = -tf.reduce_mean(data_dx4) + tf.reduce_mean(data_dx4_)
            #gradient penalty for x
            data_x_hat = data_x4*epsilon_x + data_x4_*(1-epsilon_x)
            data_dx_hat = self.d4_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,x_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,)
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d4_loss = dx4_loss + self.params['gamma']*gpx_loss
        
        # Calculate the gradients for generators and discriminators
        d4_gradients = disc4_tape.gradient(d4_loss, self.d4_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d4_optimizer.apply_gradients(zip(d4_gradients, self.d4_net.trainable_variables))
        return d4_loss

    @tf.function
    def train_disc6_step(self, data_z, data_x4, data_x6):
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc6_tape:
            data_z_combine = tf.concat([data_z, data_x4], axis=-1)
            data_x6_ = self.g46_net(data_z_combine)
            data_dx6_ = self.d6_net(data_x6_)
            data_dx6 = self.d6_net(data_x6)
            dx6_loss = -tf.reduce_mean(data_dx6) + tf.reduce_mean(data_dx6_)
            #gradient penalty for x
            data_x_hat = data_x6*epsilon_x + data_x6_*(1-epsilon_x)
            data_dx_hat = self.d6_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,x_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,)
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d6_loss = dx6_loss + self.params['gamma']*gpx_loss
        
        # Calculate the gradients for generators and discriminators
        d6_gradients = disc6_tape.gradient(d6_loss, self.d6_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d6_optimizer.apply_gradients(zip(d6_gradients, self.d6_net.trainable_variables))
        return d6_loss

    def train(self):
        batches_per_eval = 500
        batch_size = self.params['bs']
        for batch_idx in range(self.params['nb_batches']):
            for _ in range(5):
                batch_z = self.z_sampler.train(batch_size)
                batch_x2, batch_x4, batch_x6 = self.x_sampler.get_batch(batch_size) 
                d4_loss = self.train_disc4_step(batch_z, batch_x2, batch_x4)
                d6_loss = self.train_disc6_step(batch_z, batch_x4, batch_x6)
            batch_z = self.z_sampler.train(batch_size)
            batch_x2, batch_x4, batch_x6 = self.x_sampler.get_batch(batch_size) 
            g24_loss_adv, g24_displace_loss = self.train_gen24_step(batch_z, batch_x2, batch_x4)
            g46_loss_adv, g46_displace_loss = self.train_gen46_step(batch_z, batch_x4, batch_x6)
            #update TV_loss
            #tv_loss = self.train_tv_step(self.x_sampler.embeds, self.x_sampler.adj_hexigon_neighbor)
            if batch_idx % batches_per_eval == 0:
                log = "Batch_idx [%d] g24_loss_adv [%.4f] g24_displace_loss [%.4f] g46_loss_adv \
                    [%.4f] g46_displace_loss [%.4f] d4_loss [%.4f] d6_loss [%.4f] " % (batch_idx, g24_loss_adv, g24_displace_loss, 
                        g46_loss_adv, g46_displace_loss, d4_loss, d6_loss)
                print(log)
                self.f_log.write(log+'\n')
                self.evaluate(batch_idx)
        self.f_log.close()

    def evaluate(self,batch_idx, N = 10000):
        batch_z = self.z_sampler.train(N)
        data_x2, data_x4, data_x6 = self.x_sampler.load_all()
        batch_z = self.z_sampler.train(len(data_x2))
        data_x4_ = self.g24_net(np.concatenate([data_x2, batch_z],axis=-1))
        batch_z = self.z_sampler.train(len(data_x4))
        data_x6_ = self.g24_net(np.concatenate([data_x4, batch_z],axis=-1))
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_x4_,data_x6_)