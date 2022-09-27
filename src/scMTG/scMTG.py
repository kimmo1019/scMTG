import tensorflow as tf
from .model import Generator, Encoder, Discriminator
import numpy as np
from .util import Gaussian_sampler, Multiome_loader_TI, Mixture_sampler,ARC_TS_Sampler
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
        self.g24_net = Generator(input_dim=2+params['z_dim'],z_dim = params['z_dim'], 
            output_dim = params['pca_dim'],model_name='g24_net',nb_layers=8, nb_units=256, concat_every_fcl=False)
        self.d4_net = Discriminator(input_dim=params['pca_dim'],model_name='d4_net',nb_layers=3,nb_units=128)

        self.g46_net = Generator(input_dim=2+params['z_dim'],z_dim = params['z_dim'], 
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
        self.g24_net(np.zeros((1, 2+self.params['z_dim'])))
        self.d4_net(np.zeros((1, self.params['pca_dim'])))
        self.g46_net(np.zeros((1, 2+self.params['z_dim'])))
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
                #d4_loss = self.train_disc4_step(batch_z, batch_x2, batch_x4)
                #d6_loss = self.train_disc6_step(batch_z, batch_x4, batch_x6)
                d4_loss = self.train_disc4_step(batch_z, np.random.normal(size=(batch_size, 2)).astype('float32'), batch_x4)
                d6_loss = self.train_disc6_step(batch_z, np.random.normal(size=(batch_size, 2)).astype('float32'), batch_x6)
            batch_z = self.z_sampler.train(batch_size)
            batch_x2, batch_x4, batch_x6 = self.x_sampler.get_batch(batch_size) 
            #g24_loss_adv, g24_displace_loss = self.train_gen24_step(batch_z, batch_x2, batch_x4)
            #g46_loss_adv, g46_displace_loss = self.train_gen46_step(batch_z, batch_x4, batch_x6)
            g24_loss_adv, g24_displace_loss = self.train_gen24_step(batch_z, np.random.normal(size=(batch_size, 2)).astype('float32'), batch_x4)
            g46_loss_adv, g46_displace_loss = self.train_gen46_step(batch_z, np.random.normal(size=(batch_size, 2)).astype('float32'), batch_x6)
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
        data_x2 = np.random.normal(size=(len(data_x2), 2))
        data_x4_ = self.g24_net(np.concatenate([data_x2, batch_z],axis=-1))
        batch_z = self.z_sampler.train(len(data_x4))
        data_x4 = np.random.normal(size=(len(data_x4), 2))
        data_x6_ = self.g24_net(np.concatenate([data_x4, batch_z],axis=-1))
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_x4_,data_x6_)



class scDEC(object):
    """scDEC model for clustering.
    """
    def __init__(self, params):
        super(scDEC, self).__init__()
        self.params = params
        self.g_net = Generator(input_dim=params['z_dim']+params['nb_classes'],model_name='g_net',z_dim = params['z_dim'], output_dim = params['x_dim'],nb_layers=10, nb_units=512, concat_every_fcl=False)
        self.h_net = Encoder(input_dim=params['x_dim'], output_dim = params['z_dim']+params['nb_classes'],feat_dim=params['z_dim'],nb_layers=10,nb_units=256)
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',nb_layers=2,nb_units=256)
        self.dx_net = Discriminator(input_dim=params['x_dim']+params['nb_classes'],model_name='dx_net',nb_layers=2,nb_units=256)
        self.g_h_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Mixture_sampler(nb_classes=params['nb_classes'],N=10000,dim=params['z_dim'],sd=1)
        self.x_sampler = ARC_TS_Sampler(start_t=params['start_t'], prior_dim=params['nb_classes'])
        self.initilize_nets()
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "checkpoints/%s" % self.timestamp
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "results/%s" % self.timestamp
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   
        ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   h_net = self.h_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_h_optimizer = self.g_h_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim']+self.params['nb_classes'])))
        self.h_net(np.zeros((1, self.params['x_dim'])))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dx_net(np.zeros((1, self.params['x_dim']+self.params['nb_classes'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.dx_net.summary())    

    @tf.function
    def train_gen_step(self, data_z, data_z_onehot, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
                Fourth item: 0: update generators, 1: update discriminators
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            data_x = tf.cast(data_x, tf.float32)
            data_z_combine = tf.concat([data_z, data_z_onehot], axis=-1)
            #print('1',data_z, data_z_onehot, data_x, data_z_combine)
            data_x_ = self.g_net(data_z_combine)
            data_z_latent_, data_z_onehot_ = self.h_net(data_x)
            data_z_ = data_z_latent_[:,:self.params['z_dim']]
            data_z_logits_ = data_z_latent_[:,self.params['z_dim']:]
            
            data_z_latent__, data_z_onehot__ = self.h_net(data_x_)
            data_z__ = data_z_latent__[:,:self.params['z_dim']]
            data_z_logits__ = data_z_latent__[:,self.params['z_dim']:]
            
            data_z_combine_ = tf.concat([data_z_, data_z_onehot_], axis=-1)
            data_x__ = self.g_net(data_z_combine_)
            
            data_dx_ = self.dx_net(tf.concat([data_x_, data_z_onehot], axis=-1))
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            #CE_loss_z = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_z_onehot, logits=data_z_logits__))
            CE_loss_z = tf.reduce_mean((data_z_onehot - data_z_logits__)**2)
            
            g_loss_adv = -tf.reduce_mean(data_dx_)
            h_loss_adv = -tf.reduce_mean(data_dz_)
            g_h_loss = g_loss_adv+h_loss_adv+self.params['alpha']*(l2_loss_x + l2_loss_z)+ \
                        self.params['beta']*CE_loss_z
            
        # Calculate the gradients for generators and discriminators
        g_h_gradients = gen_tape.gradient(g_h_loss, self.g_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_h_optimizer.apply_gradients(zip(g_h_gradients, self.g_net.trainable_variables+self.h_net.trainable_variables))
        return g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss

    @tf.function
    def train_disc_step(self, data_z, data_z_onehot, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
                Fourth item: 0: update generators, 1: update discriminators
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            data_x = tf.cast(data_x, tf.float32)
            data_z_combine = tf.concat([data_z, data_z_onehot], axis=-1)
            #print('1',data_z, data_z_onehot, data_x, data_z_combine)
            data_x_ = self.g_net(data_z_combine)
            data_z_latent_, data_z_onehot_ = self.h_net(data_x)
            data_z_ = data_z_latent_[:,:self.params['z_dim']]
            
            data_dx_ = self.dx_net(tf.concat([data_x_, data_z_onehot], axis=-1))
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(tf.concat([data_x, data_z_onehot], axis=-1))
            data_dz = self.dz_net(data_z)
            
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            dx_loss = -tf.reduce_mean(data_dx) + tf.reduce_mean(data_dx_)

            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))

            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(tf.concat([data_x_hat, data_z_onehot], axis=-1))
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,x_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss = dx_loss + dz_loss + self.params['gamma']*(gpz_loss+gpx_loss)
        
        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dx_loss, dz_loss, d_loss

    def train(self): 
        batches_per_eval = 500
        batch_size = self.params['bs']
        for batch_idx in range(self.params['nb_batches']):
            for _ in range(5):
                batch_z, _ = self.z_sampler.train(batch_size)
                batch_x, batch_onehot = self.x_sampler.get_batch(batch_size) 
                dx_loss, dz_loss, d_loss = self.train_disc_step(batch_z, batch_onehot, batch_x)
            batch_z, _ = self.z_sampler.train(batch_size)
            batch_x, batch_onehot = self.x_sampler.get_batch(batch_size)            
            #print(batch_x.shape, batch_onehot.shape)
            g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss = self.train_gen_step(batch_z, batch_onehot,batch_x)
            if batch_idx % batches_per_eval == 0:
                print(batch_onehot.shape,np.max(batch_onehot),np.min(batch_onehot),batch_onehot[0])
                #print(batch_idx, g_loss_adv, h_loss_adv, CE_loss_z, l2_loss_z, l2_loss_x, g_h_loss, dz_loss, dx_loss, d_loss)
                print("Batch_idx [%d] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss_z [%.4f] l2_loss_z [%.4f] l2_loss_x [%.4f] g_h_loss [%.4f] dz_loss [%.4f] dx_loss [%.4f] d_loss [%.4f]" % 
                      (batch_idx, g_loss_adv, h_loss_adv, CE_loss_z, l2_loss_z, l2_loss_x, g_h_loss, dz_loss, dx_loss, d_loss))
                self.evaluate(batch_idx)
                #ckpt_save_path = self.ckpt_manager.save()
                #print ('Saving checkpoint for epoch {} at {}'.format(batch_idx,ckpt_save_path))

    def evaluate(self,batch_idx):
        data_x, label_ts = self.x_sampler.load_all()
        batch_z, _ = self.z_sampler.train(len(data_x))
        #print(data_x.shape,label_ts.shape,batch_z.shape)
        #sys.exit()
        data_gen = self.g_net(np.concatenate([batch_z,label_ts],axis=1))
        np.save('{}/data_pre_{}.npy'.format(self.save_dir, batch_idx),data_gen)