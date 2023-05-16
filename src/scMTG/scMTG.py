import tensorflow as tf
from .model import Generator, Encoder, Discriminator, BaseFullyConnectedNet
import numpy as np
from .util import Gaussian_sampler, Multiome_loader_TI, Mixture_sampler,ARC_TS_Sampler, Base_sampler
import dateutil.tz
import datetime
import sys
import copy
import os
import json
#tf.keras.utils.set_random_seed(123)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
        batches_per_eval = 2000
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

    def evaluate(self, batch_idx, nb_per_sample = 1000):
        data_x, label_ts = self.x_sampler.load_all()
        batch_z, _ = self.z_sampler.train(len(data_x))
        data_gen = self.g_net(np.concatenate([batch_z, label_ts],axis=1))
        np.save('{}/data_pre_{}.npy'.format(self.save_dir, batch_idx),data_gen)
        prior = self.x_sampler.prior
        data_gen = []
        for i in range(len(prior)):
            batch_z, _ = self.z_sampler.train(nb_per_sample)
            data_gen.append(self.g_net(np.concatenate([batch_z, np.tile(prior[i,:],(nb_per_sample,1))],axis=1)))
        np.save('{}/data_pre_{}_all.npy'.format(self.save_dir, batch_idx),np.stack(data_gen))

class scMulReg(object):
    """single cell multiome autoencoder mapping with regulatory score
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(scMulReg, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.ee_net = BaseFullyConnectedNet(input_dim=params['e_dim'],output_dim = params['z_dim'], 
                                        model_name='ee_net', nb_units=params['ee_units'])
        self.ea_net = BaseFullyConnectedNet(input_dim=params['a_dim'],output_dim = params['z_dim'], 
                                        model_name='ea_net', nb_units=params['ea_units'])
        self.de_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['e_dim'],#+977 
                                        model_name='de_net', nb_units=params['de_units'])
        self.da_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['a_dim'], 
                                        model_name='da_net', nb_units=params['da_units'])
        self.reg_net = BaseFullyConnectedNet(input_dim=955,output_dim = 1, 
                                        model_name='reg_net', nb_units=params['reg_units'])
        self.optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)

        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(ee_net = self.ee_net,
                                   ea_net = self.ea_net,
                                   de_net = self.de_net,
                                   da_net = self.da_net,
                                   reg_net = self.reg_net,
                                   optimizer = self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
        
    def get_config(self):
        return {
                "params": self.params,
        }

    #@tf.function
    def train_step(self, data_exp, data_atac, 
        reg_dic, TF_gene_idx, TG_idx):
        """train step.
        """  
        with tf.GradientTape(persistent=True) as tape:
            #data_tf_exp = data_exp[:,self.params['e_dim']:]
            #data_exp = data_exp[:,:self.params['e_dim']]
            data_z_exp = self.ee_net(data_exp)
            data_z_atac = self.ea_net(data_atac)
            align_loss = tf.reduce_mean((data_z_exp - data_z_atac)**2)

            data_z = (data_z_exp+data_z_atac)/2
            data_exp_rec = self.de_net(data_z)
            #data_exp_rec = self.de_net(tf.concat([data_z,data_tf_exp],axis=1))
            data_atac_rec = self.da_net(data_z)

            #(nb_TFs, nb_TGs, bs)
            #reg_score = self.get_reg_score_v2(reg_dic, data_exp_rec, data_atac_rec, TFs, TGs, all_genes, query_genes)
            reg_score = self.get_reg_score_v2(reg_dic, data_exp_rec, data_atac_rec, TF_gene_idx, TG_idx)
            #[ nb_TGs, nb_TFs  bs]

            reg_score = tf.transpose(reg_score,[2,0,1])
            reg_score = tf.reshape(reg_score,[-1, len(TF_gene_idx)])
            reg_pre = self.reg_net(reg_score) #(bs*nb_TGs, 1)
            reg_pre = tf.reshape(reg_pre,[tf.shape(data_exp)[0],len(TG_idx)])
            #reg_loss = tf.reduce_mean((reg_pre - tf.gather(data_exp_rec, axis=1,indices=gene_idx))**2)
            reg_loss = tf.reduce_mean((reg_pre - tf.gather(data_exp, axis=1,indices=TG_idx))**2)

            exp_rec_loss = tf.reduce_mean((data_exp - data_exp_rec)**2)
            atac_rec_loss = tf.reduce_mean((data_atac - data_atac_rec)**2)

            total_loss = exp_rec_loss + atac_rec_loss + self.params['alpha']*align_loss + self.params['beta']*reg_loss

        # Calculate the gradients
        gradients = tape.gradient(total_loss, self.ee_net.trainable_variables+self.ea_net.trainable_variables+
                        self.de_net.trainable_variables+self.da_net.trainable_variables+self.reg_net.trainable_variables)

        # Apply the gradients to the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.ee_net.trainable_variables+self.ea_net.trainable_variables+
                        self.de_net.trainable_variables+self.da_net.trainable_variables+self.reg_net.trainable_variables))
        return exp_rec_loss, atac_rec_loss, align_loss, reg_loss, total_loss

    @tf.function
    def train_ae_step(self, data_exp, data_atac):
        """train step with out regulatory score.
        """  
        with tf.GradientTape(persistent=True) as tape:
            data_z_exp = self.ee_net(data_exp)
            data_z_atac = self.ea_net(data_atac)
            align_loss = tf.reduce_mean((data_z_exp - data_z_atac)**2)

            data_z = (data_z_exp+data_z_atac)/2
            data_exp_rec = self.de_net(data_z)
            data_atac_rec = self.da_net(data_z)

            exp_rec_loss = tf.reduce_mean((data_exp - data_exp_rec)**2)
            atac_rec_loss = tf.reduce_mean((data_atac - data_atac_rec)**2)

            total_loss = exp_rec_loss + atac_rec_loss + self.params['alpha']*align_loss
            
        # Calculate the gradients
        gradients = tape.gradient(total_loss, self.ee_net.trainable_variables+self.ea_net.trainable_variables+
                        self.de_net.trainable_variables+self.da_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.ee_net.trainable_variables+self.ea_net.trainable_variables+
                        self.de_net.trainable_variables+self.da_net.trainable_variables))
        return exp_rec_loss, atac_rec_loss, align_loss, total_loss


    def train(self, data=None, data_file=None, sep='\t', header=0, normalize=False,
            batch_size=16, n_iter=50000, batches_per_eval=5000, batches_per_save=500,
            startoff=0, verbose=1, save_format='txt'):
        """
        Train a scMulReg model given the input data.
        
        Parameters
        ----------
        data
            List object containing the triplet data [X,Y,V]. Default: ``None``.
        data_file
            Str object denoting the path to the input file (csv, txt, npz).
        sep
            Str object denoting the delimiter for the input file. Default: ``\t``.
        header
            Int object denoting row number(s) to use as the column names. Default: ``0``.
        normalize
            Bool object denoting whether apply standard normalization to covariates. Default: ``False``.
        batch_size
            Int object denoting the batch size in training. Default: ``32``.
        n_iter
            Int object denoting the training iterations. Default: ``30000``.
        batches_per_eval
            Int object denoting the number of iterations per evaluation. Default: ``500``.
        batches_per_save
            Int object denoting the number of iterations per save. Default: ``10000``.
        startoff
            Int object denoting the beginning iterations to jump without save and evaluation. Defalt: ``0``.
        verbose
            Bool object denoting whether showing the progress bar. Default: ``False``.
        save_format
            Str object denoting the format (csv, txt, npz) to save the results. Default: ``txt``.
        """
        
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        if data is None and data_file is None:
            self.data_sampler = Dataset_selector(self.params['dataset'])(batch_size=batch_size)
        elif data is not None:
            if len(data) != 2:
                print('Data imcomplete error, please provide pair-wise (X, Y) in a list or tuple.')
                sys.exit()
            else:
                self.data_sampler = Base_sampler(x=data[0],y=data[1],batch_size=batch_size,normalize=normalize)
        else:
            data = parse_file(data_file, sep, header, normalize)
            self.data_sampler = Base_sampler(x=data[0],y=data[1],batch_size=batch_size,normalize=normalize)

        #train reg_score
        import pickle as pkl
        import scanpy as sc
        #reg_info = pkl.load(open('../reg_info.pkl', 'rb'))
        reg_dic = pkl.load(open('../reg_dic.pkl', 'rb'))
        TFs = [item.strip() for item in open('../motifscan/TFs.txt').readlines()]
        TGs = [item.strip() for item in open('../data/pbmc10k/TGs.txt').readlines()]
        all_genes = sc.read('../data/pbmc10k/adata_rna.h5').var['gene_ids'].index.to_list()
        TF_gene_idx = [all_genes.index(item) for item in TFs]
        TG_idx_all = list(reg_dic.keys())
        import random
        #train autoencoders
        for batch_idx in range(n_iter+1):
            batch_exp, batch_atac = self.data_sampler.next_batch()
            if self.params['use_reg'] and batch_idx % self.params['n_freq'] == 0:
                #print('error')
                TG_idx = random.sample(TG_idx_all, 500)
                exp_rec_loss, atac_rec_loss, align_loss,reg_loss, total_loss = self.train_step(batch_exp, batch_atac,
                                        reg_dic, TF_gene_idx, TG_idx)
                print('reg_loss %.4f'%reg_loss)
            else:
                exp_rec_loss, atac_rec_loss, align_loss, total_loss = self.train_ae_step(batch_exp, batch_atac)

            if batch_idx % batches_per_eval == 0:
                loss_contents = '''Iteration [%d] : exp_rec_loss [%.4f], atac_rec_loss [%.4f], align_loss [%.4f], total_loss [%.4f]''' \
                %(batch_idx, exp_rec_loss, atac_rec_loss, align_loss, total_loss)
                if verbose:
                    print(loss_contents)
                self.evaluate(self.data_sampler.load_all(), batch_idx)

    def evaluate(self, data, batch_idx):
        data_exp, data_atac = data
        #data_tf_exp = data_exp[:,self.params['e_dim']:]
        #data_exp = data_exp[:,:self.params['e_dim']]

        data_z_exp = self.ee_net.predict(data_exp, batch_size=1)
        #np.save('{}/data_embeds_at_{}.npy'.format(self.save_dir, batch_idx),data_z_exp)
        
        data_z_atac = np.vstack([self.ea_net.predict_on_batch(data_atac[i:(i+1),:]) 
                    for i in np.arange(len(data_atac))])
        #data_z_atac = self.ea_net.predict_on_batch(data_atac)
        np.savez('{}/data_embeds_at_{}.npz'.format(self.save_dir, batch_idx),data_z_exp,data_z_atac)

    def get_reg_score_numpy(self, reg_info, rna_mat, atac_mat, TFs, TGs, all_genes, query_genes):
        assert len(TFs) == len(reg_info)
        reg_score = []
        for i in range(len(reg_info)):
            reg_tf_score = []
            for tg in query_genes:
                tg_idx = TGs.index(tg)
                idx = np.where(reg_info[i][:,0]==tg_idx)[0]
                if len(idx)>0:
                    sub_reg_info = reg_info[i][idx,:]
                    peak_idx = sub_reg_info[:,1].astype('int8')
                    s = np.mean(atac_mat[:,peak_idx] * sub_reg_info[:,2], axis=1)
                    s *= rna_mat[:,all_genes.index(TFs[i])]
                else:
                    s = np.zeros(rna_mat.shape[0])
                reg_tf_score.append(s)
            reg_tf_score = np.stack(reg_tf_score).T
            reg_score.append(reg_tf_score)
        return np.stack(reg_score)

    def get_reg_score(self, reg_info, rna_mat, atac_mat, TFs, TGs, all_genes, query_genes):
        assert len(TFs) == len(reg_info)
        reg_score = []
        for i in range(len(reg_info)):
            reg_tf_score = []
            for tg in query_genes:
                tg_idx = TGs.index(tg)
                idx = np.where(reg_info[i][:,0]==tg_idx)[0]
                if len(idx)>0:
                    sub_reg_info = reg_info[i][idx,:]
                    peak_idx = sub_reg_info[:,1].astype('int32')
                    s = tf.reduce_mean(tf.gather(atac_mat, axis=1,indices=peak_idx) * sub_reg_info[:,2], axis=1)
                    s *= rna_mat[:,all_genes.index(TFs[i])]
                else:
                    s = tf.zeros(tf.shape(rna_mat)[0])
                reg_tf_score.append(s)
            reg_tf_score = tf.stack(reg_tf_score)
            reg_score.append(reg_tf_score)
        return tf.stack(reg_score)

    def get_reg_score_v2(self, reg_dic, rna_mat, atac_mat, TF_gene_idx, TG_idx):
        reg_score = []
        for j in range(len(TG_idx)):
            reg_score_tf = []
            for i in range(len(TF_gene_idx)):
                if TF_gene_idx[i] in reg_dic[TG_idx[j]]:
                    sub_reg_info = reg_dic[TG_idx[j]][TF_gene_idx[i]]
                    peak_idx = sub_reg_info[:,0].astype('int32')
                    #s = tf.reduce_mean(tf.gather(atac_mat, axis=1,indices=peak_idx) * sub_reg_info[:,1], axis=1)
                    s = tf.reduce_mean(tf.gather(atac_mat, axis=1,indices=peak_idx), axis=1)
                    s *= rna_mat[:,TF_gene_idx[i]]
                else:
                    s = tf.zeros(tf.shape(rna_mat)[0])
                reg_score_tf.append(s)
            reg_score.append(tf.stack(reg_score_tf))
        return tf.stack(reg_score)
                